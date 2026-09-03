"""HGQ2 quantization-aware training of the deployable student, distillation on.

Follows the action brief: QDense phi over the candidate axis, mean+max pool,
concat the quantized event scalars, QDense rho; SAT_SYM on place='all', WRAP on
place='datalane', EBOPs enabled with a swept beta0; warm-started from the float
checkpoint and trained with KL+BCE against the team soft targets throughout
(not float -> distill -> quantize as separate stages); then the trace_minmax
calibration pass that the WRAP datalane requires.

Run in the HGQ2 venv:
  KERAS_BACKEND=torch ~/venv-hgq/bin/python hgq/qat_hgq.py --beta0 1e-5 --tag hgq_b5
"""
import argparse, json, os, sys
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
CACHE = os.path.join(os.path.dirname(HERE), "cache")
EXPORT = os.path.join(os.path.dirname(HERE), "export")


def load(tag):
    d = os.path.join(CACHE, tag)
    return (np.load(f"{d}/X.npy"), np.load(f"{d}/F.npy"),
            np.load(f"{d}/y.npy"), np.load(f"{d}/group.npy"),
            json.load(open(f"{d}/meta.json")))


def build(phi, rho, n_part, n_ch, n_evt, beta0, meanmax=True):
    from hgq.layers import QDense, QGlobalAveragePooling1D, QGlobalMaxPooling1D
    from hgq.config import LayerConfigScope, QuantizerConfigScope

    with QuantizerConfigScope(place="all", default_q_type="kbi", overflow_mode="SAT_SYM"), \
         QuantizerConfigScope(place="datalane", default_q_type="kif", overflow_mode="WRAP"), \
         LayerConfigScope(enable_ebops=True, beta0=beta0):
        x = keras.Input(shape=(n_part, n_ch), name="particles")
        e = keras.Input(shape=(n_evt,), name="event")
        h = x
        for i, w in enumerate(phi):
            h = QDense(w, activation="relu", name=f"phi{i}")(h)
        if meanmax:
            h = keras.layers.Concatenate(name="pool")(
                [QGlobalAveragePooling1D(name="pool_mean")(h),
                 QGlobalMaxPooling1D(name="pool_max")(h)])
        else:
            h = QGlobalAveragePooling1D(name="pool")(h)
        h = keras.layers.Concatenate(name="concat")([h, e])
        for i, w in enumerate(rho):
            h = QDense(w, activation="relu", name=f"rho{i}")(h)
        out = QDense(1, name="score")(h)      # logits; sigmoid applied outside
        return keras.Model([x, e], out)


def warm_start(model, pt_path):
    """Copy the folded PyTorch weights in. torch Linear (out,in) -> keras (in,out)."""
    sd = torch.load(pt_path, map_location="cpu")
    keys = [k for k in sd if k.endswith("weight")]
    qlayers = [l for l in model.layers if l.name.startswith(("phi", "rho", "score"))
               and not l.name.startswith("pool")]
    assert len(keys) == len(qlayers), f"{len(keys)} torch layers vs {len(qlayers)} keras"
    for k, lyr in zip(keys, qlayers):
        W = sd[k].numpy().T
        b = sd[k.replace("weight", "bias")].numpy()
        cur = lyr.get_weights()
        cur[0][...] = W
        cur[1][...] = b
        lyr.set_weights(cur)
    print(f"warm-started {len(keys)} layers from {os.path.basename(pt_path)}")


def ebops(model):
    tot = 0.0
    for l in model.layers:
        v = getattr(l, "ebops", None)
        if v is not None:
            try:
                tot += float(keras.ops.convert_to_numpy(v))
            except Exception:
                pass
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(EXPORT, "model_2777_rich.pt"))
    ap.add_argument("--train-tag", default="train1M_s")
    ap.add_argument("--eval-tag", default="eval100k_s")
    ap.add_argument("--soft-targets", default=os.path.join(os.path.dirname(HERE), "teacher"))
    ap.add_argument("--parent-tag", default="train1M")
    ap.add_argument("--phi", default="32,16,8")
    ap.add_argument("--rho", default="32,16")
    ap.add_argument("--beta0", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    dims = lambda s: [int(v) for v in s.split(",") if v.strip()]
    Xtr, Ftr, ytr, gtr, mtr = load(a.train_tag)
    Xev, Fev, yev, gev, mev = load(a.eval_tag)
    zt = np.load(os.path.join(a.soft_targets, f"soft_targets_{a.parent_tag}.npy")).astype("float32").ravel()
    assert len(zt) == len(Xtr)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xtr))
    nv = len(perm) // 10
    vi, ti = perm[:nv], perm[nv:]

    model = build(dims(a.phi), dims(a.rho), Xtr.shape[1], Xtr.shape[2], Ftr.shape[1], a.beta0)
    model.summary(print_fn=lambda s: print("  " + s))
    warm_start(model, a.init)

    T, alpha = a.temperature, a.alpha

    def loss_fn(y_true, y_pred):
        """y_true packs [hard label, teacher logit]; KL(T)+BCE, KD scaled by T^2."""
        y = y_true[:, 0:1]
        z = y_true[:, 1:2]
        bce = keras.ops.binary_crossentropy(y, y_pred, from_logits=True)
        kd = keras.ops.binary_crossentropy(
            keras.ops.sigmoid(z / T), y_pred / T, from_logits=True) * (T * T)
        return alpha * kd + (1.0 - alpha) * bce

    model.compile(optimizer=keras.optimizers.Adam(a.lr), loss=loss_fn, jit_compile=False)

    Ytr = np.stack([ytr[ti], zt[ti]], axis=1)
    Yva = np.stack([ytr[vi], zt[vi]], axis=1)

    def auc_of(X, F, y):
        p = keras.ops.convert_to_numpy(model.predict([X, F], batch_size=8192, verbose=0)).ravel()
        return roc_auc_score(y, p), p

    print(f"pre-QAT (warm-started, quantized forward): AUC {auc_of(Xev, Fev, yev)[0]:.5f}")
    # The EBOPs term keeps shrinking the network, so val AUC peaks partway through
    # and then decays; keep the best epoch rather than the last one. Selection is
    # on val AUC alone -- the EBOPs at that epoch is reported, not optimised for.
    best_va, best_w, best_ep = -1.0, None, -1
    for ep in range(1, a.epochs + 1):
        model.fit([Xtr[ti], Ftr[ti]], Ytr, batch_size=a.batch_size, epochs=1, verbose=0,
                  shuffle=True)
        va, _ = auc_of(Xtr[vi], Ftr[vi], ytr[vi])
        flag = ""
        if va > best_va:
            best_va, best_w, best_ep, flag = va, model.get_weights(), ep, "  *"
            best_eb = ebops(model)
        print(f"  epoch {ep:2d}/{a.epochs}  val_auc={va:.5f}  EBOPs={ebops(model):,.0f}{flag}",
              flush=True)
    if best_w is not None:
        model.set_weights(best_w)
        print(f"restored best epoch {best_ep} (val_auc {best_va:.5f}, EBOPs {best_eb:,.0f})")

    # WRAP on the datalane needs the ranges calibrated on held-out data
    from hgq.utils import trace_minmax
    trace_minmax(model, [Xtr[vi], Ftr[vi]], batch_size=8192, verbose=False)
    print(f"after calibration: EBOPs={ebops(model):,.0f}")

    auc, scores = auc_of(Xev, Fev, yev)
    print(f"\n=== HGQ2 QAT '{a.tag}' beta0={a.beta0:g} ===")
    print(f"  eval AUC {auc:.5f}")
    for gid, name in ((0, "QCD"), (2, "tt"), (3, "Wjets")):
        sel = (yev == 1) | (gev == gid)
        print(f"    vs {name:<6s} {roc_auc_score(yev[sel], scores[sel]):.5f}")

    os.makedirs(EXPORT, exist_ok=True)
    h5 = os.path.join(EXPORT, f"qat_{a.tag}.keras")
    model.save(h5)
    meta = dict(tag=a.tag, beta0=a.beta0, eval_auc=float(auc), ebops=ebops(model),
                best_epoch=best_ep, best_val_auc=float(best_va),
                phi=dims(a.phi), rho=dims(a.rho), n_particles=int(Xtr.shape[1]),
                n_features=int(Xtr.shape[2]), n_event_features=int(Ftr.shape[1]),
                pool="meanmax", init_from=os.path.basename(a.init),
                temperature=T, alpha=alpha, epochs=a.epochs,
                quant=dict(place_all="SAT_SYM", datalane="WRAP", enable_ebops=True),
                convert=dict(strategy="distributed_arithmetic", reuse_factor=1,
                             note="no manual precision -- HGQ2 carries per-layer bitwidths"))
    json.dump(meta, open(os.path.join(EXPORT, f"qat_{a.tag}.json"), "w"), indent=2)
    np.save(os.path.join(EXPORT, f"qat_{a.tag}_eval_scores.npy"), scores)
    print(f"  saved {h5}")


if __name__ == "__main__":
    main()
