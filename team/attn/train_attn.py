"""Train the c3 attention student, with distillation from the teacher's logits.

Same recipe as c1's `team/distill.py` (KL at temperature T mixed with BCE, weight
alpha on the KD term), re-implemented in Keras 3 so the float model and the HGQ2
quantization-aware model are the same object. Runs on the A10 via
`KERAS_BACKEND=torch`.

  # float
  KERAS_BACKEND=torch ~/hlsenv/bin/python train_attn.py --tag a_d16 --epochs 30
  # QAT, warm-started from the float run
  KERAS_BACKEND=torch ~/hlsenv/bin/python train_attn.py --tag a_d16_q --quantized \
        --init-from a_d16 --beta0 3e-6 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras                                                    # noqa: E402
from keras import ops                                           # noqa: E402
import attn_data                                                # noqa: E402
from model import Cfg, build, transfer_weights, n_synth_params  # noqa: E402

RUNS = HERE / "runs"


def bce_logits(z, p):
    """BCE with logits, target p in [0,1]. mean over the batch."""
    return ops.mean(ops.maximum(z, 0.0) - z * p + ops.log1p(ops.exp(-ops.abs(z))))


def make_loss(T: float, alpha: float):
    def loss(y_true, y_pred):
        y = y_true[:, 0:1]
        zt = y_true[:, 1:2]
        z = y_pred
        kd = bce_logits(z / T, ops.sigmoid(zt / T)) * (T * T)
        hard = bce_logits(z, y)
        return alpha * kd + (1.0 - alpha) * hard
    return loss


class ValAUC(keras.callbacks.Callback):
    """Per-epoch AUC on the held-out slice of the training cache; keeps the best
    weights in memory (Keras' ModelCheckpoint cannot see a metric we compute here)."""

    def __init__(self, x, y, ebops=False, select_from=1, group=None, ckpt=None):
        super().__init__()
        self.x, self.y, self.ebops, self.group = x, y, ebops, group
        # Write the best-so-far weights to disk every time they improve, so a long QAT
        # run can be converted and handed to the synthesis box before it finishes.
        self.ckpt = ckpt
        self.partial = None      # set once the config is known, for in-flight summaries
        # Under an EBOPs penalty the first epochs have both the highest AUC and the
        # highest bit widths, so selecting on AUC over the whole run would hand back a
        # model that never paid the penalty. Only epochs from `select_from` on -- i.e.
        # after beta has finished ramping -- are eligible.
        self.select_from = select_from
        self.best, self.best_w, self.best_ebops, self.history = -1.0, None, None, []

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import roc_auc_score

        s = np.asarray(self.model.predict(self.x, batch_size=16384, verbose=0)).ravel()
        auc = float(roc_auc_score(self.y, s))
        rec = dict(epoch=epoch + 1, loss=float((logs or {}).get("loss", np.nan)), val_auc=auc)
        if self.group is not None:
            sig, pg = self.y == 1, {}
            for gid, name in sorted(attn_data.GROUP_NAME.items()):
                if name == "HH_4b":
                    continue
                sel = sig | ((self.y == 0) & (self.group == gid))
                pg[name] = float(roc_auc_score(self.y[sel], s[sel]))
            rec["val_official"] = attn_data.official_auc(pg)
            # model selection follows the scored metric, not the even-thirds one
            auc = rec["val_official"]
        if self.ebops:
            rec["ebops"] = float((logs or {}).get("ebops", np.nan))
        flag = ""
        if auc > self.best and epoch + 1 >= self.select_from:
            self.best, flag = auc, "  *"
            self.best_ebops = rec.get("ebops")
            self.best_w = [np.array(w) for w in self.model.get_weights()]
            if self.ckpt is not None:
                self.model.save_weights(self.ckpt)
        self.history.append(rec)
        if self.ckpt is not None and self.partial is not None and self.best_w is not None:
            import json as _json
            p = self.ckpt.parent / f"{self.ckpt.name.split('.weights')[0]}_summary.json"
            p.write_text(_json.dumps({**self.partial, "partial": True,
                                      "val_auc": self.best, "ebops": self.best_ebops,
                                      "history": self.history}, indent=2))
        eb = f"  ebops={rec['ebops']:.0f}" if self.ebops and np.isfinite(rec.get("ebops", np.nan)) else ""
        off = f"  val_off={rec['val_official']:.5f}" if rec.get("val_official") else ""
        print(f"  epoch {epoch+1:3d}  loss={rec['loss']:.4f}  val_auc={rec['val_auc']:.5f}"
              f"{off}{eb}{flag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--init-from", help="tag of a run whose weights warm-start this one")
    ap.add_argument("--beta0", type=float, default=0.0, help="HGQ EBOPs regularization strength")
    ap.add_argument("--beta-ramp", type=int, default=0,
                    help="epochs over which beta goes 0 -> beta0 (0 = constant beta0)")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--mlp-ratio", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--pool", default="meanmax", choices=["mean", "meanmax"])
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--no-rich", action="store_true", help="5 base channels instead of 11")
    ap.add_argument("--train-tag", default="train1M")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--teacher", default="soft_targets",
                    help="prefix of the published soft targets in team/teacher/")
    args = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    rich = not args.no_rich

    Xtr, Ftr, ytr, gtr, meta_tr = attn_data.load(args.train_tag, rich=rich)
    Xev, Fev, yev, gev, meta_ev = attn_data.load(args.eval_tag, rich=rich)
    zt, tmeta = attn_data.soft_targets(args.train_tag, args.teacher)
    print(f"teacher '{tmeta.get('source_run')}': eval AUC {tmeta.get('eval_auc')}, "
          f"per-group {tmeta.get('eval_per_group')}")
    assert len(zt) == len(Xtr), f"{len(zt)} soft targets vs {len(Xtr)} rows"
    if args.no_distill:
        zt = np.zeros_like(zt)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr))
    n_val = int(len(perm) * args.val_frac)
    va, tr = perm[:n_val], perm[n_val:]

    Yt = np.stack([ytr, zt], axis=1).astype(np.float32)
    if args.init_from:
        # QAT must reproduce the float run's topology exactly, or the warm start is
        # silently partial; take the shape from the source run, not from the flags.
        src_summary = json.loads((RUNS / f"{args.init_from}_summary.json").read_text())
        cfg = Cfg(**src_summary["cfg"])
        rich = src_summary.get("rich", True)
        if rich != (not args.no_rich):
            raise SystemExit(f"--init-from {args.init_from} was trained with rich={rich}")
    else:
        cfg = Cfg(n_channels=Xtr.shape[2], n_event=Ftr.shape[1], d=args.d, heads=args.heads,
                  blocks=args.blocks, mlp_ratio=args.mlp_ratio, head_dim=args.head_dim,
                  pool=args.pool, residual=not args.no_residual)

    if args.quantized:
        from hgq.config import LayerConfigScope, QuantizerConfigScope
        with QuantizerConfigScope(q_type='kbi', place='datalane', overflow_mode='SAT_SYM'), \
             QuantizerConfigScope(q_type='kbi', place='weight', overflow_mode='SAT_SYM'), \
             LayerConfigScope(enable_ebops=True, beta0=args.beta0):
            model = build(cfg, quantized=True)
    else:
        model = build(cfg, quantized=False)

    if args.init_from:
        src = build(cfg, quantized=False)
        src.load_weights(RUNS / f"{args.init_from}.weights.h5")
        moved = transfer_weights(src, model)
        print(f"warm start from '{args.init_from}': {moved} tensors transferred")

    nsyn = n_synth_params(model)
    print(f"{args.tag}: {model.count_params():,} keras params | {nsyn:,} synthesized "
          f"weights | cfg={cfg.to_dict()} | quantized={args.quantized}")

    steps = int(np.ceil(len(tr) / args.batch_size))
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr, decay_steps=args.epochs * steps, alpha=1e-3)
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4),
                  loss=make_loss(args.temperature, args.alpha))

    cb_val = ValAUC([Xtr[va], Ftr[va]], ytr[va], group=gtr[va], ebops=args.quantized,
                    ckpt=RUNS / f"{args.tag}.weights.h5",
                    select_from=(args.beta_ramp + 1) if (args.quantized and args.beta0) else 1)
    callbacks = [cb_val]
    if args.quantized:
        from hgq.utils.sugar import FreeEBOPs
        callbacks.insert(0, FreeEBOPs())
        if args.beta_ramp:
            from hgq.utils.sugar import BetaScheduler
            callbacks.append(BetaScheduler(
                lambda ep: 0.0 if ep < 1 else args.beta0 * min(1.0, ep / args.beta_ramp)))

    cb_val.partial = dict(run=args.tag, model="attn_student", cfg=cfg.to_dict(),
                          quantized=args.quantized, beta0=args.beta0, rich=rich,
                          params=int(nsyn), eval_tag=args.eval_tag)
    t0 = time.perf_counter()
    model.fit([Xtr[tr], Ftr[tr]], Yt[tr], batch_size=args.batch_size, epochs=args.epochs,
              shuffle=True, verbose=0, callbacks=callbacks)
    train_s = time.perf_counter() - t0

    if cb_val.best_w is not None:
        model.set_weights(cb_val.best_w)
    model.save_weights(RUNS / f"{args.tag}.weights.h5")

    ev = np.asarray(model.predict([Xev, Fev], batch_size=16384, verbose=0)).ravel()
    auc, per_group, eff = attn_data.auc_report(
        ev, yev, gev, f"attention student '{args.tag}' -- held-out EVAL slice")

    summary = dict(run=args.tag, model="attn_student", cfg=cfg.to_dict(),
                   quantized=args.quantized, beta0=args.beta0, init_from=args.init_from,
                   rich=rich, keras_params=int(model.count_params()), params=int(nsyn),
                   temperature=args.temperature, alpha=args.alpha, distill=not args.no_distill,
                   teacher=tmeta.get("source_run"), teacher_auc=tmeta.get("eval_auc"),
                   teacher_prefix=args.teacher,
                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                   train_tag=args.train_tag, eval_tag=args.eval_tag,
                   eval_auc=auc, official_auc=attn_data.official_auc(per_group),
                   val_auc=cb_val.best, per_background_auc=per_group,
                   signal_eff=eff, train_seconds=train_s, history=cb_val.history)
    if args.quantized:
        # EBOPs of the checkpoint actually kept. `hgq.utils.sugar.ebops` is a module,
        # not a function -- the number comes from the FreeEBOPs callback, which writes
        # `logs['ebops']` at each epoch end.
        summary["ebops"] = cb_val.best_ebops
        summary["ebops_final"] = cb_val.history[-1].get("ebops") if cb_val.history else None
    (RUNS / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(RUNS / f"{args.tag}_eval_scores.npy", ev)
    print(f"\n  {args.tag}: EVAL AUC = {auc:.5f}  ({nsyn:,} synthesized weights, "
          f"{train_s/60:.1f} min)")


if __name__ == "__main__":
    main()
