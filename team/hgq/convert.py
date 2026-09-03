"""Convert an HGQ2 QAT export through hls4ml and hand the project to the FPGA lane.

Per the contract: Vitis backend, xcu200-fsgd2104-2-e, 5 ns, io_parallel,
Strategy=distributed_arithmetic, ReuseFactor=1, and **no manual precision** --
HGQ2 carries its own per-layer bit widths, so overriding them would discard the
thing QAT trained. Runs compile()+predict() closure on the real eval sample
(overall and per background), writes the project, and tars it.

  KERAS_BACKEND=torch ~/venv-hgq/bin/python hgq/convert.py --tag b1e-7
"""
import argparse, json, os, subprocess, sys
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
import hls4ml
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
EXPORT, PROJ = os.path.join(TEAM, "export"), os.path.join(TEAM, "fpga", "projects")


def bitwidths(model):
    """Per-layer effective bit widths that HGQ2 learned."""
    out = {}
    for l in model.layers:
        for attr in ("kq", "iq", "oq"):
            q = getattr(l, attr, None)
            if q is None:
                continue
            for f in ("bits", "k", "i", "f"):
                v = getattr(q, f, None)
                if v is None:
                    continue
                try:
                    a = keras.ops.convert_to_numpy(v)
                    out.setdefault(l.name, {})[f"{attr}.{f}"] = [float(a.min()), float(a.max())]
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sample", default=os.path.join(EXPORT, "eval_sample_rich.npz"))
    ap.add_argument("--part", default="xcu200-fsgd2104-2-e")
    ap.add_argument("--clock", type=float, default=5.0)
    a = ap.parse_args()

    model = keras.models.load_model(os.path.join(EXPORT, f"qat_{a.tag}.keras"))
    meta = json.load(open(os.path.join(EXPORT, f"qat_{a.tag}.json")))
    bw = bitwidths(model)
    print("per-layer learned bit widths (min..max over channels):")
    for k, v in bw.items():
        print(f"  {k:<10} " + "  ".join(f"{n}={lo:g}..{hi:g}" for n, (lo, hi) in v.items()))

    npz = np.load(a.sample)
    X, F, y, g = (npz["X"].astype("float32"), npz["F"].astype("float32"),
                  npz["y"], npz["group"])

    yk = keras.ops.convert_to_numpy(model.predict([X, F], batch_size=4096, verbose=0)).ravel()
    yk = 1.0 / (1.0 + np.exp(-yk))

    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
    for lc in cfg.get("LayerName", {}).values():
        lc["Strategy"] = "distributed_arithmetic"
        lc["ReuseFactor"] = 1
    out = os.path.join(PROJ, f"hls_{a.tag}")
    hm = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=out, backend="Vitis", part=a.part,
        clock_period=a.clock, io_type="io_parallel")
    hm.compile()
    yh = np.asarray(hm.predict([X, F])).ravel()
    yh = 1.0 / (1.0 + np.exp(-yh)) if yh.min() < 0 or yh.max() > 1 else yh

    res = {"keras_auc": float(roc_auc_score(y, yk)), "hls_auc": float(roc_auc_score(y, yh)),
           "max_abs_diff": float(np.abs(yk - yh).max())}
    print(f"\nclosure on {len(y)} eval events:")
    print(f"  keras AUC {res['keras_auc']:.5f}   hls AUC {res['hls_auc']:.5f}   "
          f"max|diff| {res['max_abs_diff']:.4f}")
    for gid, name in ((0, "QCD"), (2, "tt"), (3, "Wjets")):
        sel = (y == 1) | (g == gid)
        res[f"hls_auc_{name}"] = float(roc_auc_score(y[sel], yh[sel]))
        res[f"keras_auc_{name}"] = float(roc_auc_score(y[sel], yk[sel]))
        print(f"    vs {name:<6s} keras {res[f'keras_auc_{name}']:.5f}  "
              f"hls {res[f'hls_auc_{name}']:.5f}")

    hm.write()
    tar = os.path.join(PROJ, f"{a.tag}.tar.gz")
    subprocess.run(["tar", "czf", tar, "-C", PROJ, f"hls_{a.tag}"], check=True)
    mb = os.path.getsize(tar) / 1e6
    meta.update(hls=res, bitwidths=bw, project=tar, project_mb=round(mb, 1),
                convert=dict(strategy="distributed_arithmetic", reuse_factor=1,
                             part=a.part, clock_ns=a.clock, io_type="io_parallel",
                             manual_precision=False))
    json.dump(meta, open(os.path.join(EXPORT, f"qat_{a.tag}.json"), "w"), indent=2)
    print(f"\nproject: {tar} ({mb:.1f} MB){'  -- too big to commit' if mb >= 20 else ''}")


if __name__ == "__main__":
    main()
