"""hls4ml conversion + closure for the c3 attention student.

  KERAS_BACKEND=torch ~/hlsenv/bin/python synth_attn.py --run q_d16 [--write]

Rebuilds the quantized run, converts it with the Vitis backend for
`xcu200-fsgd2104-2-e` at 5 ns, and checks closure on `team/export/eval_sample.npz`
(the same 5,000 real eval events the DeepSet lane used) -- keras vs HLS max |diff|
and both AUCs. `--write` also emits the HLS project and tars it for the orchestrator.

io_parallel, not io_stream: hls4ml refuses heterogeneous activation quantization
under io_stream, and heterogeneous bit widths are the whole point of HGQ. See LOG.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras                                        # noqa: E402
import hgq2_compat                                  # noqa: E402,F401
import hls4ml                                       # noqa: E402
import attn_data                                    # noqa: E402
from model import Cfg, build                        # noqa: E402

RUNS = HERE / "runs"


def load_run(tag: str):
    summary = json.loads((RUNS / f"{tag}_summary.json").read_text())
    cfg = Cfg(**summary["cfg"])
    if summary["quantized"]:
        from hgq.config import LayerConfigScope, QuantizerConfigScope
        with QuantizerConfigScope(q_type='kbi', place='datalane', overflow_mode='SAT_SYM'), \
             QuantizerConfigScope(q_type='kbi', place='weight', overflow_mode='SAT_SYM'), \
             LayerConfigScope(enable_ebops=True, beta0=summary.get("beta0", 0.0)):
            model = build(cfg, quantized=True)
    else:
        model = build(cfg, quantized=False)
    model.load_weights(RUNS / f"{tag}.weights.h5")
    return model, summary


def sample_inputs(rich: bool):
    """The DeepSet lane's closure sample, in this lane's input format."""
    d = np.load(TEAM / "export" / "eval_sample.npz")
    X, F, y = d["X"], d["F"], d["y"]
    if rich:
        sys.path.insert(0, str(TEAM))
        from physics.derived import rich_particles
        X = rich_particles(X)
    return np.ascontiguousarray(X), np.ascontiguousarray(F), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--outdir")
    ap.add_argument("--write", action="store_true", help="write the HLS project and tar it")
    ap.add_argument("--part", default="xcu200-fsgd2104-2-e")
    ap.add_argument("--clock", type=float, default=5.0)
    ap.add_argument("--io-type", default="io_parallel")
    ap.add_argument("--full-eval", action="store_true",
                    help="also run the whole eval100k slice through the HLS model (slow)")
    args = ap.parse_args()

    model, summary = load_run(args.run)
    rich = summary.get("rich", True)
    Xs, Fs, ys = sample_inputs(rich)

    yk = np.asarray(model.predict([Xs, Fs], batch_size=4096, verbose=0)).ravel()

    outdir = args.outdir or f"/tmp/hls_attn_{args.run}"
    t0 = time.perf_counter()
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend="Vitis", io_type=args.io_type, output_dir=outdir,
        part=args.part, clock_period=args.clock)
    hls_model.compile()
    print(f"converted + compiled in {time.perf_counter()-t0:.1f}s -> {outdir}")

    yh = np.asarray(hls_model.predict([Xs, Fs])).ravel()

    from sklearn.metrics import roc_auc_score
    dif = np.abs(yk - yh)
    print(f"\nclosure on export/eval_sample.npz ({len(ys)} real eval events)")
    print(f"  max |keras - hls| = {dif.max():.6g}   mean |Δ| = {dif.mean():.6g}")
    print(f"  AUC keras = {roc_auc_score(ys, yk):.5f}")
    print(f"  AUC hls   = {roc_auc_score(ys, yh):.5f}")

    result = dict(run=args.run, part=args.part, clock=args.clock, io_type=args.io_type,
                  n_sample=int(len(ys)), max_abs_diff=float(dif.max()),
                  mean_abs_diff=float(dif.mean()),
                  auc_keras_sample=float(roc_auc_score(ys, yk)),
                  auc_hls_sample=float(roc_auc_score(ys, yh)))

    if args.full_eval:
        Xe, Fe, ye, ge, _ = attn_data.load(summary["eval_tag"], rich=rich)
        yke = np.asarray(model.predict([Xe, Fe], batch_size=16384, verbose=0)).ravel()
        yhe = np.asarray(hls_model.predict([np.ascontiguousarray(Xe),
                                            np.ascontiguousarray(Fe)])).ravel()
        a_k, pg_k, _ = attn_data.auc_report(yke, ye, ge, f"{args.run} keras -- full eval100k")
        a_h, pg_h, eff_h = attn_data.auc_report(yhe, ye, ge, f"{args.run} HLS -- full eval100k")
        result.update(auc_keras_eval=a_k, auc_hls_eval=a_h,
                      per_background_auc_keras=pg_k, per_background_auc_hls=pg_h,
                      signal_eff_hls=eff_h,
                      max_abs_diff_eval=float(np.abs(yke - yhe).max()))

    if args.write:
        t0 = time.perf_counter()
        hls_model.write()
        tar = HERE / f"hls_attn_{args.run}.tar.gz"
        subprocess.run(["tar", "czf", str(tar), "-C", str(Path(outdir).parent),
                        Path(outdir).name], check=True)
        mb = tar.stat().st_size / 1e6
        print(f"wrote project in {time.perf_counter()-t0:.1f}s -> {tar} ({mb:.1f} MB)")
        result.update(project=str(tar), project_mb=mb)

    (RUNS / f"{args.run}_hls.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
