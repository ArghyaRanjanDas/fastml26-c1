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


def sample_groups(Xs: np.ndarray) -> np.ndarray | None:
    """Recover the background label of each closure-sample row by matching it back to
    `cache/eval100k` -- the sample carries no `group` array. The raw 5-channel particle
    tensor is the key; it is unique per event. Returns None unless every row matches."""
    Xe = np.load(TEAM / "cache" / "eval100k" / "X.npy")
    ge = np.load(TEAM / "cache" / "eval100k" / "group.npy").ravel()
    Xe = np.ascontiguousarray(Xe.reshape(len(Xe), -1))
    index = {Xe[i].tobytes(): int(ge[i]) for i in range(len(Xe))}
    flat = np.ascontiguousarray(Xs.reshape(len(Xs), -1))
    out = np.array([index.get(flat[i].tobytes(), -1) for i in range(len(flat))])
    matched = (out >= 0).mean()
    print(f"  closure sample matched to eval100k: {matched * 100:.1f}%")
    return out if matched == 1.0 else None


def sample_inputs(rich: bool):
    """The DeepSet lane's closure sample, in this lane's input format."""
    d = np.load(TEAM / "export" / "eval_sample.npz")
    X, F, y = d["X"], d["F"], d["y"]
    g = sample_groups(np.ascontiguousarray(X))
    if rich:
        sys.path.insert(0, str(TEAM))
        from physics.derived import rich_particles
        X = rich_particles(X)
    return np.ascontiguousarray(X), np.ascontiguousarray(F), y, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--outdir")
    ap.add_argument("--write", action="store_true", help="write the HLS project and tar it")
    ap.add_argument("--part", default="xcu200-fsgd2104-2-e")
    ap.add_argument("--clock", type=float, default=5.0)
    ap.add_argument("--io-type", default="io_parallel")
    ap.add_argument("--strategy", default="distributed_arithmetic",
                    help="'distributed_arithmetic' (DSP-free, requires reuse 1) or 'latency'")
    ap.add_argument("--reuse", type=int, default=1)
    ap.add_argument("--full-eval", action="store_true",
                    help="also run the whole eval100k slice through the HLS model (slow)")
    args = ap.parse_args()

    model, summary = load_run(args.run)
    rich = summary.get("rich", True)
    Xs, Fs, ys, gs = sample_inputs(rich)

    yk = np.asarray(model.predict([Xs, Fs], batch_size=4096, verbose=0)).ravel()

    outdir = args.outdir or f"/tmp/hls_attn_{args.run}"
    t0 = time.perf_counter()
    # Precision comes from the HGQ2 bit-exact pass, so the only thing this config
    # carries is the arithmetic strategy. distributed_arithmetic replaces every
    # multiplier with an adder tree (0 DSP) and requires reuse factor 1.
    # No manual precision: HGQ2 carries its own per-parameter widths through hls4ml's
    # bit-exact pass, and overriding them would throw away what QAT trained.
    hls_config = {"Model": {"Strategy": args.strategy, "ReuseFactor": args.reuse}}
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend="Vitis", io_type=args.io_type, output_dir=outdir,
        part=args.part, clock_period=args.clock, hls_config=hls_config)
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
                  strategy=args.strategy, reuse=args.reuse,
                  n_sample=int(len(ys)), max_abs_diff=float(dif.max()),
                  mean_abs_diff=float(dif.mean()),
                  auc_keras_sample=float(roc_auc_score(ys, yk)),
                  auc_hls_sample=float(roc_auc_score(ys, yh)))

    if gs is not None:
        per_k, per_h, sig = {}, {}, ys == 1
        for gid, name in sorted(attn_data.GROUP_NAME.items()):
            if name == "HH_4b":
                continue
            sel = sig | ((ys == 0) & (gs == gid))
            if (ys[sel] == 0).sum() == 0:
                continue
            per_k[name] = float(roc_auc_score(ys[sel], yk[sel]))
            per_h[name] = float(roc_auc_score(ys[sel], yh[sel]))
            print(f"    vs {name:<6s}: keras {per_k[name]:.4f} -> HLS {per_h[name]:.4f}"
                  f"  ({int((ys[sel] == 0).sum())} bkg events)")
        result.update(per_background_auc_keras_sample=per_k,
                      per_background_auc_hls_sample=per_h)
    else:
        print("    (could not match the sample rows back to eval100k -- no per-group AUC)")

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
        projects = TEAM / "fpga" / "projects"
        projects.mkdir(parents=True, exist_ok=True)
        tar = projects / f"{args.run}.tar.gz"
        subprocess.run(["tar", "czf", str(tar), "-C", str(Path(outdir).parent),
                        Path(outdir).name], check=True)
        mb = tar.stat().st_size / 1e6
        print(f"wrote project in {time.perf_counter()-t0:.1f}s -> {tar} ({mb:.1f} MB)")
        result.update(project=str(tar), project_mb=mb)

    (RUNS / f"{args.run}_hls.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
