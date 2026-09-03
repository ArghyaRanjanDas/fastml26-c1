#!/usr/bin/env python
"""DeepSet -> hls4ml -> Vitis HLS (VU9P xcu200) resource/latency report.

Usage:
  python synth.py --dummy --phi 64 32 16 --rho 64 32 --nfeat 5 --npart 16 --nevt 0 --tag test
  python synth.py --export team/export/model_9k.json --weights team/export/model_9k.pt --tag m9k

The DeepSet is expressed exactly in Keras as Conv1D(kernel=1) per-particle phi,
GlobalAveragePooling1D, optional Concatenate with event-level features, Dense rho.
All are layers hls4ml's Vitis backend supports with io_parallel.
"""
import argparse, json, os, sys, time
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
import hls4ml

p = argparse.ArgumentParser()
p.add_argument("--dummy", action="store_true")
p.add_argument("--phi", type=int, nargs="+", default=[64, 32, 16])
p.add_argument("--rho", type=int, nargs="+", default=[64, 32])
p.add_argument("--nfeat", type=int, default=5)
p.add_argument("--npart", type=int, default=16)
p.add_argument("--nevt", type=int, default=0, help="event-level features concatenated after pooling")
p.add_argument("--export", help="architecture json from the pod")
p.add_argument("--weights", help="state_dict .pt (torch) from the pod")
p.add_argument("--tag", default="run")
p.add_argument("--precision", default="ap_fixed<16,6>")
p.add_argument("--reuse", type=int, default=1)
p.add_argument("--clock", type=float, default=5.0, help="ns; 5 ns = 200 MHz")
p.add_argument("--part", default="xcu200-fsgd2104-2-e")
p.add_argument("--no-synth", action="store_true")
a = p.parse_args()

if a.export:
    spec = json.load(open(a.export))
    a.phi, a.rho = spec["phi"], spec["rho"]; a.nfeat, a.npart, a.nevt = spec["n_features"], spec["n_particles"], spec.get("n_event_features", 0)

inp = keras.Input(shape=(a.npart, a.nfeat), name="particles")
x = inp
for i, w in enumerate(a.phi):
    x = L.Conv1D(w, 1, activation="relu", name=f"phi{i}")(x)
x = L.GlobalAveragePooling1D(name="pool")(x)
inputs = [inp]
if a.nevt:
    ev = keras.Input(shape=(a.nevt,), name="event"); inputs.append(ev)
    x = L.Concatenate(name="concat")([x, ev])
for i, w in enumerate(a.rho):
    x = L.Dense(w, activation="relu", name=f"rho{i}")(x)
out = L.Dense(1, activation="sigmoid", name="score")(x)
model = keras.Model(inputs, out)
model.summary(print_fn=lambda s: print("  " + s))
nparams = model.count_params(); print(f"params: {nparams}")

if a.weights:  # map torch state_dict -> keras (Linear weight is (out,in); Conv1D k=1 wants (1,in,out))
    if a.weights.endswith(".npz"):   # torch-free path: dict of numpy arrays exported on the training box
        z = np.load(a.weights); sd = {k: z[k] for k in z.files}
    else:
        import torch
        sd = {k: v.detach().cpu().numpy() for k, v in torch.load(a.weights, map_location="cpu").items()}
    keys = [k for k in sd if k.endswith("weight")]
    print("weight keys:", keys)
    li = 0
    for lyr in model.layers:
        if isinstance(lyr, (L.Conv1D, L.Dense)):
            wk = keys[li]; bk = wk.replace("weight", "bias"); W = np.asarray(sd[wk]).T; b = np.asarray(sd[bk])
            lyr.set_weights([W[None, :, :] if isinstance(lyr, L.Conv1D) else W, b]); li += 1
    print("weights loaded:", li, "layers")

cfg = hls4ml.utils.config_from_keras_model(model, granularity="name", default_precision=a.precision, default_reuse_factor=a.reuse)
outdir = os.path.expanduser(f"~/fastml26/hls_{a.tag}")
hm = hls4ml.converters.convert_from_keras_model(model, hls_config=cfg, output_dir=outdir, backend="Vitis", part=a.part, clock_period=a.clock, io_type="io_parallel")
hm.compile()
# closure check: prefer the exported eval sample (real preprocessed inputs + labels + float scores)
sample = os.path.join(os.path.dirname(a.export), "eval_sample.npz") if a.export else None
if a.weights and a.weights.endswith("_8p.npz"): sample = sample.replace("eval_sample.npz", "eval_sample_8p.npz")
if sample and os.path.exists(sample):
    z = np.load(sample); keys = z.files; print("eval sample keys:", keys)
    X = z[[k for k in keys if k.startswith("X")][0]].astype("float32"); F = z[[k for k in keys if k.startswith("F") or k=="event"][0]].astype("float32") if a.nevt else None
    y = z[[k for k in keys if k.startswith("y") or k=="labels"][0]]; s_ref = z[[k for k in keys if "score" in k][0]] if any("score" in k for k in keys) else None
    xs = [X] + ([F] if a.nevt else [])
else:
    xs = [np.random.rand(200, a.npart, a.nfeat).astype("float32")] + ([np.random.rand(200, a.nevt).astype("float32")] if a.nevt else []); y = s_ref = None
yk = model.predict(xs, verbose=0).ravel(); yh = hm.predict(xs if len(xs) > 1 else xs[0]).ravel()
print(f"keras-vs-hls max|diff| = {np.max(np.abs(yk - yh)):.4f}  mean|diff| = {np.mean(np.abs(yk - yh)):.4f}  (fixed-point {a.precision})")
if y is not None:
    from sklearn.metrics import roc_auc_score
    print(f"AUC on sample: keras {roc_auc_score(y, yk):.5f}  hls {roc_auc_score(y, yh):.5f}" + (f"  (stored float scores {roc_auc_score(y, s_ref):.5f})" if s_ref is not None else ""))
if a.no_synth: sys.exit(0)
t0 = time.time(); rep = hm.build(csim=False, synth=True, vsynth=False); dt = time.time() - t0
print(f"csynth done in {dt/60:.1f} min"); hls4ml.report.read_vivado_report(outdir)
r = rep.get("CSynthesisReport", rep)
summary = {k: r.get(k) for k in ("EstimatedClockPeriod", "BestLatency", "WorstLatency", "LUT", "FF", "DSP", "BRAM_18K", "URAM") if k in r}
summary.update(tag=a.tag, params=nparams, precision=a.precision, reuse=a.reuse, clock_ns=a.clock, part=a.part)
json.dump(summary, open(f"{outdir}/summary.json", "w"), indent=1); print("SUMMARY", json.dumps(summary))
