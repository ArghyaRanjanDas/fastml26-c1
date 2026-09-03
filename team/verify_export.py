"""Check an export against the mapping team/fpga/synth.py actually performs.

synth.py builds the DeepSet in Keras and loads our torch state_dict positionally:

    keys = [k for k in sd if k.endswith("weight")]
    W = sd[wk].numpy().T                       # torch Linear (out,in) -> keras (in,out)
    Conv1D gets W[None, :, :]

so an off-by-one in layer order or a missing transpose silently produces a
working-but-wrong firmware.  This reimplements that mapping in numpy, runs the
5000 exported eval events through it, and compares against the scores stored in
the npz.  No hls4ml or TensorFlow needed -- it is pure arithmetic.

  python verify_export.py --json export/model_2041.json --weights export/model_2041.pt \
                          --sample export/eval_sample.npz
"""
import argparse, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--sample", required=True)
a = ap.parse_args()

spec = json.load(open(a.json))
sd = torch.load(a.weights, map_location="cpu")
npz = np.load(a.sample)

n_phi, n_rho = len(spec["phi"]), len(spec["rho"])
keys = [k for k in sd if k.endswith("weight")]
assert len(keys) == n_phi + n_rho + 1, f"expected {n_phi + n_rho + 1} Linear layers, got {keys}"

# exactly synth.py's mapping
mats = [(sd[k].numpy().T, sd[k.replace("weight", "bias")].numpy()) for k in keys]

X = npz["X"].astype(np.float32)
assert X.shape[1] == spec["n_particles"], \
    f"json says n_particles={spec['n_particles']} but sample is {X.shape[1]}"
assert X.shape[2] == spec["n_features"]

relu = lambda v: np.maximum(v, 0.0)
h = X
for W, b in mats[:n_phi]:                      # Conv1D kernel-1 == per-particle Linear
    h = relu(h @ W + b)

# GlobalAveragePooling1D, or mean+max concatenated
h = (np.concatenate([h.mean(axis=1), h.max(axis=1)], axis=1)
     if spec.get("pool") == "meanmax" else h.mean(axis=1))
if spec["n_event_features"]:
    F = npz["F"].astype(np.float32)
    assert F.shape[1] == spec["n_event_features"]
    h = np.concatenate([h, F], axis=1)         # Concatenate([pool, event])
for W, b in mats[n_phi:n_phi + n_rho]:
    h = relu(h @ W + b)
W, b = mats[-1]
score = 1.0 / (1.0 + np.exp(-(h @ W + b))).ravel()

ref = npz["scores"].astype(np.float64)
diff = np.abs(score - ref).max()
print(f"layers mapped      : {keys}")
print(f"n_particles        : {spec['n_particles']}   n_event_features: {spec['n_event_features']}")
print(f"max|numpy - torch| : {diff:.3e}")
print(f"AUC (numpy replica): {roc_auc_score(npz['y'], score):.5f}")
print(f"AUC (stored)       : {roc_auc_score(npz['y'], ref):.5f}")
print("PASS" if diff < 1e-5 else "FAIL -- export does not match synth.py's mapping")
raise SystemExit(0 if diff < 1e-5 else 1)
