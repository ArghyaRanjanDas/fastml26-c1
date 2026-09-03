"""Per-layer activation and weight ranges for an export -- fixed-point sizing.

ap_fixed<W,I> represents roughly +-2^(I-1).  ap_fixed<16,6> is +-32, so ANY
weight, bias, accumulator or activation beyond +-32 saturates and the network
degrades regardless of how many fractional bits it has.  This walks the exported
model exactly as verify_export.py does and reports what actually has to fit.
"""
import argparse, json
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--sample", required=True)
a = ap.parse_args()

spec = json.load(open(a.json)); sd = torch.load(a.weights, map_location="cpu"); npz = np.load(a.sample)
n_phi, n_rho = len(spec["phi"]), len(spec["rho"])
keys = [k for k in sd if k.endswith("weight")]
mats = [(sd[k].numpy().T, sd[k.replace("weight", "bias")].numpy()) for k in keys]
names = [f"phi{i}" for i in range(n_phi)] + [f"rho{i}" for i in range(n_rho)] + ["out"]

X = npz["X"].astype(np.float32)
F = npz["F"].astype(np.float32) if spec["n_event_features"] else None
relu = lambda v: np.maximum(v, 0.0)

def rng(v): return float(np.abs(v).max())
def fits(v, I): return "ok" if v < 2 ** (I - 1) else "OVER"

print(f"model {a.json}   {spec['n_particles']} particles, {spec['n_event_features']} event features\n")
print(f"{'tensor':<26} {'max|.|':>12}   {'<16,6>=+-32':>11} {'<18,8>=+-128':>12}")
print("-" * 68)
print(f"{'INPUT particles':<26} {rng(X):>12.4f}   {fits(rng(X),6):>11} {fits(rng(X),8):>12}")
if F is not None:
    print(f"{'INPUT event feats':<26} {rng(F):>12.4f}   {fits(rng(F),6):>11} {fits(rng(F),8):>12}")

h = X
for i, (W, b) in enumerate(mats):
    if i == n_phi:
        h = h.mean(axis=1)
        print(f"{'  [mean pool]':<26} {rng(h):>12.4f}   {fits(rng(h),6):>11} {fits(rng(h),8):>12}")
        if F is not None:
            h = np.concatenate([h, F], axis=1)
            print(f"{'  [concat pool+event]':<26} {rng(h):>12.4f}   {fits(rng(h),6):>11} {fits(rng(h),8):>12}")
    z = h @ W + b
    for label, v in ((f"{names[i]} weight", rng(W)), (f"{names[i]} bias", rng(b)),
                     (f"{names[i]} preact", rng(z))):
        print(f"{label:<26} {v:>12.4f}   {fits(v,6):>11} {fits(v,8):>12}")
    h = relu(z) if i < len(mats) - 1 else z

print()
worst = max(max(rng(W), rng(b)) for W, b in mats)
print(f"worst weight/bias magnitude anywhere: {worst:.4f}  "
      f"-> needs ap_fixed<_,{int(np.ceil(np.log2(worst)))+1}> integer bits minimum")

if F is not None:
    print(f"\nevent features as the exported model sees them (standardized, x0.2 folded into rho0):")
    print(f"{'feature':<16} {'min':>9} {'max':>9} {'mean':>8} {'std':>7}")
    for i, n in enumerate(spec["event_feature_names"]):
        v = F[:, i]
        print(f"{n:<16} {v.min():>9.4f} {v.max():>9.4f} {v.mean():>8.4f} {v.std():>7.4f}")
    print(f"\nsame features after the x0.2 the folded rho0 weights apply: "
          f"max|.| = {rng(F)*0.2:.4f}")
