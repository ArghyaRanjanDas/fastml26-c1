"""Simulate ap_fixed<W,I> through an exported model, in numpy.

Reproduces what hls4ml does to the network so fixed-point problems can be found
and fixed here, without occupying the synthesis box.  hls4ml's default ap_fixed
mode is AP_TRN/AP_WRAP -- truncation and *wrapping* overflow -- so a value past
the integer range does not saturate, it wraps to the opposite sign, which is why
overflow destroys AUC rather than gently degrading it.  Both modes are offered.

  python quantsim.py --json export/model_2041.json --weights export/model_2041.pt \
                     --sample export/eval_sample.npz --bits 16,6 18,8 22,10 28,12
"""
import argparse, json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--sample", required=True)
ap.add_argument("--bits", nargs="+", default=["16,6", "18,8", "22,10", "28,12"])
ap.add_argument("--overflow", default="wrap", choices=["wrap", "sat"])
ap.add_argument("--per-layer", action="store_true",
                help="give each tensor its own integer-bit count sized to its measured "
                     "range, at fixed total width (hls4ml granularity='name')")
a = ap.parse_args()

spec = json.load(open(a.json)); sd = torch.load(a.weights, map_location="cpu"); npz = np.load(a.sample)
n_phi, n_rho = len(spec["phi"]), len(spec["rho"])
keys = [k for k in sd if k.endswith("weight")]
mats = [(sd[k].numpy().T.astype(np.float64), sd[k.replace("weight", "bias")].numpy().astype(np.float64))
        for k in keys]
X = npz["X"].astype(np.float64)
F = npz["F"].astype(np.float64) if spec["n_event_features"] else None
y, ref = npz["y"], npz["scores"].astype(np.float64)


def q(v, W, I, mode):
    """Quantize to ap_fixed<W,I>: step 2^(I-W), truncation, wrap or saturate."""
    step = 2.0 ** (I - W)
    lo, hi = -(2.0 ** (I - 1)), 2.0 ** (I - 1) - step
    v = np.floor(v / step) * step                      # AP_TRN
    if mode == "sat":
        return np.clip(v, lo, hi)
    span = 2.0 ** I                                    # AP_WRAP
    return (v - lo) % span + lo


def ibits(v, headroom=1):
    """Integer bits needed to hold max|v| without overflow."""
    m = float(np.abs(v).max())
    return max(2, int(np.ceil(np.log2(m + 1e-12))) + 1 + headroom)


def run(W, I, mode, per_layer=False):
    II = (lambda v: ibits(v)) if per_layer else (lambda v: I)
    h = q(X, W, II(X), mode)
    for i, (Wt, b) in enumerate(mats):
        if i == n_phi:
            h = q(h.mean(axis=1), W, II(h), mode)
            if F is not None:
                h = np.concatenate([h, q(F, W, II(F), mode)], axis=1)
        z = h @ q(Wt, W, II(Wt), mode) + q(b, W, II(b), mode)
        z = q(z, W, II(z), mode)
        h = np.maximum(z, 0.0) if i < len(mats) - 1 else z
    return 1.0 / (1.0 + np.exp(-h.ravel()))


print(f"{a.json}   float AUC = {roc_auc_score(y, ref):.5f}   (overflow mode: {a.overflow})\n")
print(f"{'precision':<18} {'range':>12} {'AUC':>9} {'loss':>9}")
print("-" * 52)
base = roc_auc_score(y, ref)
for spec_s in a.bits:
    W, I = (int(v) for v in spec_s.split(","))
    auc = roc_auc_score(y, run(W, I, a.overflow, a.per_layer))
    print(f"ap_fixed<{W},{I}>{'':<6} {'+-' + str(2**(I-1)):>12} {auc:>9.5f} {auc - base:>+9.5f}")
