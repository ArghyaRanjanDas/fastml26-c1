"""Cross-layer equalization: make an exported model fit in narrow fixed point.

ReLU is positive-homogeneous, so for adjacent layers scaling output channel i of
layer k by s and input column i of layer k+1 by 1/s leaves the function exactly
unchanged.  That freedom is what lets us move magnitude between layers until
every weight, bias and activation fits a target range -- without retraining and
without changing a single output.

Why it is needed here: folding the pooled BatchNorm into the first rho Linear
produces weights up to ~184 (the BN scale gamma/sqrt(var+eps) blows up for
pooled channels with near-zero variance), while the phi activations independently
reach ~115.  hls4ml's default ap_fixed is AP_WRAP, so anything past the integer
range wraps sign instead of clipping, which is why AUC collapses rather than
degrading gently.

Reference: Nagel et al., "Data-Free Quantization Through Weight Equalization and
Bias Correction" (ICCV 2019).
"""

from __future__ import annotations

import numpy as np
import torch


def _linears(model):
    """The Linear layers in forward order: phi..., rho..., out."""
    return ([m for m in model.phi if isinstance(m, torch.nn.Linear)]
            + [m for m in model.rho if isinstance(m, torch.nn.Linear)]
            + [model.out])


@torch.no_grad()
def channel_activation_max(model, X, F, use_evt, batch=4096):
    """max|pre-activation| per output channel for every layer except the last.

    Deliberately the *pre*-ReLU magnitude: a large negative pre-activation is
    invisible after ReLU but still has to be representable in the accumulator,
    and under AP_WRAP it wraps to a large positive value instead of clipping.
    Bounding only the post-ReLU output leaves exactly that hole -- it is what
    left phi2 at -33.5 on the first attempt.
    """
    layers = _linears(model)
    n_phi = len([m for m in model.phi if isinstance(m, torch.nn.Linear)])
    acc = [None] * (len(layers) - 1)
    for i in range(0, len(X), batch):
        h = X[i:i + batch]
        f = F[i:i + batch] if use_evt else None
        for k, lyr in enumerate(layers[:-1]):
            if k == n_phi:
                h = h.mean(dim=1)
                h = model.norm(h)
                if use_evt:
                    h = torch.cat([h, f * model.event_scale], dim=1)
            z = lyr(h)
            m = z.abs().amax(dim=tuple(range(z.dim() - 1))).cpu().numpy()
            acc[k] = m if acc[k] is None else np.maximum(acc[k], m)
            h = torch.relu(z)
    return acc


@torch.no_grad()
def equalize(model, X, F, use_evt, iters: int = 20, verbose: bool = True):
    """Rebalance per-channel scales in place.  The function is unchanged."""
    layers = _linears(model)
    n_phi = len([m for m in model.phi if isinstance(m, torch.nn.Linear)])
    acts = channel_activation_max(model, X, F, use_evt)

    for _ in range(iters):
        for k in range(len(layers) - 1):
            Wk, bk = layers[k].weight.data, layers[k].bias.data
            Wn = layers[k + 1].weight.data
            # phi's last layer feeds only the pooled columns of rho0; the event
            # feature columns that follow them are driven by the inputs, not by phi.
            cols = Wn[:, :Wk.shape[0]] if k == n_phi - 1 else Wn

            a = torch.as_tensor(acts[k], dtype=Wk.dtype, device=Wk.device)
            r1 = torch.maximum(torch.maximum(Wk.abs().amax(1), bk.abs()), a)
            r2 = cols.abs().amax(0)
            ok = (r1 > 1e-12) & (r2 > 1e-12)
            s = torch.where(ok, torch.sqrt(r2 / torch.clamp(r1, min=1e-12)),
                            torch.ones_like(r1))

            Wk *= s[:, None]
            bk *= s
            acts[k] = (a * s).cpu().numpy()
            cols /= s[None, :]

    if verbose:
        worst_w = max(float(l.weight.data.abs().max()) for l in layers)
        worst_b = max(float(l.bias.data.abs().max()) for l in layers)
        worst_a = max(float(m.max()) for m in acts)
        print(f"  after equalization: max|W| {worst_w:.3f}  max|b| {worst_b:.3f}  "
              f"max|preact| {worst_a:.3f}")
    return model
