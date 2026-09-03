"""Shared pieces for the teacher lane: cache I/O, derived features, metrics.

Everything the teacher sees is a deterministic function of the SAME cache
tensors the student sees (X = [N,16,5] per-candidate features, F = [N,11]
event features, see team/data.py), so the teacher's logits are valid soft
targets for a student that consumes exactly those tensors.

Per-candidate cache layout (data.preprocess):
    x[...,0] = log1p(pt) / 8      x[...,1] = eta / 4      x[...,2] = clip(dxy, +-2) / 2
    x[...,3] = cos(phi)           x[...,4] = sin(phi)
Padded slots are all-zero (none exist in the current caches: every event has 16).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent
CACHE_ROOT = TEAM / "cache"
RUNS = HERE / "runs"

# Must match team/data.py (PT_LOG_SCALE, ETA_SCALE, DXY_CLIP) -- asserted in load_cache.
PT_LOG_SCALE = 8.0
ETA_SCALE = 4.0
DXY_CLIP = 2.0
GROUP_ID = {"QCD": 0, "HH_4b": 1, "tt": 2, "Wjets": 3}
GROUP_NAME = {v: k for k, v in GROUP_ID.items()}
CACHE_TAGS = ("train1M", "train300k", "eval100k")

N_RAW_PART = 5
N_RICH_PART = 11
N_PAIR = 4


def _check_constants():
    """Assert the scalings above equal the ones in team/data.py (parsed, not imported:
    data.py pulls in awkward/pyarrow which this env need not have)."""
    import ast
    tree = ast.parse((TEAM / "data.py").read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("PT_LOG_SCALE", "ETA_SCALE", "DXY_CLIP", "GROUP_ID"):
                found[name] = ast.literal_eval(node.value)
    assert found == dict(PT_LOG_SCALE=PT_LOG_SCALE, ETA_SCALE=ETA_SCALE, DXY_CLIP=DXY_CLIP, GROUP_ID=GROUP_ID), found


def load_cache(tag: str):
    """Return X (N,16,5) f32, F (N,11) f32, y (N,) f32 in {0,1}, group (N,) int8, meta."""
    d = CACHE_ROOT / tag
    meta = json.loads((d / "meta.json").read_text())
    X = np.load(d / "X.npy")
    F = np.load(d / "F.npy")
    y = np.load(d / "y.npy").astype(np.float32)
    g = np.load(d / "group.npy")
    assert len(X) == len(F) == len(y) == len(g) == meta["n_events"], tag
    assert X.shape[1:] == (16, 5) and F.shape[1] == 11, (X.shape, F.shape)
    return X, F, y, g, meta


# ------------------------------------------------------------- derived features

def decode(x: torch.Tensor):
    """(B,P,5) cached features -> pt [GeV], eta, phi, dxy [cm-ish], mask (B,P). fp32."""
    x = x.float()
    mask = x[..., 0] > 0
    pt = torch.expm1(x[..., 0] * PT_LOG_SCALE) * mask
    eta = x[..., 1] * ETA_SCALE
    phi = torch.atan2(x[..., 4], x[..., 3])
    dxy = x[..., 2] * DXY_CLIP
    return pt, eta, phi, dxy, mask


def _wrap(dphi: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(dphi), torch.cos(dphi))


def particle_features(x: torch.Tensor, rich: bool = True) -> torch.Tensor:
    """Per-candidate input for the teacher.

    raw  : the 5 cached features, untouched (what the student sees).
    rich : those 5 plus 6 derived, all O(1):
           ln(pt/HT)/4, log1p(E)/8, cos/sin(dphi to leading cand), deta to leading cand / 2, |dxy|/2.
    Zero-padded slots stay all-zero.
    """
    if not rich:
        return x.float()
    with torch.autocast(device_type="cuda", enabled=False):
        pt, eta, phi, dxy, mask = decode(x)
        m = mask.float()
        ht = (pt * m).sum(1, keepdim=True).clamp_min(1e-6)
        lnz = torch.log(pt.clamp_min(1e-6) / ht) * m / 4.0
        lnE = torch.log1p(pt * torch.cosh(eta)) / PT_LOG_SCALE * m
        dphi = _wrap(phi - phi[:, :1])
        deta = (eta - eta[:, :1]) / 2.0
        x = x.float()
        feats = torch.stack([x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4],
                             lnz, lnE, torch.cos(dphi), torch.sin(dphi), deta, x[..., 2].abs()], -1)
        return feats * m[..., None]


def pair_features(x: torch.Tensor, eps: float = 1e-8):
    """ParT pairwise interaction features for every ordered pair (i,j).

    Returns f (B,P,P,4) = [ln dR, ln kT, ln z, ln m^2] (massless candidates) and the
    boolean pair mask (B,P,P) with the diagonal and padded slots removed; f is zero
    where the mask is false (ParT's remove_self_pair convention).
    """
    with torch.autocast(device_type="cuda", enabled=False):
        pt, eta, phi, _, mask = decode(x)
        P = pt.shape[1]
        deta = eta[:, :, None] - eta[:, None, :]
        dphi = _wrap(phi[:, :, None] - phi[:, None, :])
        dr = torch.sqrt(deta * deta + dphi * dphi)
        pti, ptj = pt[:, :, None], pt[:, None, :]
        ptmin = torch.minimum(pti, ptj)
        lndelta = torch.log(dr.clamp_min(eps))
        lnkt = torch.log((ptmin * dr).clamp_min(eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp_min(eps)).clamp_min(eps))
        m2 = 2.0 * pti * ptj * (torch.cosh(deta) - torch.cos(dphi))
        lnm2 = torch.log(m2.clamp_min(eps))
        f = torch.stack([lndelta, lnkt, lnz, lnm2], -1)
        pm = mask[:, :, None] & mask[:, None, :]
        pm = pm & ~torch.eye(P, dtype=torch.bool, device=x.device)
        return f * pm[..., None], pm


# ------------------------------------------------------------------- metrics

def auc_report(logits: np.ndarray, y: np.ndarray, group: np.ndarray, title: str, quiet=False):
    """Same definitions as team/train.py::auc_report (overall AUC, AUC vs each
    background group = signal vs that group only, signal eff at fixed bkg rejection)."""
    auc = float(roc_auc_score(y, logits))
    sig = y == 1
    per_group = {}
    for gid, name in sorted(GROUP_NAME.items()):
        if name == "HH_4b":
            continue
        sel = sig | ((y == 0) & (group == gid))
        if (y[sel] == 0).sum() == 0:
            continue
        per_group[name] = float(roc_auc_score(y[sel], logits[sel]))
    eff, bkg = {}, logits[y == 0]
    for rej in (0.99, 0.999):
        thr = np.quantile(bkg, rej)
        eff[str(rej)] = float((logits[sig] > thr).mean())
    if not quiet:
        print(f"\n=== {title} ===")
        print(f"  events: {len(y)}  signal: {int(sig.sum())}  background: {int((~sig).sum())}")
        print(f"  BINARY AUC (signal vs all background): {auc:.5f}")
        for name, v in per_group.items():
            print(f"    vs {name:<6s}: AUC {v:.5f}")
        for rej, v in eff.items():
            print(f"    signal eff @ {float(rej) * 100:g}% bkg rejection: {v:.4f}")
    return auc, per_group, eff


def quick_auc(logits: np.ndarray, y: np.ndarray, group: np.ndarray):
    """(overall AUC, AUC vs tt) -- the two numbers tracked per epoch."""
    auc = float(roc_auc_score(y, logits))
    sel = (y == 1) | (group == GROUP_ID["tt"])
    return auc, float(roc_auc_score(y[sel], logits[sel]))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
