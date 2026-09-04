"""Data for the c3 attention student.

Reads c1's caches (`team/cache/<tag>/{X,F,y,group}.npy`) and, when `rich=True`,
appends c2's six derived per-candidate channels with `physics.derived.rich_particles`
-- the same tensor as `cache/<tag>_rich`, but derived from the base cache so it also
works for `train1M`, for which no rich cache was built. The result is memoised under
`team/attn/cache/` (gitignored).

Firmware note: channels 5-10 (ln pt/HT, ln E, cos/sin Δφ_lead, Δη_lead, |dxy|) are
treated as *inputs*, exactly like the base five (log1p(pt)/8, η/4, cos φ, sin φ,
dxy/2) already are -- they are fixed per-candidate functions of quantities the L1
object already carries, computed once upstream of the network, not layers we synthesize.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TEAM = Path(__file__).resolve().parent.parent
CACHE = TEAM / "cache"
MEMO = Path(__file__).resolve().parent / "cache"
GROUP_NAME = {0: "QCD", 1: "HH_4b", 2: "tt", 3: "Wjets"}   # data.py:GROUP_ID


def _rich_path(tag: str) -> Path:
    return MEMO / f"{tag}_rich_X.npy"


def rich_tensor(tag: str, X: np.ndarray) -> np.ndarray:
    """(N,16,5) -> (N,16,11), memoised on disk."""
    p = _rich_path(tag)
    if p.exists():
        R = np.load(p, mmap_mode="r")
        if len(R) == len(X):
            return np.ascontiguousarray(R)
        print(f"  stale rich memo for {tag} ({len(R)} vs {len(X)} rows) -- rebuilding")
    import sys

    sys.path.insert(0, str(TEAM))
    from physics.derived import rich_particles

    MEMO.mkdir(parents=True, exist_ok=True)
    out = np.empty((len(X), X.shape[1], 11), dtype=np.float32)
    step = 250_000
    for i in range(0, len(X), step):
        out[i:i + step] = rich_particles(X[i:i + step])
    np.save(p, out)
    return out


def load(tag: str, rich: bool = True):
    """-> X (N,16,C), F (N,11), y (N,), group (N,), meta"""
    d = CACHE / tag
    X = np.load(d / "X.npy")
    F = np.load(d / "F.npy")
    y = np.load(d / "y.npy").astype(np.float32).ravel()
    g = np.load(d / "group.npy").ravel()
    meta = json.loads((d / "meta.json").read_text())
    if rich and X.shape[2] == 5:
        X = rich_tensor(tag, X)
    elif not rich and X.shape[2] > 5:
        X = np.ascontiguousarray(X[..., :5])
    return X, F, y, g, meta


def soft_targets(tag: str, prefix: str = "soft_targets"):
    """-> (logits, meta). `prefix` selects which published teacher to distil from:
    "soft_targets" is whatever the teacher lane currently publishes, "soft_targets_dsbig"
    pins the original BigDeepSet. 4-class files are reduced to the binary
    HH-vs-rest logit with the recipe their own meta prescribes."""
    d = TEAM / "teacher"
    meta = json.loads((d / f"{prefix}_meta.json").read_text())
    z = np.load(d / f"{prefix}_{tag}.npy").astype(np.float32)
    if z.ndim == 2 and z.shape[1] > 1:
        from scipy.special import logsumexp
        order = meta["class_order"]
        sig = order.index("HH_4b")
        rest = [i for i in range(len(order)) if i != sig]
        z = z[:, sig] - logsumexp(z[:, rest], axis=1)
    return z.ravel(), meta


def auc_report(scores, y, group, title):
    """Same table the other lanes print (train.py:auc_report), duplicated so this
    lane does not import c1's training module."""
    from sklearn.metrics import roc_auc_score

    scores = np.asarray(scores).ravel()
    auc = roc_auc_score(y, scores)
    print(f"\n=== {title} ===")
    print(f"  events: {len(y)}  signal: {int(y.sum())}  background: {int((1 - y).sum())}")
    print(f"  BINARY AUC (signal vs all background): {auc:.5f}")
    per_group, sig = {}, y == 1
    for gid, name in sorted(GROUP_NAME.items()):
        if name == "HH_4b":
            continue
        sel = sig | ((y == 0) & (group == gid))
        if (y[sel] == 0).sum() == 0:
            continue
        per_group[name] = float(roc_auc_score(y[sel], scores[sel]))
        print(f"    vs {name:<6s}: AUC {per_group[name]:.5f}  ({int((y[sel] == 0).sum())} bkg events)")
    eff, bkg = {}, scores[y == 0]
    for rej in (0.99, 0.999):
        thr = np.quantile(bkg, rej)
        eff[str(rej)] = float((scores[sig] > thr).mean())
        print(f"    signal eff @ {rej * 100:g}% bkg rejection: {eff[str(rej)]:.4f}")
    return float(auc), per_group, eff
