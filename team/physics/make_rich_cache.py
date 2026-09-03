"""Write a cache whose X carries the teacher's 11 per-candidate channels.

Pure transform of an existing cache -- no parquet, no re-read.  `team/train.py`
takes the per-candidate width from `Xtr.shape[2]`, so a cache built here trains
with c1's script unchanged:

    python physics/make_rich_cache.py --tag train300k
    python train.py --train-tag train300k_rich --eval-tag eval100k_rich ...

Optionally appends event-level columns to F as well (`--pair4` adds the 24
leading-4 pair scalars), which is the other half of the teacher's extras.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from data import CACHE_ROOT, cache_paths   # noqa: E402
import derived                             # noqa: E402


def standardize(v: np.ndarray):
    """Zero-mean / unit-variance with constants measured here and stored in meta.

    The pair scalars are logs of quantities spanning decades; feeding them to rho
    raw would swamp the 11 incumbent features, which are already standardized.
    """
    mu, sd = v.mean(0), v.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.clip((v - mu) / sd, -5.0, 5.0).astype(np.float32), mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suffix", default="_rich")
    ap.add_argument("--pair4", action="store_true", help="also append the 24 pair scalars to F")
    ap.add_argument("--keep-x", action="store_true",
                    help="leave X at the cached 5 channels (event-level extras only)")
    ap.add_argument("--norm-from", default=None,
                    help="reuse the pair standardization measured on this cache tag")
    a = ap.parse_args()

    Xp, Fp, yp, gp, mp = cache_paths(a.tag)
    meta = json.loads(mp.read_text())
    X = np.load(Xp)
    out = CACHE_ROOT / (a.tag + a.suffix)
    out.mkdir(parents=True, exist_ok=True)

    if a.keep_x:
        # symlink rather than copy: X is 192 MB for train300k and is byte-identical
        link = out / "X.npy"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Xp.resolve())
        R = X
        meta = dict(meta, tag=a.tag + a.suffix, derived_from=a.tag)
    else:
        R = np.empty((len(X), X.shape[1], derived.N_RICH), dtype=np.float32)
        for i in range(0, len(X), 100_000):
            R[i:i + 100_000] = derived.rich_particles(X[i:i + 100_000])
        np.save(out / "X.npy", R)
        meta = dict(meta, tag=a.tag + a.suffix, n_features=derived.N_RICH,
                    particle_channels=list(derived.RICH_CHANNELS), derived_from=a.tag)

    F = np.load(Fp)
    if a.pair4:
        names, chunks = None, []
        for i in range(0, len(X), 100_000):
            names, v = derived.pair_scalars(X[i:i + 100_000], 4)
            chunks.append(v)
        V = np.concatenate(chunks)
        if a.norm_from:
            ref = json.loads((CACHE_ROOT / a.norm_from / "meta.json").read_text())
            mu = np.array(ref["pair_standardize"]["mean"], dtype=np.float32)
            sd = np.array(ref["pair_standardize"]["std"], dtype=np.float32)
            V = np.clip((V - mu) / sd, -5.0, 5.0).astype(np.float32)
        else:
            V, mu, sd = standardize(V)
        F = np.concatenate([F, V], axis=1)
        meta["event_features"] = list(meta["event_features"]) + list(names)
        meta["pair_standardize"] = dict(mean=[float(x) for x in mu], std=[float(x) for x in sd])
    np.save(out / "F.npy", F)

    for src, name in ((yp, "y.npy"), (gp, "group.npy")):
        shutil.copyfile(src, out / name)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{out}: X {R.shape}  F {F.shape}")


if __name__ == "__main__":
    main()
