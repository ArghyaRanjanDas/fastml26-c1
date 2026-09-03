"""Build the student cache with c2's selected extra inputs.

Per the round-3 priority list, taking exactly what c2 measured as worth its cost
and nothing more:

  X: 5 -> 11 per-candidate channels  (derived.rich_particles: the cached 5, plus
     ln(pt/HT)/4, log1p(E)/8, cos dphi_lead, sin dphi_lead, deta_lead/2, |dxy|/2)
  F: 11 -> 19 event features         (the incumbent 11, plus iso_lead_pt, n_iso,
     and ln dR of the 6 leading-4 pairs)

Deliberately NOT included: ln kT / ln m2 / ln z pair quantities (c2 measured them
as the same information as ln dR, +0.000 on top) and any full pairwise block
(pooling all 120 pairs keeps only +0.009 of +0.023 -- the value is in *which*
pair, not the ensemble).

This is a pure transform of an existing cache: no parquet re-read.  Extra-feature
standardization constants are measured on the train cache and reused for eval via
--norm-from, so train and eval share one frozen set, and they are written into the
cache meta so export.py can put them in the model json.

  python make_student_cache.py --tag train1M
  python make_student_cache.py --tag eval100k --norm-from train1M
"""

from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "physics"))
from data import CACHE_ROOT, cache_paths, EVENT_CLIP          # noqa: E402
from physics import derived                                    # noqa: E402
from physics.features import decode, _dphi                     # noqa: E402

EXTRA_NAMES = ("iso_lead_pt", "n_iso")
EXTRA_TRANSFORM = {"iso_lead_pt": "log1p", "n_iso": "linear"}


def iso_features(pt, eta, phi, mask):
    """iso_lead_pt and n_iso, matching physics/features.py.

    Only the 16x16 Delta-R cone part of c2's compute_raw is needed here; running
    the whole feature bank (which includes cone jet clustering) over 2M events to
    get two numbers would be wasteful.
    """
    dr2 = ((eta[:, :, None] - eta[:, None, :]) ** 2
           + _dphi(phi[:, :, None], phi[:, None, :]) ** 2)
    pair = mask[:, :, None] & mask[:, None, :]
    np.einsum("ijj->ij", dr2)[:] = 1e9
    cone = pair & (dr2 < 0.16)
    iso_sum = (pt[:, None, :] * cone).sum(2)
    iso = np.where(mask, iso_sum / np.maximum(pt, 1e-6), 1e9)
    hard = mask & (pt > 10.0)
    iso_h = np.where(hard, iso, 1e9)
    rows = np.arange(len(pt))
    best = np.argmin(iso_h, axis=1)
    has_hard = hard.any(1)
    return (np.where(has_hard, pt[rows, best], 0.0),
            (hard & (iso < 0.15)).sum(1).astype(np.float64))


def build_extra(X, chunk=100_000):
    """(N, P, 5) -> (N, 8) transformed-but-unstandardized extra event features."""
    names, out = None, []
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        pt, eta, phi, dxy, mask = decode(xb)
        ilp, nis = iso_features(pt, eta, phi, mask)
        pnames, pv = derived.pair_scalars(xb, 4, ("lndR",))
        names = list(EXTRA_NAMES) + list(pnames)
        cols = [np.log1p(np.maximum(ilp, 0.0)), nis.astype(np.float64)]
        out.append(np.concatenate([np.stack(cols, 1), pv], axis=1).astype(np.float32))
    return names, np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suffix", default="_s")
    ap.add_argument("--norm-from", default=None,
                    help="reuse the extra-feature standardization from this built cache")
    a = ap.parse_args()

    Xp, Fp, yp, gp, mp = cache_paths(a.tag)
    meta = json.loads(mp.read_text())
    X = np.load(Xp)
    out = CACHE_ROOT / (a.tag + a.suffix)
    out.mkdir(parents=True, exist_ok=True)

    R = np.empty((len(X), X.shape[1], derived.N_RICH), dtype=np.float32)
    for i in range(0, len(X), 100_000):
        R[i:i + 100_000] = derived.rich_particles(X[i:i + 100_000])
    np.save(out / "X.npy", R)

    names, V = build_extra(X)
    if a.norm_from:
        ref = json.loads((CACHE_ROOT / a.norm_from).joinpath("meta.json").read_text())
        mu = np.array(ref["extra_standardize"]["mean"], dtype=np.float32)
        sd = np.array(ref["extra_standardize"]["std"], dtype=np.float32)
    else:
        mu, sd = V.mean(0), V.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
    V = np.clip((V - mu) / sd, -EVENT_CLIP, EVENT_CLIP).astype(np.float32)

    F = np.concatenate([np.load(Fp), V], axis=1)
    np.save(out / "F.npy", F)
    for src, name in ((yp, "y.npy"), (gp, "group.npy")):
        shutil.copyfile(src, out / name)
    meta.update(tag=a.tag + a.suffix, derived_from=a.tag, n_features=derived.N_RICH,
                particle_channels=list(derived.RICH_CHANNELS),
                event_features=list(meta["event_features"]) + names,
                extra_event_features=names,
                extra_transform={n: EXTRA_TRANSFORM.get(n, "linear") for n in EXTRA_NAMES},
                extra_standardize=dict(mean=[float(x) for x in mu], std=[float(x) for x in sd]))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{out}: X {R.shape}  F {F.shape}  extra={names}")


if __name__ == "__main__":
    main()
