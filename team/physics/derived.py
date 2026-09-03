"""The teacher's derived quantities, priced for a trigger-level student.

`team/teacher/common.py` gives the teacher two things the 2k student does not have:

  * `particle_features(rich=True)` -- 6 extra numbers per candidate, all O(n):
      ln(pt/HT)/4, log1p(E)/8, cos/sin(Δφ to the leading candidate), Δη/2, |dxy|.
  * `pair_features()` -- 4 numbers for every ordered pair (i, j):
      ln ΔR, ln kT, ln z, ln m^2.  16x16x4 = 1,024 numbers per event.

Note which teacher actually published the 0.9151 / tt 0.8261 soft targets: it is
`ds_big_s0`, a **BigDeepSet with rich=True** -- per-candidate features only.  The
pairwise block is `ParTLite`'s, and no ParT run has been published yet.  So the
teacher-student gap is currently *not* evidence about relational information; it
is evidence about 6 cheap per-candidate features, mean+max pooling, 72k params
and 40 epochs.  This module separates those, and prices the pairwise part in case
a ParT teacher does land.

Two representations are built here, both consumable by a small student:

  RICH_*   per-candidate extras, appended as channels to X (the student's phi()
           grows from 5 to 11 inputs: +24% phi MACs, no new sequential stages).
  PAIR_*   pairwise quantities restricted to the leading k candidates, flattened
           to event-level scalars appended to F (k=4 -> 6 pairs x 4 = 24 numbers,
           computed once per event, after the pool -- free in the phi budget).
"""

from __future__ import annotations

import itertools

import numpy as np

try:                                  # imported as a package: physics.derived
    from .features import decode, _dphi
except ImportError:                   # run with physics/ on sys.path
    from features import decode, _dphi

PT_LOG_SCALE = 8.0
N_RICH = 11        # matches common.N_RICH_PART


def rich_particles(X: np.ndarray) -> np.ndarray:
    """(N, P, 5) cache tensor -> (N, P, 11), numerically matching common.particle_features.

    Channel order is the teacher's: the 5 cached features, then
    ln(pt/HT)/4, log1p(E)/8, cos Δφ_lead, sin Δφ_lead, Δη_lead/2, |dxy|/2.
    """
    return _rich(X.astype(np.float64), *decode(X))


def rich_from_raw(pt, eta, phi, dxy):
    """Same 11 channels from *physical* arrays, for data.py's streaming path.

    The first five channels are rebuilt with data.preprocess's constants rather
    than read back out of a cache, so this does not depend on a cache existing.
    """
    from data import preprocess   # local import: physics/ must not need pyarrow

    mask = pt > 0.0
    return _rich(preprocess(pt, eta, phi, dxy).astype(np.float64),
                 pt * mask, eta, phi, dxy, mask)


def _rich(X: np.ndarray, pt, eta, phi, dxy, mask) -> np.ndarray:
    m = mask.astype(np.float64)
    ht = np.maximum((pt * m).sum(1, keepdims=True), 1e-6)
    lnz = np.log(np.maximum(pt, 1e-6) / ht) * m / 4.0
    lnE = np.log1p(pt * np.cosh(eta)) / PT_LOG_SCALE * m
    dphi = _dphi(phi, phi[:, :1])
    deta = (eta - eta[:, :1]) / 2.0
    x = X
    out = np.stack([x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4],
                    lnz, lnE, np.cos(dphi), np.sin(dphi), deta, np.abs(x[..., 2])], axis=-1)
    return (out * m[..., None]).astype(np.float32)


RICH_CHANNELS = ("log_pt", "eta", "dxy", "cos_phi", "sin_phi",
                 "lnz", "lnE", "cos_dphi_lead", "sin_dphi_lead", "deta_lead", "abs_dxy")


def _pair_quantities(pt, eta, phi, i, j, eps=1e-8):
    """ParT's four pair quantities for one (i, j), matching common.pair_features."""
    deta = eta[:, i] - eta[:, j]
    dphi = _dphi(phi[:, i], phi[:, j])
    dr = np.sqrt(deta * deta + dphi * dphi)
    pti, ptj = pt[:, i], pt[:, j]
    ptmin = np.minimum(pti, ptj)
    m2 = 2.0 * pti * ptj * (np.cosh(deta) - np.cos(dphi))
    return dict(
        lndR=np.log(np.maximum(dr, eps)),
        lnkt=np.log(np.maximum(ptmin * dr, eps)),
        lnz=np.log(np.maximum(ptmin / np.maximum(pti + ptj, eps), eps)),
        lnm2=np.log(np.maximum(m2, eps)),
    )


PAIR_QUANTITIES = ("lndR", "lnkt", "lnz", "lnm2")


def pair_scalars(X: np.ndarray, k: int = 4, quantities=PAIR_QUANTITIES):
    """Pairwise quantities among the leading k candidates -> (names, (N, C)).

    k(k-1)/2 unordered pairs x len(quantities) numbers, event-level: computed once,
    injected after the pool, never replicated inside phi().  k=4 is 6 pairs.
    """
    pt, eta, phi, dxy, mask = decode(X)
    names, cols = [], []
    for i, j in itertools.combinations(range(k), 2):
        q = _pair_quantities(pt, eta, phi, i, j)
        for name in quantities:
            names.append(f"p{i+1}{j+1}_{name}")
            cols.append(q[name])
    return names, np.stack(cols, axis=1).astype(np.float32)


def pair_pooled(X: np.ndarray, quantities=PAIR_QUANTITIES):
    """Mean / min / max over *all* 120 unordered pairs -- the full-pairwise ceiling.

    Not a trigger proposal (it needs the whole 16x16 table); it is the yardstick
    that says how much the leading-k restriction gives up.
    """
    pt, eta, phi, dxy, mask = decode(X)
    P = pt.shape[1]
    per = {q: [] for q in quantities}
    for i, j in itertools.combinations(range(P), 2):
        qq = _pair_quantities(pt, eta, phi, i, j)
        for q in quantities:
            per[q].append(qq[q])
    names, cols = [], []
    for q in quantities:
        A = np.stack(per[q], axis=1)
        for stat, v in (("mean", A.mean(1)), ("min", A.min(1)), ("max", A.max(1)),
                        ("std", A.std(1))):
            names.append(f"pool_{q}_{stat}")
            cols.append(v)
    return names, np.stack(cols, axis=1).astype(np.float32)


def rich_summaries(X: np.ndarray):
    """Event-level summaries of the 6 rich per-candidate channels.

    The cheap way to ask "is the information in this channel useful" without
    widening phi(): for each derived channel, its value on the leading 4
    candidates plus the mean over all 16.
    """
    R = rich_particles(X).astype(np.float64)
    names, cols = [], []
    for c in range(5, N_RICH):
        ch = RICH_CHANNELS[c]
        for p in range(4):
            names.append(f"{ch}_c{p+1}")
            cols.append(R[:, p, c])
        names.append(f"{ch}_mean")
        cols.append(R[:, :, c].mean(1))
    return names, np.stack(cols, axis=1).astype(np.float32)


def families(X: np.ndarray):
    """{family name: (column names, (N, C) array)} for the ranking driver."""
    out = {}
    rn, rv = rich_summaries(X)
    for c in range(5, N_RICH):
        ch = RICH_CHANNELS[c]
        idx = [i for i, n in enumerate(rn) if n.startswith(ch + "_")]
        out[f"rich:{ch}"] = ([rn[i] for i in idx], rv[:, idx])
    out["rich:ALL"] = (rn, rv)

    for q in PAIR_QUANTITIES:
        n4, v4 = pair_scalars(X, 4, (q,))
        out[f"pair4:{q}"] = (n4, v4)
    out["pair4:ALL"] = pair_scalars(X, 4)
    out["pair6:ALL"] = pair_scalars(X, 6)
    out["pairfull:pooled"] = pair_pooled(X)
    return out
