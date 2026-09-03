"""Physics features for the tt-bar problem (c2's lane).

Why this file exists
--------------------
The B1e_16p baseline is 0.930 against QCD and 0.972 against W+jets but only
0.759 against tt.  tt is the background that *looks* like HH->4b: both are
high-multiplicity, high-HT, b-flavoured final states.  What differs is the
*mass structure* (HH: two ~125 GeV bb pairs; tt: a 80 GeV W inside a 173 GeV
top) and, for the semi-/fully-leptonic tt modes, an isolated lepton plus a
neutrino, i.e. a genuine pT imbalance among the visible candidates.

Everything here is computed from the leading 16 PUPPI candidates only, because
that is all the trigger has.  No new parquet reads: the cached `X` produced by
`team/data.py` is an invertible transform of (pt, eta, phi, dxy), so `decode()`
recovers the physical quantities from the cache.

All functions are vectorized over events; a chunked driver is at the bottom.
"""

from __future__ import annotations

import numpy as np

# must match team/data.py
PT_LOG_SCALE = 8.0
ETA_SCALE = 4.0
DXY_CLIP = 2.0


def decode(X: np.ndarray):
    """Cached (N, P, 5) tensor -> physical (pt, eta, phi, dxy, mask).

    data.py stores log1p(pt)/8, eta/4, clip(dxy,+-2)/2, cos(phi), sin(phi) and
    zeroes every field of a padded slot, so the inverse is exact up to float32
    rounding (and up to the dxy clip, which saturates at |dxy| = 2 -- the p99 is
    0.65, so nothing real is lost).
    """
    pt = np.expm1(X[..., 0].astype(np.float64) * PT_LOG_SCALE)
    eta = X[..., 1].astype(np.float64) * ETA_SCALE
    dxy = X[..., 2].astype(np.float64) * DXY_CLIP
    phi = np.arctan2(X[..., 4].astype(np.float64), X[..., 3].astype(np.float64))
    mask = X[..., 0] > 0.0
    pt = pt * mask
    return pt, eta, phi, dxy, mask


def _dphi(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _p4(pt, eta, phi, mask):
    """Massless four-vectors, zero for masked slots."""
    m = mask.astype(np.float64)
    p = pt * m
    return (p * np.cos(phi), p * np.sin(phi), p * np.sinh(eta), p * np.cosh(eta))


def _mass(px, py, pz, E):
    return np.sqrt(np.maximum(E * E - px * px - py * py - pz * pz, 0.0))


# ------------------------------------------------------------------ jets

def cone_jets(pt, eta, phi, dxy, mask, R=0.4, n_jets=6):
    """Greedy pT-ordered cone clustering -- a cheap stand-in for anti-kT.

    Repeatedly take the hardest unclustered candidate as a seed and absorb every
    unclustered candidate within Delta R < R.  With 16 candidates and R = 0.4
    this is what anti-kT does anyway in all but pathological overlaps, and unlike
    anti-kT it is a fixed number of vectorized passes (no per-event loop), which
    matters both for the 2M-event scan here and for any firmware version later.

    Returns a dict of (N, n_jets) arrays sorted by descending jet pT.
    """
    N, P = pt.shape
    px, py, pz, E = _p4(pt, eta, phi, mask)
    adxy = np.abs(dxy) * mask

    avail = mask.copy()
    rank = np.where(avail, pt, -1.0)
    rows = np.arange(N)

    J = {k: np.zeros((N, n_jets)) for k in ("px", "py", "pz", "E", "sumdxy", "ptdxy", "n")}
    for j in range(n_jets):
        idx = np.argmax(rank, axis=1)
        seed_pt = rank[rows, idx]
        live = seed_pt > 0.0

        dr2 = (eta - eta[rows, idx][:, None]) ** 2 + _dphi(phi, phi[rows, idx][:, None]) ** 2
        member = avail & (dr2 < R * R) & live[:, None]

        J["px"][:, j] = (px * member).sum(1)
        J["py"][:, j] = (py * member).sum(1)
        J["pz"][:, j] = (pz * member).sum(1)
        J["E"][:, j] = (E * member).sum(1)
        J["sumdxy"][:, j] = (adxy * member).sum(1)
        J["ptdxy"][:, j] = (pt * adxy * member).sum(1)
        J["n"][:, j] = member.sum(1)

        avail = avail & ~member
        rank = np.where(avail, pt, -1.0)

    jpt = np.hypot(J["px"], J["py"])
    order = np.argsort(-jpt, axis=1)
    for k in J:
        J[k] = np.take_along_axis(J[k], order, axis=1)
    jpt = np.take_along_axis(jpt, order, axis=1)

    p = np.sqrt(J["px"] ** 2 + J["py"] ** 2 + J["pz"] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        jeta = np.arcsinh(np.divide(J["pz"], np.maximum(jpt, 1e-9)))
    jphi = np.arctan2(J["py"], J["px"])
    jm = _mass(J["px"], J["py"], J["pz"], J["E"])
    return dict(pt=jpt, eta=jeta, phi=jphi, m=jm, E=J["E"],
                px=J["px"], py=J["py"], pz=J["pz"], p=p,
                sumdxy=J["sumdxy"], ptdxy=J["ptdxy"], n=J["n"])


def _comb_mass(J, combo):
    """Invariant mass of a fixed jet combination, e.g. (0, 1)."""
    px = sum(J["px"][:, i] for i in combo)
    py = sum(J["py"][:, i] for i in combo)
    pz = sum(J["pz"][:, i] for i in combo)
    E = sum(J["E"][:, i] for i in combo)
    return _mass(px, py, pz, E)


M_W, M_TOP, M_H = 80.4, 172.5, 125.0


# -------------------------------------------------------------- the features

def compute(X: np.ndarray) -> dict:
    """(N, 16, 5) cached tensor -> {feature name: (N,) float32}."""
    return compute_raw(*decode(X))


def compute_raw(pt, eta, phi, dxy, mask) -> dict:
    """Physical (N, P) arrays -> {feature name: (N,) float32}.

    data.py calls this one directly, since it has the physical arrays already;
    compute() is the entry point for anything working off a built cache.
    """
    N, P = pt.shape
    m = mask.astype(np.float64)
    px, py, pz, E = _p4(pt, eta, phi, mask)
    adxy = np.abs(dxy) * m

    ht = (pt * m).sum(1)
    ht_safe = np.maximum(ht, 1e-6)
    f = {}

    # ---- missing-pT / imbalance (the neutrino side of leptonic tt) ----------
    sx, sy = px.sum(1), py.sum(1)
    mpt = np.hypot(sx, sy)
    mphi = np.arctan2(-sy, -sx)          # direction of the missing momentum
    f["mpt"] = mpt
    f["mpt_over_ht"] = mpt / ht_safe
    f["mpt_sig"] = mpt / np.sqrt(ht_safe)
    # min Delta phi between the missing-pT direction and the 4 hardest candidates:
    # real MET points away from the jets, mismeasurement points at them.
    dphi_lead = np.abs(_dphi(phi[:, :4], mphi[:, None]))
    dphi_lead = np.where(mask[:, :4], dphi_lead, np.pi)
    f["min_dphi_mpt"] = dphi_lead.min(1)

    # ---- isolation (the lepton side of leptonic tt) -------------------------
    dr2 = ((eta[:, :, None] - eta[:, None, :]) ** 2
           + _dphi(phi[:, :, None], phi[:, None, :]) ** 2)
    pair = mask[:, :, None] & mask[:, None, :]
    np.einsum("ijj->ij", dr2)[:] = 1e9          # drop self-pairs
    cone = pair & (dr2 < 0.16)
    iso_sum = (pt[:, None, :] * cone).sum(2)    # pT around each candidate, R<0.4
    with np.errstate(divide="ignore", invalid="ignore"):
        iso = np.where(mask, iso_sum / np.maximum(pt, 1e-6), 1e9)
    hard = mask & (pt > 10.0)
    iso_h = np.where(hard, iso, 1e9)
    best = np.argmin(iso_h, axis=1)
    rows = np.arange(N)
    has_hard = hard.any(1)
    f["iso_min"] = np.where(has_hard, np.minimum(iso_h[rows, best], 10.0), 10.0)
    f["iso_lead_pt"] = np.where(has_hard, pt[rows, best], 0.0)
    f["n_iso"] = (hard & (iso < 0.15)).sum(1).astype(np.float64)
    # transverse mass of (most isolated hard candidate, missing pT): the leptonic
    # W in tt piles up under 80 GeV, HH->4b has no such object.
    lpt, lphi = f["iso_lead_pt"], np.where(has_hard, phi[rows, best], 0.0)
    f["mt_lep_mpt"] = np.sqrt(np.maximum(2.0 * lpt * mpt * (1.0 - np.cos(_dphi(lphi, mphi))), 0.0))

    # ---- displacement / b-likeness -----------------------------------------
    f["n_dxy_p05"] = (mask & (np.abs(dxy) > 0.05)).sum(1).astype(np.float64)
    f["n_dxy_p20"] = (mask & (np.abs(dxy) > 0.20)).sum(1).astype(np.float64)
    f["ptw_dxy"] = (pt * adxy).sum(1) / ht_safe
    f["max_pt_dxy"] = (pt * adxy).max(1)
    f["dxy_lead4"] = adxy[:, :4].sum(1)

    # ---- global shape -------------------------------------------------------
    f["m6"] = _mass(px[:, :6].sum(1), py[:, :6].sum(1), pz[:, :6].sum(1), E[:, :6].sum(1))
    f["m8"] = _mass(px[:, :8].sum(1), py[:, :8].sum(1), pz[:, :8].sum(1), E[:, :8].sum(1))
    f["m16"] = _mass(px.sum(1), py.sum(1), pz.sum(1), E.sum(1))
    f["ht4_frac"] = (pt[:, :4] * m[:, :4]).sum(1) / ht_safe
    f["pt_ratio_41"] = pt[:, 3] / np.maximum(pt[:, 0], 1e-6)
    ptot = np.sqrt(px ** 2 + py ** 2 + pz ** 2).sum(1)
    f["centrality"] = ht / np.maximum(ptot, 1e-6)
    mean_eta = (pt * eta * m).sum(1) / ht_safe
    f["eta_spread"] = np.sqrt(np.maximum((pt * m * (eta - mean_eta[:, None]) ** 2).sum(1) / ht_safe, 0.0))
    # transverse sphericity from the 2x2 momentum tensor (closed form)
    sxx, syy, sxy = (px * px).sum(1), (py * py).sum(1), (px * py).sum(1)
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    l1, l2 = 0.5 * (tr + disc), 0.5 * (tr - disc)
    f["sphericity_T"] = 2.0 * l2 / np.maximum(l1 + l2, 1e-9)

    # ---- jets and mass structure -------------------------------------------
    J = cone_jets(pt, eta, phi, dxy, mask)
    jpt = J["pt"]
    f["n_jets15"] = (jpt > 15.0).sum(1).astype(np.float64)
    f["n_jets30"] = (jpt > 30.0).sum(1).astype(np.float64)
    f["jet1_m"] = J["m"][:, 0]
    f["jet_m_max"] = J["m"].max(1)
    f["jet_ptdxy_max"] = J["ptdxy"].max(1)
    f["n_bjets"] = (J["ptdxy"] > 2.0).sum(1).astype(np.float64)
    f["ht_jets4"] = jpt[:, :4].sum(1)

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4),
             (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
    mjj = np.stack([_comb_mass(J, c) for c in pairs], axis=1)
    live_pair = np.stack([(jpt[:, a] > 10.0) & (jpt[:, b] > 10.0) for a, b in pairs], axis=1)
    # W proxy: the dijet closest to 80.4 GeV.  tt has one by construction.
    f["dm_W"] = np.where(live_pair.any(1),
                         np.where(live_pair, np.abs(mjj - M_W), 1e9).min(1), 200.0)
    f["m_jj_maxpt"] = mjj[:, 0]

    trips = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 4), (0, 2, 4), (0, 3, 4),
             (1, 2, 4), (1, 3, 4), (2, 3, 4), (0, 1, 5), (0, 2, 5), (1, 2, 5), (3, 4, 5)]
    mjjj = np.stack([_comb_mass(J, c) for c in trips], axis=1)
    live_trip = np.stack([(jpt[:, a] > 10.0) & (jpt[:, b] > 10.0) & (jpt[:, c] > 10.0)
                          for a, b, c in trips], axis=1)
    f["dm_top"] = np.where(live_trip.any(1),
                           np.where(live_trip, np.abs(mjjj - M_TOP), 1e9).min(1), 300.0)
    f["dm_Wtop"] = f["dm_W"] + f["dm_top"]

    # HH pairing: split the 4 hardest jets into two pairs, keep the pairing whose
    # two masses agree best -- HH gives two equal ~125 GeV masses, tt does not.
    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    ms = np.stack([np.stack([_comb_mass(J, a), _comb_mass(J, b)], 1) for a, b in pairings], 1)
    sel = np.argmin(np.abs(ms[:, :, 0] - ms[:, :, 1]), axis=1)
    chosen = ms[rows, sel]
    m_hi = np.maximum(chosen[:, 0], chosen[:, 1])
    m_lo = np.minimum(chosen[:, 0], chosen[:, 1])
    f["m_bb1"], f["m_bb2"] = m_hi, m_lo
    f["dm_pair"] = m_hi - m_lo
    f["dm_higgs"] = np.sqrt((m_hi - M_H) ** 2 + (m_lo - M_H) ** 2)
    f["m_4jet"] = _comb_mass(J, (0, 1, 2, 3))

    d01 = np.sqrt((J["eta"][:, 0] - J["eta"][:, 1]) ** 2 + _dphi(J["phi"][:, 0], J["phi"][:, 1]) ** 2)
    f["dR_j12"] = np.where((jpt[:, 1] > 10.0), d01, 6.0)
    f["deta_j12"] = np.where((jpt[:, 1] > 10.0), np.abs(J["eta"][:, 0] - J["eta"][:, 1]), 6.0)

    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


FEATURE_NAMES = None   # filled on first compute()


def compute_chunked(X: np.ndarray, chunk: int = 100_000, verbose: bool = False):
    """compute() over a large cache without blowing up on the 16x16 pair tensor."""
    global FEATURE_NAMES
    outs = []
    for i in range(0, len(X), chunk):
        outs.append(compute(X[i:i + chunk]))
        if verbose:
            print(f"  features {min(i + chunk, len(X))}/{len(X)}", flush=True)
    names = list(outs[0])
    FEATURE_NAMES = names
    return names, np.stack([np.concatenate([o[n] for o in outs]) for n in names], axis=1)


# Shape-fixing transform for each feature, in the same "log1p" / "linear"
# vocabulary team/data.py uses for its 11 event features.  Anything that is a
# momentum, an energy or a mass gets log1p (they span three decades and have
# long tails); counts, ratios and angles stay linear.  A standardizing affine
# step follows, exactly as for the incumbent features.
TRANSFORM = {
    "mpt": "log1p", "mpt_over_ht": "linear", "mpt_sig": "log1p",
    "min_dphi_mpt": "linear",
    "iso_min": "linear", "iso_lead_pt": "log1p", "n_iso": "linear",
    "mt_lep_mpt": "log1p",
    "n_dxy_p05": "linear", "n_dxy_p20": "linear", "ptw_dxy": "linear",
    "max_pt_dxy": "log1p", "dxy_lead4": "linear",
    "m6": "log1p", "m8": "log1p", "m16": "log1p", "ht4_frac": "linear",
    "pt_ratio_41": "linear", "centrality": "linear", "eta_spread": "linear",
    "sphericity_T": "linear",
    "n_jets15": "linear", "n_jets30": "linear", "jet1_m": "log1p",
    "jet_m_max": "log1p", "jet_ptdxy_max": "log1p", "n_bjets": "linear",
    "ht_jets4": "log1p",
    "dm_W": "log1p", "m_jj_maxpt": "log1p", "dm_top": "log1p", "dm_Wtop": "log1p",
    "m_bb1": "log1p", "m_bb2": "log1p", "dm_pair": "log1p", "dm_higgs": "log1p",
    "m_4jet": "log1p", "dR_j12": "linear", "deta_j12": "linear",
}


# What each feature costs in firmware, which is half of whether it is worth
# taking.  Three classes:
#   "event"    O(P) reductions over the 16 candidates -- sums, maxima, counts,
#              one invariant mass.  Essentially free next to phi()'s 12,800 MACs.
#   "pairwise" needs the 16x16 Delta-R table.  cos(dphi) = c_i c_j + s_i s_j from
#              the cos/sin already in the input, so ~512 multiplies and 256
#              comparisons -- ~4% of phi(), and it is computed once per event
#              rather than replicated per particle.
#   "jets"     needs the iterative cone clustering: 6 sequential passes over the
#              pairwise table plus four-vector sums.  Sequential passes are what
#              hurt latency, so treat these as expensive until someone prototypes
#              a fixed 4-seed unrolled version.
COST = {
    "mpt": "event", "mpt_over_ht": "event", "mpt_sig": "event", "min_dphi_mpt": "event",
    "iso_min": "pairwise", "iso_lead_pt": "pairwise", "n_iso": "pairwise",
    "mt_lep_mpt": "pairwise",
    "n_dxy_p05": "event", "n_dxy_p20": "event", "ptw_dxy": "event",
    "max_pt_dxy": "event", "dxy_lead4": "event",
    "m6": "event", "m8": "event", "m16": "event", "ht4_frac": "event",
    "pt_ratio_41": "event", "centrality": "event", "eta_spread": "event",
    "sphericity_T": "event",
    "n_jets15": "jets", "n_jets30": "jets", "jet1_m": "jets", "jet_m_max": "jets",
    "jet_ptdxy_max": "jets", "n_bjets": "jets", "ht_jets4": "jets",
    "dm_W": "jets", "m_jj_maxpt": "jets", "dm_top": "jets", "dm_Wtop": "jets",
    "m_bb1": "jets", "m_bb2": "jets", "dm_pair": "jets", "dm_higgs": "jets",
    "m_4jet": "jets", "dR_j12": "jets", "deta_j12": "jets",
}
