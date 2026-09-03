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

import itertools

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
    deta_ij = eta[:, :, None] - eta[:, None, :]
    dphi_ij = _dphi(phi[:, :, None], phi[:, None, :])
    dr2 = deta_ij ** 2 + dphi_ij ** 2
    # the multiply-only metric: 2(1 - cos dphi) needs no atan2 and no wrap, and
    # agrees with dphi^2 to fourth order (0.16 vs 0.1578 at the cone edge)
    dr2c = deta_ij ** 2 + 2.0 * (1.0 - np.cos(dphi_ij))
    pair = mask[:, :, None] & mask[:, None, :]
    np.einsum("ijj->ij", dr2)[:] = 1e9          # drop self-pairs
    np.einsum("ijj->ij", dr2c)[:] = 1e9
    cone = pair & (dr2 < 0.16)
    cone_c = pair & (dr2c < 0.16)
    iso_sum = (pt[:, None, :] * cone).sum(2)    # pT around each candidate, R<0.4
    with np.errstate(divide="ignore", invalid="ignore"):
        iso = np.where(mask, iso_sum / np.maximum(pt, 1e-6), 1e9)
    hard = mask & (pt > 10.0)
    iso_h = np.where(hard, iso, 1e9)
    rows = np.arange(N)
    best = np.argmin(iso_h, axis=1)
    has_hard = hard.any(1)
    f["iso_min"] = np.where(has_hard, np.minimum(iso_h[rows, best], 10.0), 10.0)
    f["iso_lead_pt"] = np.where(has_hard, pt[rows, best], 0.0)
    f["n_iso"] = (hard & (iso < 0.15)).sum(1).astype(np.float64)
    # Cheaper variants: only the leading 4 candidates may *be* the isolated one,
    # though the cone still sums over all 16.  That is 4x16 = 64 Delta-R
    # evaluations instead of 16x16 = 256, and the lepton in a leptonic tt is
    # almost always among the hardest few.  Whether the 4x saving costs anything
    # is measured, not assumed.
    hard4 = np.zeros_like(hard)
    hard4[:, :4] = hard[:, :4]
    iso_4 = np.where(hard4, iso, 1e9)
    best4 = np.argmin(iso_4, axis=1)
    has4 = hard4.any(1)
    f["iso_lead_pt_s4"] = np.where(has4, pt[rows, best4], 0.0)
    f["n_iso_s4"] = (hard4 & (iso < 0.15)).sum(1).astype(np.float64)
    f["iso_min_s4"] = np.where(has4, np.minimum(iso_4[rows, best4], 10.0), 10.0)
    # the same two features off the multiply-only cone, for the firmware version
    iso_sum_c = (pt[:, None, :] * cone_c).sum(2)
    with np.errstate(divide="ignore", invalid="ignore"):
        iso_c = np.where(mask, iso_sum_c / np.maximum(pt, 1e-6), 1e9)
    iso_hc = np.where(hard, iso_c, 1e9)
    bc = np.argmin(iso_hc, axis=1)
    f["iso_lead_pt_c"] = np.where(has_hard, pt[rows, bc], 0.0)
    f["n_iso_c"] = (hard & (iso_c < 0.15)).sum(1).astype(np.float64)

    hard8 = np.zeros_like(hard)
    hard8[:, :8] = hard[:, :8]
    iso_8 = np.where(hard8, iso, 1e9)
    best8 = np.argmin(iso_8, axis=1)
    has8 = hard8.any(1)
    f["iso_lead_pt_s8"] = np.where(has8, pt[rows, best8], 0.0)
    f["n_iso_s8"] = (hard8 & (iso < 0.15)).sum(1).astype(np.float64)
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
    # Order statistics of |dxy|: HH->4b has four b hadrons, tt has two, and a sum
    # cannot tell "few, very displaced" from "many, mildly displaced" -- the 2nd
    # to 4th largest can.  The baseline already carries the 1st (max_abs_dxy) and
    # the sum.  In firmware this is a comparator network: no DSP at all.
    dxy_sorted = np.sort(adxy, axis=1)[:, ::-1]
    for k in range(4):
        f[f"dxy_ord{k+1}"] = dxy_sorted[:, k]
    # and the one that says "were there four of them": the 4th largest over the
    # largest, i.e. how evenly the displacement is shared
    f["dxy_ord4_frac"] = dxy_sorted[:, 3] / np.maximum(dxy_sorted[:, 0], 1e-6)

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

    # --- top tag in disguise, from the leading 6 candidates directly -------
    # Hadronic tt is the mode the baseline loses on (0.723), and it is the one
    # with a W -> qq and a t -> Wb inside it.  Working on candidates rather than
    # on clustered jets keeps this cheap: the 15 pair masses are computed once,
    # and every 3-candidate mass is then just a sum, because for massless
    # constituents m_ijk^2 = m_ij^2 + m_ik^2 + m_jk^2.
    K = 6
    m2_pair, pair_idx = {}, list(itertools.combinations(range(K), 2))
    for i, j in pair_idx:
        de = eta[:, i] - eta[:, j]
        dp = _dphi(phi[:, i], phi[:, j])
        m2_pair[(i, j)] = np.maximum(
            2.0 * pt[:, i] * pt[:, j] * (np.cosh(de) - np.cos(dp)), 0.0)
    mjj = np.sqrt(np.stack([m2_pair[k] for k in pair_idx], axis=1))
    f["dm_W6"] = np.abs(mjj - M_W).min(1)
    f["m_W6"] = mjj[rows, np.abs(mjj - M_W).argmin(1)]
    f["mjj_max6"] = mjj.max(1)

    trip_idx = list(itertools.combinations(range(K), 3))
    mjjj = np.sqrt(np.stack([m2_pair[(a, b)] + m2_pair[(a, c)] + m2_pair[(b, c)]
                             for a, b, c in trip_idx], axis=1))
    f["dm_top6"] = np.abs(mjjj - M_TOP).min(1)
    f["m_top6"] = mjjj[rows, np.abs(mjjj - M_TOP).argmin(1)]
    # the tt signature is a W *inside* a top: score both at once
    f["dm_Wtop6"] = f["dm_W6"] + f["dm_top6"]

    # ln dR for the 6 pairs among the leading 4 candidates.  Of ParT's four pair
    # quantities this is the only one that pays for its columns (stage 4), and 6
    # numbers after the pool is a size a trigger student can carry.
    for i in range(4):
        for j in range(i + 1, 4):
            de = eta[:, i] - eta[:, j]
            dp = _dphi(phi[:, i], phi[:, j])
            f[f"p{i+1}{j+1}_lndR"] = 0.5 * np.log(np.maximum(de * de + dp * dp, 1e-16))
            # Fixed-point-friendly twin: 2(1 - cos dphi) instead of dphi^2.  cos dphi
            # is c_i c_j + s_i s_j from the cos/sin already in the input, so this
            # version needs no atan2 and no wrap -- two multiplies and an add.  It
            # equals dphi^2 to fourth order and stays monotone in |dphi| all the way
            # to pi, so it is a reparametrization, not an approximation to check.
            # Whether the network cares is measured, not assumed (see FEATURES.md).
            f[f"p{i+1}{j+1}_lndRc"] = 0.5 * np.log(
                np.maximum(de * de + 2.0 * (1.0 - np.cos(dp)), 1e-16))

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
    "iso_lead_pt_s4": "log1p", "n_iso_s4": "linear", "iso_min_s4": "linear",
    "iso_lead_pt_s8": "log1p", "n_iso_s8": "linear",
    "iso_lead_pt_c": "log1p", "n_iso_c": "linear",
    "mt_lep_mpt": "log1p",
    "n_dxy_p05": "linear", "n_dxy_p20": "linear", "ptw_dxy": "linear",
    "max_pt_dxy": "log1p", "dxy_lead4": "linear",
    "dxy_ord1": "linear", "dxy_ord2": "linear", "dxy_ord3": "linear",
    "dxy_ord4": "linear", "dxy_ord4_frac": "linear",
    "m6": "log1p", "m8": "log1p", "m16": "log1p", "ht4_frac": "linear",
    "pt_ratio_41": "linear", "centrality": "linear", "eta_spread": "linear",
    "sphericity_T": "linear",
    "n_jets15": "linear", "n_jets30": "linear", "jet1_m": "log1p",
    "jet_m_max": "log1p", "jet_ptdxy_max": "log1p", "n_bjets": "linear",
    "ht_jets4": "log1p",
    "dm_W": "log1p", "m_jj_maxpt": "log1p", "dm_top": "log1p", "dm_Wtop": "log1p",
    "m_bb1": "log1p", "m_bb2": "log1p", "dm_pair": "log1p", "dm_higgs": "log1p",
    "m_4jet": "log1p", "dR_j12": "linear", "deta_j12": "linear",
    "dm_W6": "log1p", "m_W6": "log1p", "mjj_max6": "log1p",
    "dm_top6": "log1p", "m_top6": "log1p", "dm_Wtop6": "log1p",
    **{f"p{i+1}{j+1}_lndR": "linear" for i in range(4) for j in range(i + 1, 4)},
    **{f"p{i+1}{j+1}_lndRc": "linear" for i in range(4) for j in range(i + 1, 4)},
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
    "iso_lead_pt_s4": "pairwise/4", "n_iso_s4": "pairwise/4", "iso_min_s4": "pairwise/4",
    "iso_lead_pt_s8": "pairwise/2", "n_iso_s8": "pairwise/2",
    "iso_lead_pt_c": "pairwise", "n_iso_c": "pairwise",
    "mt_lep_mpt": "pairwise",
    "n_dxy_p05": "event", "n_dxy_p20": "event", "ptw_dxy": "event",
    "max_pt_dxy": "event", "dxy_lead4": "event",
    # a sorting network over 16 values: comparators only, zero DSP
    "dxy_ord1": "event", "dxy_ord2": "event", "dxy_ord3": "event",
    "dxy_ord4": "event", "dxy_ord4_frac": "event",
    "m6": "event", "m8": "event", "m16": "event", "ht4_frac": "event",
    "pt_ratio_41": "event", "centrality": "event", "eta_spread": "event",
    "sphericity_T": "event",
    "n_jets15": "jets", "n_jets30": "jets", "jet1_m": "jets", "jet_m_max": "jets",
    "jet_ptdxy_max": "jets", "n_bjets": "jets", "ht_jets4": "jets",
    "dm_W": "jets", "m_jj_maxpt": "jets", "dm_top": "jets", "dm_Wtop": "jets",
    "m_bb1": "jets", "m_bb2": "jets", "dm_pair": "jets", "dm_higgs": "jets",
    "m_4jet": "jets", "dR_j12": "jets", "deta_j12": "jets",
    # only 6 of the 120 pairs, and the leading 4 candidates are known positions:
    # 6 x (2 subtractions + 2 squares + a log) with no search and no clustering
    # 15 pair masses over the leading 6; the 20 triple masses are then adds only
    "dm_W6": "pair-lead6", "m_W6": "pair-lead6", "mjj_max6": "pair-lead6",
    "dm_top6": "pair-lead6", "m_top6": "pair-lead6", "dm_Wtop6": "pair-lead6",
    **{f"p{i+1}{j+1}_lndR": "pair-lead4" for i in range(4) for j in range(i + 1, 4)},
    **{f"p{i+1}{j+1}_lndRc": "pair-lead4" for i in range(4) for j in range(i + 1, 4)},
}


# ------------------------------- features from the two unused candidate fields

# team/physics/COLUMNS.md: we were using 4 of L1T_PUPPIPart's 14 subfields.  Two
# of the other ten answer questions this module spent the day approximating --
# `pdgId` flags electrons and muons directly (iso_lead_pt is a hand-built proxy
# for a lepton), and `dxysig` is the impact-parameter *significance*, which is
# what a real b-tagger uses where we were feeding raw dxy.  Together they are
# worth +0.076 AUC vs tt on top of the 11 incumbent event features.
#
# dxysig is stored as float16 and overflows: |dxysig| reaches inf and its p99 is
# in the thousands, so it is clipped before anything else touches it.
DXYSIG_CLIP = 20.0

EXTRA_FIELD_FEATURES = (
    "dsig_ptw", "dsig_sum", "dsig_ord3", "dsig_ord4", "n_dsig_gt3",
    "n_lep", "lead_lep_pt", "lep_pt_frac", "charged_frac",
)


def clip_dxysig(dxysig, mask):
    """|dxysig|, de-infed and clipped -- the only safe way to use this field."""
    ds = np.abs(np.nan_to_num(np.asarray(dxysig, dtype=np.float64),
                              nan=0.0, posinf=DXYSIG_CLIP, neginf=DXYSIG_CLIP))
    return np.clip(ds, 0.0, DXYSIG_CLIP) * mask


def compute_extra_fields(pt, dxysig, pdgid, mask) -> dict:
    """Event-level features from `dxysig` and `pdgId`. Same contract as compute_raw."""
    m = mask.astype(np.float64)
    ds = clip_dxysig(dxysig, m)
    srt = np.sort(ds, axis=1)[:, ::-1]
    ht = np.maximum((pt * m).sum(1), 1e-6)
    pid = np.abs(np.asarray(pdgid)).astype(np.int32)
    is_e, is_mu = (pid == 11) & mask, (pid == 13) & mask
    lep = is_e | is_mu
    lead_lep_pt = (pt * lep).max(1)
    f = {
        "dsig_ptw": (pt * ds).sum(1) / ht,      # pT-weighted displacement significance
        "dsig_sum": ds.sum(1),
        "dsig_ord3": srt[:, 2],
        "dsig_ord4": srt[:, 3],
        "n_dsig_gt3": (ds > 3.0).sum(1).astype(np.float64),
        "n_lep": lep.sum(1).astype(np.float64),
        "lead_lep_pt": lead_lep_pt,
        "lep_pt_frac": lead_lep_pt / ht,
        "charged_frac": (pt * (((pid == 211) | lep) & mask)).sum(1) / ht,
    }
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


TRANSFORM.update({
    "dsig_ptw": "log1p", "dsig_sum": "log1p", "dsig_ord3": "linear",
    "dsig_ord4": "linear", "n_dsig_gt3": "linear",
    "n_lep": "linear", "lead_lep_pt": "log1p", "lep_pt_frac": "linear",
    "charged_frac": "linear",
})
# All nine are O(16) reductions over fields the trigger already has per candidate:
# adds, compares and one sorting network.  No DSP, no new pair table.
COST.update({k: "event" for k in EXTRA_FIELD_FEATURES})
