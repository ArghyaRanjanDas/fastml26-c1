"""The exact preprocessing contract for an exported student.

Everything the FPGA lane needs to build the preprocessing block: every input
channel, the formula that produces it from the raw L1T_PUPPIPart fields, and the
frozen constant it is scaled by.  Written into each export json under
`input_spec` so the firmware never has to re-derive a formula from python.
"""
from data import (EVENT_FEATURES, EVENT_TRANSFORM, EVENT_STANDARDIZE, EVENT_CLIP,
                  PT_LOG_SCALE, ETA_SCALE, DXY_CLIP)

# Per-candidate channels, in tensor order. Candidates are the leading
# n_particles by descending pT; "lead" means candidate 0. All channels are
# multiplied by the slot mask (pt > 0), so an empty slot is all-zero.
PARTICLE_CHANNELS = [
    ("log_pt",        "log1p(pt) / 8"),
    ("eta",           "eta / 4"),
    ("dxy",           "clip(dxy, -2, 2) / 2"),
    ("cos_phi",       "cos(phi)"),
    ("sin_phi",       "sin(phi)"),
    ("lnz",           "log(max(pt,1e-6) / HT) / 4      # HT = sum of pt over the kept candidates"),
    ("lnE",           "log1p(pt * cosh(eta)) / 8"),
    ("cos_dphi_lead", "cos(dphi(phi, phi_lead))        # dphi wrapped to (-pi, pi]"),
    ("sin_dphi_lead", "sin(dphi(phi, phi_lead))"),
    ("deta_lead",     "(eta - eta_lead) / 2"),
    ("abs_dxy",       "abs(clip(dxy, -2, 2) / 2)"),
]

BASE_EVENT_FORMULA = {
    "ht": "sum of pt over the 16 candidates",
    "lead_pt1": "pt of candidate 0", "lead_pt2": "pt of candidate 1",
    "lead_pt3": "pt of candidate 2", "lead_pt4": "pt of candidate 3",
    "n_cand": "count of slots with pt > 0 (identically 16 in this dataset -- dead input)",
    "sum_abs_dxy": "sum of |dxy| over candidates",
    "max_abs_dxy": "max of |dxy| over candidates",
    "mean_abs_dxy": "sum|dxy| / n_cand",
    "m2": "invariant mass of the leading 2 candidates, massless approximation",
    "m4": "invariant mass of the leading 4 candidates, massless approximation",
}

EXTRA_EVENT_FORMULA = {
    "iso_lead_pt": ("pt of the most isolated hard candidate, where hard = pt > 10 GeV and "
                    "iso_i = (sum of pt of other candidates with dR < 0.4) / pt_i; "
                    "0 if no hard candidate"),
    "n_iso": "count of hard candidates (pt > 10) with iso < 0.15",
}
for _i, _a in enumerate([1, 1, 1, 2, 2, 3]):
    pass
for _p in ("p12", "p13", "p14", "p23", "p24", "p34"):
    EXTRA_EVENT_FORMULA[f"{_p}_lndR"] = (
        f"log(dR) between leading candidates {_p[1]} and {_p[2]} (1-indexed), "
        "dR = sqrt(deta^2 + dphi^2)")


def build(n_particles, event_names, extra_meta=None):
    """Assemble the input_spec dict for an export json."""
    ex_mean = ex_std = None
    ex_names = []
    if extra_meta:
        ex_names = list(extra_meta.get("extra_event_features", []))
        st = extra_meta.get("extra_standardize", {})
        ex_mean, ex_std = st.get("mean"), st.get("std")

    ev = []
    for i, n in enumerate(event_names):
        if n in EXTRA_EVENT_FORMULA:
            j = ex_names.index(n) if n in ex_names else None
            ev.append(dict(
                name=n, formula=EXTRA_EVENT_FORMULA[n],
                transform="log1p" if n == "iso_lead_pt" else "linear (already a log for *_lndR)",
                mean=(ex_mean[j] if ex_mean and j is not None else None),
                std=(ex_std[j] if ex_std and j is not None else None),
                clip=EVENT_CLIP))
        else:
            m, sd = EVENT_STANDARDIZE[n]
            ev.append(dict(name=n, formula=BASE_EVENT_FORMULA.get(n, ""),
                           transform=EVENT_TRANSFORM[n], mean=m, std=sd, clip=EVENT_CLIP))
    return dict(
        note=("Candidates are the leading n_particles L1T_PUPPIPart entries by descending pT "
              "(the dataset stores >=415 per event, already sorted, so this is pure truncation). "
              "Every per-candidate channel is masked by pt>0. Event features are computed from "
              "the full 16-candidate list even when the particle branch is fed fewer."),
        n_particles=n_particles,
        particle_channels=[dict(index=i, name=n, formula=f)
                           for i, (n, f) in enumerate(PARTICLE_CHANNELS)],
        particle_constants=dict(pt_log_scale=PT_LOG_SCALE, eta_scale=ETA_SCALE,
                                dxy_clip=DXY_CLIP),
        event_features=[dict(index=i, **e) for i, e in enumerate(ev)],
        event_feature_order="as listed; standardized value = clip((transform(raw)-mean)/std, +-clip)",
    )
