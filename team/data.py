"""Streaming data pipeline for FastML26 Challenge 1 (HH->4b vs QCD/tt/W+jets).

The full C1 dataset is 118 GB (train) + 14 GB (eval), so nothing here ever loads
a whole process.  We iterate parquet row batches, keep only the leaf columns we
need (`L1T_PUPPIPart.{pt,eta,phi,dxy}` + `label`), truncate/zero-pad each event
to a fixed number of particles, and stop as soon as the requested event cap is
reached.  The result is cached as .npy so repeated training runs are instant.

Selecting nested leaves instead of the whole `L1T_PUPPIPart` struct is a ~4x
speedup (the struct has 14 subfields, we use 4).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

DATA_ROOT = Path.home() / "hack-data" / "C1_HH4b"
CACHE_ROOT = Path.home() / "fastml26-hackathon" / "team" / "cache"

N_PARTICLES = 16          # candidates kept per event, matches the intro notebook
N_FEATURES = 5            # pt, eta, dxy, cos(phi), sin(phi)
CAND_FIELDS = ("pt", "eta", "phi", "dxy")
PARQUET_COLUMNS = [f"L1T_PUPPIPart.{f}" for f in CAND_FIELDS] + ["label"]

# label convention from the intro notebook
LABEL_QCD, LABEL_HH, LABEL_TT, LABEL_W = 0, 1, 2, 3


@dataclass(frozen=True)
class Process:
    directory: str
    label: int          # dataset label (0 QCD / 1 HH / 2 tt / 3 W+jets)
    group: str          # coarse group used for mixture weights and AUC breakdowns
    weight: float       # share of its group's event budget


SIGNAL = [Process("HH_4b", LABEL_HH, "HH_4b", 1.0)]

# Background budget is split evenly across the three background *groups*, and
# within a group evenly across its directories.  Trigger rates in the real
# experiment are QCD-dominated; an even mixture keeps the classifier from
# ignoring tt/W entirely and makes the pooled AUC easier to interpret.  All of
# this is overridable from the CLI.
BACKGROUND = [
    Process("QCD_HT250toInf", LABEL_QCD, "QCD", 1.0),
    Process("tt0123j_5f_ckm_LO_MLM_hadronic", LABEL_TT, "tt", 1 / 3),
    Process("tt0123j_5f_ckm_LO_MLM_leptonic", LABEL_TT, "tt", 1 / 3),
    Process("tt0123j_5f_ckm_LO_MLM_semiLeptonic", LABEL_TT, "tt", 1 / 3),
    Process("WJetsToLNu_13TeV-madgraphMLM-pythia8", LABEL_W, "Wjets", 0.5),
    Process("WJetsToQQ_13TeV-madgraphMLM-pythia8", LABEL_W, "Wjets", 0.5),
]
ALL_PROCESSES = SIGNAL + BACKGROUND
GROUP_ID = {"QCD": 0, "HH_4b": 1, "tt": 2, "Wjets": 3}


# ---------------------------------------------------------------- preprocessing

# Fixed (data-independent) scalings, deliberately *not* fitted min/max like the
# intro notebook: the FPGA implementation needs constants that do not change
# between train and inference, and they keep every feature inside ~[-1, 1],
# which is what the quantized version will want later.
PT_LOG_SCALE = 8.0        # log1p(pt) maxes out around 7.5 in this dataset
ETA_SCALE = 4.0           # |eta| <= 3.0 at trigger level
DXY_CLIP = 2.0            # |dxy| p99 is ~0.65 (signal) / ~0.06 (QCD); tails saturate


def preprocess(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, dxy: np.ndarray) -> np.ndarray:
    """(N, P) per-field arrays -> (N, P, 5) float32 tensor.

    Feature order: log-pt, eta, dxy, cos(phi), sin(phi).
    Zero-padded slots stay ~zero in every feature except cos(phi)=1; we mask
    them explicitly so padding is a true all-zero vector.
    """
    mask = pt > 0.0
    out = np.empty(pt.shape + (N_FEATURES,), dtype=np.float32)
    out[..., 0] = np.log1p(pt) / PT_LOG_SCALE
    out[..., 1] = eta / ETA_SCALE
    out[..., 2] = np.clip(dxy, -DXY_CLIP, DXY_CLIP) / DXY_CLIP
    out[..., 3] = np.cos(phi)
    out[..., 4] = np.sin(phi)
    out *= mask[..., None]
    return out


# ------------------------------------------------------- event-level features

# Names in the order they appear in the feature vector.  Anything reading the
# exported model must use this exact order.
EVENT_FEATURES = (
    "ht",              # scalar sum of the kept candidate pT
    "lead_pt1", "lead_pt2", "lead_pt3", "lead_pt4",
    "n_cand",          # number of non-empty candidate slots
    "sum_abs_dxy", "max_abs_dxy", "mean_abs_dxy",
    "m2",              # invariant mass of the leading 2 candidates (massless approx)
    "m4",              # invariant mass of the leading 4 candidates (massless approx)
)
N_EVENT_FEATURES = len(EVENT_FEATURES)

# Per feature: the shape-fixing transform applied to the physical value.
# "log1p" -> log1p(x), "linear" -> x.  A second, standardizing affine step
# follows (EVENT_STANDARDIZE); both are fixed constants for the same reason the
# particle scalings are -- the firmware needs them baked in and train/inference
# must agree.
EVENT_TRANSFORM = {
    "ht": "log1p",
    "lead_pt1": "log1p", "lead_pt2": "log1p", "lead_pt3": "log1p", "lead_pt4": "log1p",
    "n_cand": "linear",
    "sum_abs_dxy": "log1p",
    "max_abs_dxy": "log1p",
    "mean_abs_dxy": "linear",
    "m2": "log1p",
    "m4": "log1p",
}

# (mean, std) of the transformed value, measured once over the 600k-event
# train300k mixture and then frozen -- see `python data.py --fit-event-norm`.
# Squashing these into [0,1] instead (the first thing we tried) leaves features
# like ht at mean 0.69 / std 0.06, which measurably *hurt* the AUC; standardizing
# is what makes the event features actually usable by the network.
EVENT_STANDARDIZE = {
    "ht": (5.4634, 0.4903),
    "lead_pt1": (3.6467, 0.6996),
    "lead_pt2": (3.2596, 0.6106),
    "lead_pt3": (3.0494, 0.5573),
    "lead_pt4": (2.9041, 0.5187),
    "n_cand": (16.0000, 0.0000),
    "sum_abs_dxy": (0.3660, 0.4410),
    "max_abs_dxy": (0.2510, 0.3318),
    "mean_abs_dxy": (0.0385, 0.0620),
    "m2": (3.4064, 1.5627),
    "m4": (4.8331, 0.8401),
}
EVENT_CLIP = 5.0   # standardized features are clipped to +/- this


def _inv_mass(pt, eta, phi, mask, k):
    """Invariant mass of the leading `k` candidates, massless approximation.

    Candidates arrive already sorted by descending pT, so a plain leading-k slice
    is the right selection.  Empty slots are masked to zero so they add nothing
    to the four-vector sum.
    """
    s = slice(0, k)
    m = mask[:, s]
    p, e, f = pt[:, s] * m, eta[:, s], phi[:, s]
    E = (p * np.cosh(e)).sum(1)
    px = (p * np.cos(f)).sum(1)
    py = (p * np.sin(f)).sum(1)
    pz = (p * np.sinh(e)).sum(1)
    return np.sqrt(np.maximum(E * E - px * px - py * py - pz * pz, 0.0))


def event_features(pt, eta, phi, dxy):
    """(N, P) raw per-field arrays -> (N, 11) normalized event-level features.

    Computed from *physical* units before the per-particle scaling, then squashed
    then standardized with the fixed EVENT_STANDARDIZE constants.
    """
    mask = (pt > 0.0).astype(np.float32)
    n_cand = mask.sum(1)
    abs_dxy = np.abs(dxy) * mask

    lead = np.zeros((len(pt), 4), dtype=np.float32)
    k = min(4, pt.shape[1])
    lead[:, :k] = pt[:, :k] * mask[:, :k]

    raw = {
        "ht": (pt * mask).sum(1),
        "lead_pt1": lead[:, 0], "lead_pt2": lead[:, 1],
        "lead_pt3": lead[:, 2], "lead_pt4": lead[:, 3],
        "n_cand": n_cand,
        "sum_abs_dxy": abs_dxy.sum(1),
        "max_abs_dxy": abs_dxy.max(1),
        "mean_abs_dxy": abs_dxy.sum(1) / np.maximum(n_cand, 1.0),
        "m2": _inv_mass(pt, eta, phi, mask, 2),
        "m4": _inv_mass(pt, eta, phi, mask, 4),
    }

    return standardize_event_features(raw_event_features(raw))


def raw_event_features(raw: dict) -> np.ndarray:
    """Apply only the per-feature log1p/linear transform, no standardization."""
    out = np.empty((len(raw["ht"]), N_EVENT_FEATURES), dtype=np.float32)
    for i, name in enumerate(EVENT_FEATURES):
        v = raw[name].astype(np.float32)
        out[:, i] = np.log1p(np.maximum(v, 0.0)) if EVENT_TRANSFORM[name] == "log1p" else v
    return out


def standardize_event_features(t: np.ndarray) -> np.ndarray:
    """(N, 11) transformed features -> zero-mean/unit-variance, clipped.

    A feature with zero recorded spread (n_cand is identically 16 whenever all
    16 candidate slots are filled, which is every event in this dataset) is
    emitted as 0 rather than dividing by zero: it is a dead input, kept because
    it stops being dead the moment we lower N_PARTICLES or apply a pT threshold.
    """
    out = np.empty_like(t)
    for i, name in enumerate(EVENT_FEATURES):
        mean, std = EVENT_STANDARDIZE[name]
        out[:, i] = 0.0 if std < 1e-6 else np.clip((t[:, i] - mean) / std, -EVENT_CLIP, EVENT_CLIP)
    return out


# ------------------------------------------------- extra tt-focused features

# c2's lane.  The 11 EVENT_FEATURES above were chosen before we knew tt was the
# weak background; these three come out of the ranked study in team/physics
# (see team/physics/README.md and the tt section of RESULTS.md).  They are
# computed by team/physics/features.py from the same 16 candidates and appended
# to the event-feature vector, so they inherit the property that makes the
# event features cheap in firmware: one evaluation per event, after the pool,
# never replicated 16x inside phi().
#
# Off by default -- a cache built without --extra-features is byte-identical to
# what it was before this block existed.
EXTRA_FEATURES = (
    "iso_lead_pt",   # pT of the most isolated candidate with pT>10 (sum pT in dR<0.4)
    "n_iso",         # how many pT>10 candidates have iso < 0.15
    # ln dR' for the 6 pairs among the leading 4 candidates, where dR'^2 =
    # deta^2 + 2(1 - cos dphi).  Identical discriminating power to the textbook
    # dR (measured: +0.0158 vs +0.0156 AUC vs tt) and it needs no atan2 and no
    # 2pi wrap in firmware -- cos dphi is c_i c_j + s_i s_j from inputs already
    # present.  See team/fpga/FEATURES.md.
    "p12_lndRc", "p13_lndRc", "p14_lndRc", "p23_lndRc", "p24_lndRc", "p34_lndRc",
)

# (mean, std) of the transformed extra features, measured once over a 300k-event
# train mixture and then frozen, exactly like EVENT_STANDARDIZE.
# Re-measure with: python data.py --fit-extra-norm
EXTRA_STANDARDIZE = {
    "iso_lead_pt": (3.0265, 0.7147),
    "n_iso": (1.9184, 1.4955),
    "p12_lndRc": (-0.2405, 1.7891),
    "p13_lndRc": (-0.1209, 1.6921),
    "p14_lndRc": (-0.0330, 1.6142),
    "p23_lndRc": (-0.1362, 1.6832),
    "p24_lndRc": (-0.0614, 1.6157),
    "p34_lndRc": (-0.0356, 1.5873),
}


def extra_event_features(pt, eta, phi, dxy, names=None):
    """(N, P) raw per-field arrays -> (N, len(EXTRA_FEATURES)) normalized features."""
    from physics.features import compute_raw, TRANSFORM

    names = EXTRA_FEATURES if names is None else names
    if not names:
        return np.zeros((len(pt), 0), dtype=np.float32)
    raw = compute_raw(pt, eta, phi, dxy, pt > 0.0)
    out = np.empty((len(pt), len(names)), dtype=np.float32)
    for i, n in enumerate(names):
        v = raw[n].astype(np.float32)
        v = np.log1p(np.maximum(v, 0.0)) if TRANSFORM[n] == "log1p" else v
        mean, std = EXTRA_STANDARDIZE.get(n, (0.0, 1.0))
        out[:, i] = 0.0 if std < 1e-6 else np.clip((v - mean) / std, -EVENT_CLIP, EVENT_CLIP)
    return out


def effective_event_features(extra: bool):
    """The feature names actually present in F, in order."""
    return list(EVENT_FEATURES) + (list(EXTRA_FEATURES) if extra else [])


# ------------------------------------------ rich per-candidate channels (c2)

# The teacher (team/teacher/common.py) feeds phi() 6 derived numbers per
# candidate on top of the 5 cached ones, and they are worth more to the student
# than anything else measured in round 3: +0.037 AUC vs tt, +0.0136 overall, for
# +192 parameters (team/RESULTS.md, c2 section).  All six are O(1) per candidate.
PARTICLE_CHANNELS = ("log_pt", "eta", "dxy", "cos_phi", "sin_phi")
RICH_CHANNELS = PARTICLE_CHANNELS + (
    "lnz",             # ln(pt / HT) / 4
    "lnE",             # log1p(pt cosh eta) / 8
    "cos_dphi_lead",   # cos(phi - phi_1)
    "sin_dphi_lead",   # sin(phi - phi_1)
    "deta_lead",       # (eta - eta_1) / 2
    "abs_dxy",         # |dxy| / 2
)
N_RICH_FEATURES = len(RICH_CHANNELS)


def rich_preprocess(pt, eta, phi, dxy):
    """(N, P) raw per-field arrays -> (N, P, 11) float32, the canonical student input.

    Channels 0-4 are exactly `preprocess()`; 5-10 are the teacher's derived ones,
    in the teacher's order, so a model trained on either side sees the same tensor.
    """
    from physics.derived import rich_from_raw

    return rich_from_raw(pt, eta, phi, dxy).astype(np.float32)


def effective_particle_channels(rich: bool):
    return list(RICH_CHANNELS if rich else PARTICLE_CHANNELS)


# ------------------------------------------------------------------- streaming

def _pad(cands: ak.Array, field: str, n_particles: int) -> np.ndarray:
    padded = ak.pad_none(cands[field], n_particles, clip=True, axis=1)
    return ak.to_numpy(ak.fill_none(padded, 0.0)).astype(np.float32)


def stream_process(
    process_dir: Path,
    max_events: int,
    n_particles: int = N_PARTICLES,
    batch_size: int = 20_000,
    skip_files: int = 0,
    extra: bool = False,
    rich: bool = False,
):
    """Yield (X, F, y) chunks from a process directory until `max_events` is reached.

    X is (n, n_particles, 5) float32 per-particle input, F is (n, 11) float32
    event-level features, y is (n,) int8 holding the dataset label.
    """
    files = sorted(process_dir.glob(f"{process_dir.name}_*.parquet"))[skip_files:]
    if not files:
        raise FileNotFoundError(f"no parquet fragments under {process_dir}")

    seen = 0
    for path in files:
        if seen >= max_events:
            return
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=PARQUET_COLUMNS):
            arr = ak.from_arrow(batch)
            cands = arr["L1T_PUPPIPart"]
            fields = {f: _pad(cands, f, n_particles) for f in CAND_FIELDS}
            build = rich_preprocess if rich else preprocess
            X = build(fields["pt"], fields["eta"], fields["phi"], fields["dxy"])
            F = event_features(fields["pt"], fields["eta"], fields["phi"], fields["dxy"])
            if extra:
                F = np.concatenate([F, extra_event_features(
                    fields["pt"], fields["eta"], fields["phi"], fields["dxy"])], axis=1)
            y = ak.to_numpy(arr["label"]).astype(np.int8)

            take = min(len(X), max_events - seen)
            seen += take
            yield X[:take], F[:take], y[:take]
            if seen >= max_events:
                return


def load_split(
    split: str,
    n_signal: int,
    n_background: int,
    n_particles: int = N_PARTICLES,
    skip_files: int = 0,
    verbose: bool = True,
    extra: bool = False,
    rich: bool = False,
):
    """Load a capped, mixed signal+background sample from train/ or eval/.

    Returns (X, F, y_binary, group): per-particle inputs, event-level features,
    the binary target (1 for HH_4b), and the coarse process group id used for
    the per-background AUC breakdown.
    """
    root = DATA_ROOT / split
    Xs, Fs, ys, gs = [], [], [], []

    budgets = [(p, n_signal) for p in SIGNAL]
    budgets += [(p, int(round(n_background * p.weight / 3.0))) for p in BACKGROUND]

    for proc, budget in budgets:
        if budget <= 0:
            continue
        chunks = list(stream_process(root / proc.directory, budget, n_particles,
                                     skip_files=skip_files, extra=extra, rich=rich))
        X = np.concatenate([c[0] for c in chunks])
        F = np.concatenate([c[1] for c in chunks])
        y = np.concatenate([c[2] for c in chunks])
        Xs.append(X)
        Fs.append(F)
        ys.append(y)
        gs.append(np.full(len(X), GROUP_ID[proc.group], dtype=np.int8))
        if verbose:
            print(f"  {split}/{proc.directory:<44s} {len(X):>8d} events (label {int(y[0])})", flush=True)

    X = np.concatenate(Xs)
    F = np.concatenate(Fs)
    y_raw = np.concatenate(ys)
    group = np.concatenate(gs)
    y = (y_raw == LABEL_HH).astype(np.float32)
    return X, F, y, group


# ----------------------------------------------------------------------- cache

CACHE_VERSION = 2   # v2 added the event-level feature array


def cache_paths(tag: str):
    d = CACHE_ROOT / tag
    return d / "X.npy", d / "F.npy", d / "y.npy", d / "group.npy", d / "meta.json"


def build_cache(tag: str, split: str, n_signal: int, n_background: int,
                n_particles: int = N_PARTICLES, skip_files: int = 0, force: bool = False,
                extra: bool = False, rich: bool = False):
    Xp, Fp, yp, gp, mp = cache_paths(tag)
    if mp.exists() and not force:
        stale = json.loads(mp.read_text()).get("version", 1) < CACHE_VERSION
        if not stale:
            print(f"cache '{tag}' already exists at {Xp.parent} (use --force to rebuild)")
            return
        print(f"cache '{tag}' predates the event features -- rebuilding", flush=True)
    Xp.parent.mkdir(parents=True, exist_ok=True)
    print(f"building cache '{tag}' from {split}/ ...", flush=True)
    X, F, y, group = load_split(split, n_signal, n_background, n_particles, skip_files,
                                extra=extra, rich=rich)
    np.save(Xp, X)
    np.save(Fp, F)
    np.save(yp, y)
    np.save(gp, group)
    meta = dict(tag=tag, version=CACHE_VERSION, split=split, n_signal=n_signal,
                n_background=n_background, n_particles=n_particles,
                n_features=X.shape[2], particle_channels=effective_particle_channels(rich),
                event_features=effective_event_features(extra), skip_files=skip_files,
                n_events=int(len(X)), n_pos=int(y.sum()), n_neg=int((1 - y).sum()))
    mp.write_text(json.dumps(meta, indent=2))
    print(f"  -> {len(X)} events, {int(y.sum())} signal / {int((1-y).sum())} background")
    print(f"  -> {Xp}  ({X.nbytes / 1e6:.0f} MB)")


def load_cache(tag: str):
    Xp, Fp, yp, gp, mp = cache_paths(tag)
    if not mp.exists():
        raise FileNotFoundError(f"no cache '{tag}'; run: python team/data.py --tag {tag} ...")
    return np.load(Xp), np.load(Fp), np.load(yp), np.load(gp), json.loads(mp.read_text())


def fit_event_norm(split: str = "train", n_signal: int = 100_000, n_background: int = 100_000):
    """Measure (mean, std) of the transformed event features and print them.

    Run this once when the feature list or transforms change, then paste the
    result into EVENT_STANDARDIZE.  Deliberately not fitted at load time: the
    constants have to be frozen so training, evaluation and firmware agree.
    """
    root = DATA_ROOT / split
    budgets = [(p_, n_signal) for p_ in SIGNAL]
    budgets += [(p_, int(round(n_background * p_.weight / 3.0))) for p_ in BACKGROUND]
    chunks = []
    for proc, budget in budgets:
        for X, F, y in stream_process(root / proc.directory, budget):
            chunks.append(F)
    F = np.concatenate(chunks)
    # F comes back standardized with the *current* constants; undo that to
    # recover the transformed values, so re-fitting converges in one pass.
    cur = np.array([EVENT_STANDARDIZE[n] for n in EVENT_FEATURES], dtype=np.float64)
    T = np.where(cur[:, 1] < 1e-6, cur[:, 0], F * cur[:, 1] + cur[:, 0])
    print("EVENT_STANDARDIZE = {")
    for i, name in enumerate(EVENT_FEATURES):
        m, sd = T[:, i].mean(), T[:, i].std()
        print(f'    "{name}": ({m:.4f}, {sd:.4f}),')
    print("}")


def fit_extra_norm(split: str = "train", n_signal: int = 100_000, n_background: int = 100_000):
    """Measure (mean, std) of the *transformed* extra features and print them.

    Same contract as fit_event_norm: run once when EXTRA_FEATURES changes, paste
    the result into EXTRA_STANDARDIZE, and freeze it, so training, evaluation
    and firmware all see the same numbers.
    """
    from physics.features import compute_raw, TRANSFORM

    root = DATA_ROOT / split
    budgets = [(p_, n_signal) for p_ in SIGNAL]
    budgets += [(p_, int(round(n_background * p_.weight / 3.0))) for p_ in BACKGROUND]
    acc = {n: [] for n in EXTRA_FEATURES}
    for proc, budget in budgets:
        for X, F, y in stream_process(root / proc.directory, budget):
            pt = np.expm1(X[..., 0] * PT_LOG_SCALE)
            eta, dxy = X[..., 1] * ETA_SCALE, X[..., 2] * DXY_CLIP
            phi = np.arctan2(X[..., 4], X[..., 3])
            raw = compute_raw(pt, eta, phi, dxy, X[..., 0] > 0.0)
            for n in EXTRA_FEATURES:
                v = raw[n].astype(np.float64)
                acc[n].append(np.log1p(np.maximum(v, 0.0)) if TRANSFORM[n] == "log1p" else v)
    print("EXTRA_STANDARDIZE = {")
    for n in EXTRA_FEATURES:
        v = np.concatenate(acc[n])
        print(f'    "{n}": ({v.mean():.4f}, {v.std():.4f}),')
    print("}")


def main():
    ap = argparse.ArgumentParser(description="build a cached, capped C1 sample")
    ap.add_argument("--fit-event-norm", action="store_true",
                    help="print freshly measured EVENT_STANDARDIZE constants and exit")
    ap.add_argument("--tag", help="cache name, e.g. train300k")
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--n-signal", type=int, default=300_000)
    ap.add_argument("--n-background", type=int, default=300_000)
    ap.add_argument("--n-particles", type=int, default=N_PARTICLES)
    ap.add_argument("--skip-files", type=int, default=0,
                    help="skip the first N parquet fragments per process (disjoint samples)")
    ap.add_argument("--rich-particles", action="store_true",
                    help="feed phi the 11 canonical channels instead of the cached 5")
    ap.add_argument("--extra-features", action="store_true",
                    help="append EXTRA_FEATURES (the tt-focused ones) to the event vector")
    ap.add_argument("--fit-extra-norm", action="store_true",
                    help="print freshly measured EXTRA_STANDARDIZE constants and exit")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.fit_event_norm:
        fit_event_norm(a.split, a.n_signal, a.n_background)
        return
    if a.fit_extra_norm:
        fit_extra_norm(a.split, a.n_signal, a.n_background)
        return
    if not a.tag:
        ap.error("--tag is required unless --fit-event-norm is given")
    build_cache(a.tag, a.split, a.n_signal, a.n_background, a.n_particles, a.skip_files,
                a.force, a.extra_features, a.rich_particles)


if __name__ == "__main__":
    main()
