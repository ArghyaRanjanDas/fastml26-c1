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
):
    """Yield (X, y) chunks from a process directory until `max_events` is reached.

    X is (n, n_particles, 5) float32, y is (n,) int8 holding the dataset label.
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
            X = preprocess(fields["pt"], fields["eta"], fields["phi"], fields["dxy"])
            y = ak.to_numpy(arr["label"]).astype(np.int8)

            take = min(len(X), max_events - seen)
            seen += take
            yield X[:take], y[:take]
            if seen >= max_events:
                return


def load_split(
    split: str,
    n_signal: int,
    n_background: int,
    n_particles: int = N_PARTICLES,
    skip_files: int = 0,
    verbose: bool = True,
):
    """Load a capped, mixed signal+background sample from train/ or eval/.

    Returns (X, y_binary, group) where y_binary is 1 for HH_4b and 0 otherwise,
    and group is the coarse process group id (for per-background AUC).
    """
    root = DATA_ROOT / split
    Xs, ys, gs = [], [], []

    budgets = [(p, n_signal) for p in SIGNAL]
    budgets += [(p, int(round(n_background * p.weight / 3.0))) for p in BACKGROUND]

    for proc, budget in budgets:
        if budget <= 0:
            continue
        chunks = list(stream_process(root / proc.directory, budget, n_particles,
                                     skip_files=skip_files))
        X = np.concatenate([c[0] for c in chunks])
        y = np.concatenate([c[1] for c in chunks])
        Xs.append(X)
        ys.append(y)
        gs.append(np.full(len(X), GROUP_ID[proc.group], dtype=np.int8))
        if verbose:
            print(f"  {split}/{proc.directory:<44s} {len(X):>8d} events (label {int(y[0])})", flush=True)

    X = np.concatenate(Xs)
    y_raw = np.concatenate(ys)
    group = np.concatenate(gs)
    y = (y_raw == LABEL_HH).astype(np.float32)
    return X, y, group


# ----------------------------------------------------------------------- cache

def cache_paths(tag: str):
    d = CACHE_ROOT / tag
    return d / "X.npy", d / "y.npy", d / "group.npy", d / "meta.json"


def build_cache(tag: str, split: str, n_signal: int, n_background: int,
                n_particles: int = N_PARTICLES, skip_files: int = 0, force: bool = False):
    Xp, yp, gp, mp = cache_paths(tag)
    if mp.exists() and not force:
        print(f"cache '{tag}' already exists at {Xp.parent} (use --force to rebuild)")
        return
    Xp.parent.mkdir(parents=True, exist_ok=True)
    print(f"building cache '{tag}' from {split}/ ...", flush=True)
    X, y, group = load_split(split, n_signal, n_background, n_particles, skip_files)
    np.save(Xp, X)
    np.save(yp, y)
    np.save(gp, group)
    meta = dict(tag=tag, split=split, n_signal=n_signal, n_background=n_background,
                n_particles=n_particles, n_features=N_FEATURES, skip_files=skip_files,
                n_events=int(len(X)), n_pos=int(y.sum()), n_neg=int((1 - y).sum()))
    mp.write_text(json.dumps(meta, indent=2))
    print(f"  -> {len(X)} events, {int(y.sum())} signal / {int((1-y).sum())} background")
    print(f"  -> {Xp}  ({X.nbytes / 1e6:.0f} MB)")


def load_cache(tag: str):
    Xp, yp, gp, mp = cache_paths(tag)
    if not mp.exists():
        raise FileNotFoundError(f"no cache '{tag}'; run: python team/data.py --tag {tag} ...")
    return np.load(Xp), np.load(yp), np.load(gp), json.loads(mp.read_text())


def main():
    ap = argparse.ArgumentParser(description="build a cached, capped C1 sample")
    ap.add_argument("--tag", required=True, help="cache name, e.g. train300k")
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--n-signal", type=int, default=300_000)
    ap.add_argument("--n-background", type=int, default=300_000)
    ap.add_argument("--n-particles", type=int, default=N_PARTICLES)
    ap.add_argument("--skip-files", type=int, default=0,
                    help="skip the first N parquet fragments per process (disjoint samples)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_cache(a.tag, a.split, a.n_signal, a.n_background, a.n_particles, a.skip_files, a.force)


if __name__ == "__main__":
    main()
