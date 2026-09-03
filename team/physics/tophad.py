"""Hadronic tt is the remaining gap. Can a top tag built on candidates close it?

The baseline is 0.723 against all-hadronic tt and 0.795 against fully leptonic:
the lepton features fixed the easy modes and left the hard one. All-hadronic tt
is a W -> qq inside a t -> Wb, twice, so the obvious handles are a dijet mass
near 80 and a trijet mass near 173.

The cheap version works on the leading 6 *candidates* rather than on clustered
jets: 15 pair masses, and then every 3-candidate mass is a sum of three of them
(m_ijk^2 = m_ij^2 + m_ik^2 + m_jk^2 for massless constituents), so the 20 triples
cost adds only.

Everything is measured on top of the canonical 19 event features, split by tt
decay mode, because a feature that only helps leptonic tt is a feature we
already have.

  python tophad.py --n 30000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from data import DATA_ROOT, load_cache, stream_process   # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
import features as ft                                     # noqa: E402

PROCS = {
    "tt_hadronic": "tt0123j_5f_ckm_LO_MLM_hadronic",
    "tt_semilep": "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
    "tt_leptonic": "tt0123j_5f_ckm_LO_MLM_leptonic",
    "QCD": "QCD_HT250toInf",
    "Wjets": "WJetsToLNu_13TeV-madgraphMLM-pythia8",
}
TOP6 = ["dm_W6", "m_W6", "mjj_max6", "dm_top6", "m_top6", "dm_Wtop6"]
DXY_ORD = ["dxy_ord2", "dxy_ord3", "dxy_ord4", "dxy_ord4_frac"]


def phys(X, names):
    """Named physics features for a cache tensor, in chunks (the pair table is big)."""
    cols = []
    for i in range(0, len(X), 100_000):
        f = ft.compute(X[i:i + 100_000, :, :5])
        cols.append(np.stack([f[n] for n in names], axis=1))
    return np.concatenate(cols)


def grab(directory, n):
    chunks = list(stream_process(DATA_ROOT / "eval" / directory, n, extra=True))
    return (np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--n-fit", type=int, default=400_000)
    a = ap.parse_args()

    Xtr, Ftr, ytr, gtr, _ = load_cache("train300k_c2")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ytr))[:a.n_fit]
    Xtr, Ftr, ytr = Xtr[idx], Ftr[idx], ytr[idx]
    Ntr = phys(Xtr, TOP6 + DXY_ORD)
    print(f"train {Ftr.shape} + {Ntr.shape}", flush=True)

    Xs, Fs = grab("HH_4b", a.n)
    Ns = phys(Xs, TOP6 + DXY_ORD)
    ev = {}
    for k, v in PROCS.items():
        Xb, Fb = grab(v, a.n)
        ev[k] = (Fb, phys(Xb, TOP6 + DXY_ORD))
        print(f"  eval {k} {Fb.shape}", flush=True)

    top_i = [0, 1, 2, 3, 4, 5]
    dxy_i = [6, 7, 8, 9]
    blocks = {
        "A  canonical 19": [],
        "B  + top6 (6 cols)": top_i,
        "C  + dxy order stats (4 cols)": dxy_i,
        "D  + both (10 cols)": top_i + dxy_i,
    }

    print(f"\n{'setup':<32s}" + "".join(f"{k:>13s}" for k in PROCS) + f"{'all bkg':>10s}")
    out = {}
    for label, ix in blocks.items():
        tr = Ftr if not ix else np.hstack([Ftr, Ntr[:, ix]])

        def model_in(F, N, ix=ix):
            return F if not ix else np.hstack([F, N[:, ix]])

        # one fit, then predict on every process separately
        clf = HistGradientBoostingClassifier(max_iter=a.max_iter, learning_rate=0.1,
                                             max_leaf_nodes=31, early_stopping=False,
                                             random_state=0).fit(tr, ytr)
        s_sig = clf.predict_proba(model_in(Fs, Ns))[:, 1]
        row, allb = {}, []
        for k, (Fb, Nb) in ev.items():
            s_b = clf.predict_proba(model_in(Fb, Nb))[:, 1]
            y = np.concatenate([np.ones(len(s_sig)), np.zeros(len(s_b))])
            row[k] = float(roc_auc_score(y, np.concatenate([s_sig, s_b])))
            allb.append(s_b)
        s_all = np.concatenate(allb)
        y = np.concatenate([np.ones(len(s_sig)), np.zeros(len(s_all))])
        row["all"] = float(roc_auc_score(y, np.concatenate([s_sig, s_all])))
        out[label] = row
        print(f"{label:<32s}" + "".join(f"{row[k]:13.4f}" for k in PROCS) +
              f"{row['all']:10.4f}", flush=True)

    print("\nsingle-feature AUC vs each tt mode")
    print(f"{'feature':<16s}" + "".join(f"{k:>13s}" for k in PROCS))
    singles = {}
    for j, name in enumerate(TOP6 + DXY_ORD):
        r = {}
        for k, (Fb, Nb) in ev.items():
            v = np.concatenate([Ns[:, j], Nb[:, j]]).astype(np.float64)
            y = np.concatenate([np.ones(len(Ns)), np.zeros(len(Nb))])
            auc = roc_auc_score(y, v)
            r[k] = max(auc, 1 - auc)
        singles[name] = r
        print(f"{name:<16s}" + "".join(f"{r[k]:13.4f}" for k in PROCS))

    (HERE / "stage8_tophad.json").write_text(json.dumps(
        dict(blocks=out, singles=singles, n_per_process=a.n, n_fit=a.n_fit), indent=1))


if __name__ == "__main__":
    main()
