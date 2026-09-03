"""Is the student limited by what it is told, or by how it is told?

The question from IDEAS.md lane 4: fit trees on event-level scalars only and see
how close they get to the DeepSet.  If scalars alone reach the DeepSet's 0.884,
the per-particle branch is buying almost nothing and the hours belong in
features; if they fall well short, the gap is representational and the hours
belong in the model.

Five fits, all gradient-boosted trees on the same train300k / eval100k split the
DeepSet rows use, so the numbers are directly comparable:

  A  the 11 event features, exactly what B1e_16p gets alongside its particle branch
  B  A + the |dxy| order statistics (2nd-4th largest, plus the 4th/1st ratio)
  C  A + every *trigger-affordable* event scalar measured in this lane
  D  C + the jet-clustered ones (not affordable; the ceiling of hand-made features)
  E  D + summaries of the teacher's per-candidate channels and the leading-4 pair
     distances -- the ceiling of everything this lane can compute per event

  python diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from rank import get_features, gbdt, auc_vs, TT, QCD, W   # noqa: E402
from rank_derived import build                            # noqa: E402
from features import COST                                 # noqa: E402
from sklearn.metrics import roc_auc_score                 # noqa: E402

AFFORDABLE = {"event", "pairwise", "pairwise/2", "pairwise/4", "pair-lead4"}
DXY_ORD = ("dxy_ord2", "dxy_ord3", "dxy_ord4", "dxy_ord4_frac")

# what the 2,057-parameter DeepSet gets on the same eval slice, for reference
REFERENCE = {
    "B1e_16p_1M (2,057 params, 2M events, GPU)": (0.88687, 0.9303, 0.7587, 0.9716),
    "c2_base_cpu (same net, 600k events, CPU)": (0.88397, 0.9292, 0.7514, 0.9712),
    "c2_canon (canonical inputs, 600k, CPU)": (0.90099, 0.9339, 0.7961, 0.9729),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tag", default="train300k")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--n-fit", type=int, default=400_000)
    ap.add_argument("--max-iter", type=int, default=300)
    a = ap.parse_args()

    names, Ptr, Ftr, ytr, gtr = get_features(a.train_tag)
    _, Pev, Fev, yev, gev = get_features(a.eval_tag)
    Dtr, Dev = build(a.train_tag, {}), build(a.eval_tag, {})
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ytr))[:a.n_fit]

    def cols(sel):
        ix = [names.index(n) for n in sel]
        return Ptr[np.ix_(idx, ix)], Pev[:, ix]

    affordable = [n for n in names if COST.get(n, "event") in AFFORDABLE]
    jets = [n for n in names if COST.get(n) == "jets"]
    rich_pair_tr = np.hstack([Dtr["rich:ALL"][1][idx], Dtr["pair4:ALL"][1][idx]])
    rich_pair_ev = np.hstack([Dev["rich:ALL"][1], Dev["pair4:ALL"][1]])

    setups = [
        ("A  11 event features only", None),
        ("B  A + |dxy| order statistics", cols(DXY_ORD)),
        ("C  A + affordable event scalars", cols(affordable)),
        ("D  C + jet-clustered scalars", cols(affordable + jets)),
        ("E  D + rich summaries + pair distances",
         (np.hstack([cols(affordable + jets)[0], rich_pair_tr]),
          np.hstack([cols(affordable + jets)[1], rich_pair_ev]))),
    ]

    rows = []
    print(f"{'setup':<40s} {'cols':>5s} {'AUC all':>9s} {'vs QCD':>8s} {'vs tt':>8s} {'vs W':>8s}")
    for label, blk in setups:
        Xtr = Ftr[idx] if blk is None else np.hstack([Ftr[idx], blk[0]])
        Xev = Fev if blk is None else np.hstack([Fev, blk[1]])
        s = gbdt(Xtr, ytr[idx], Xev, max_iter=a.max_iter)
        r = dict(setup=label, n_cols=int(Xtr.shape[1]),
                 auc_all=float(roc_auc_score(yev, s)),
                 auc_qcd=auc_vs(s, yev, gev, QCD), auc_tt=auc_vs(s, yev, gev, TT),
                 auc_w=auc_vs(s, yev, gev, W))
        rows.append(r)
        print(f"{label:<40s} {r['n_cols']:5d} {r['auc_all']:9.4f} {r['auc_qcd']:8.4f} "
              f"{r['auc_tt']:8.4f} {r['auc_w']:8.4f}", flush=True)

    print()
    for label, (aa, q, t, w) in REFERENCE.items():
        print(f"{label:<40s} {'':5s} {aa:9.4f} {q:8.4f} {t:8.4f} {w:8.4f}")

    (HERE / "stage6_diagnostic.json").write_text(json.dumps(
        dict(n_fit=a.n_fit, max_iter=a.max_iter, rows=rows,
             affordable=affordable, jets=jets, reference=REFERENCE), indent=1))


if __name__ == "__main__":
    main()
