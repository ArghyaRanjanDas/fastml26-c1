"""Rank the teacher's derived quantities (team/teacher/common.py) by tt gain.

Same protocol as rank.py stage 2 -- gradient-boosted trees on B1e_16p's 11 event
features, with and without the candidate block, differenced -- but the unit is a
*family* of columns rather than one feature, because the teacher's extras are
per-candidate or per-pair, not scalars.

Each family is also priced:
  rich:*        6 derived channels x 16 candidates.  In a student these are extra
                phi() input channels: 5 -> 11 widens phi MACs by ~24%, adds no
                sequential stage.  Summarized here as (leading 4 + mean) per channel.
  pair4:*       ParT pair quantities for the 6 pairs among the leading 4 candidates,
                as event-level scalars injected after the pool: 24 numbers per event,
                zero cost inside phi.
  pair6:ALL     the same for the leading 6 (15 pairs, 60 numbers).
  pairfull      mean/min/max/std over all 120 pairs -- the ceiling a full pairwise
                block could reach, not a trigger proposal.

  python rank_derived.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from rank import get_features, _score, auc_vs, TT   # noqa: E402
import derived                                       # noqa: E402
from data import load_cache                          # noqa: E402


COST = {
    "rich": "phi-width (+24% phi MACs, O(n))",
    "pair4": "24 event scalars (6 pairs x 4)",
    "pair6": "60 event scalars (15 pairs x 4)",
    "pairfull": "full 16x16 table (not trigger-affordable)",
}


def build(tag: str, cache: dict):
    """Family blocks for a cache tag, memoized on disk (they are not cheap on 600k)."""
    npz = HERE / "cache" / f"{tag}_derived.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        return {k: (list(z[f"n_{k}"]), z[f"v_{k}"]) for k in json.loads(str(z["keys"]))}
    X, F, y, g, meta = load_cache(tag)
    t0 = time.perf_counter()
    out = {}
    for i in range(0, len(X), 100_000):
        part = derived.families(X[i:i + 100_000])
        for k, (n, v) in part.items():
            out.setdefault(k, [n, []])[1].append(v)
    fam = {k: (n, np.concatenate(vs)) for k, (n, vs) in out.items()}
    print(f"  derived families for {tag} in {time.perf_counter() - t0:.1f}s", flush=True)
    npz.parent.mkdir(exist_ok=True)
    np.savez(npz, keys=json.dumps(list(fam)),
             **{f"n_{k}": np.array(v[0]) for k, v in fam.items()},
             **{f"v_{k}": v[1] for k, v in fam.items()})
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tag", default="train300k")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--n-fit", type=int, default=300_000)
    ap.add_argument("--max-iter", type=int, default=150)
    a = ap.parse_args()

    names, Ptr, Ftr, ytr, gtr = get_features(a.train_tag)
    _, Pev, Fev, yev, gev = get_features(a.eval_tag)
    Dtr, Dev = build(a.train_tag, {}), build(a.eval_tag, {})

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ytr))[:a.n_fit]
    Ftr_s, ytr_s = Ftr[idx], ytr[idx]

    base = _score(Ftr_s, ytr_s, Fev, yev, gev, max_iter=a.max_iter)
    print(f"baseline (11 event features): tt {base[0]:.4f}  all {base[1]:.4f}", flush=True)

    # c2's own winner, as the reference the teacher's extras have to beat
    iso = names.index("iso_lead_pt")
    blocks = {k: (Dtr[k][1][idx], Dev[k][1]) for k in Dtr}
    blocks["c2:iso_lead_pt"] = (Ptr[idx, iso:iso + 1], Pev[:, iso:iso + 1])
    blocks["c2:iso + rich:ALL"] = (
        np.hstack([Ptr[idx, iso:iso + 1], Dtr["rich:ALL"][1][idx]]),
        np.hstack([Pev[:, iso:iso + 1], Dev["rich:ALL"][1]]))
    blocks["rich:ALL + pair4:ALL"] = (
        np.hstack([Dtr["rich:ALL"][1][idx], Dtr["pair4:ALL"][1][idx]]),
        np.hstack([Dev["rich:ALL"][1], Dev["pair4:ALL"][1]]))
    blocks["c2:iso + rich:ALL + pair4:ALL"] = (
        np.hstack([Ptr[idx, iso:iso + 1], Dtr["rich:ALL"][1][idx], Dtr["pair4:ALL"][1][idx]]),
        np.hstack([Pev[:, iso:iso + 1], Dev["rich:ALL"][1], Dev["pair4:ALL"][1]]))

    rows = []
    for k, (tr, ev) in blocks.items():
        r = _score(Ftr_s, ytr_s, Fev, yev, gev, tr, ev, a.max_iter)
        rows.append(dict(family=k, n_cols=int(tr.shape[1]), auc_tt=r[0], auc_all=r[1],
                         auc_qcd=r[2], auc_w=r[3], d_tt=r[0] - base[0], d_all=r[1] - base[1],
                         cost=COST.get(k.split(":")[0], "mixed")))
        print(f"  {k:<30s} ({tr.shape[1]:3d} cols)  tt {r[0]:.4f} ({r[0]-base[0]:+.4f})  "
              f"all {r[1]:.4f} ({r[1]-base[1]:+.4f})", flush=True)

    rows.sort(key=lambda r: -r["d_tt"])
    (HERE / "stage4_derived.json").write_text(json.dumps(
        dict(baseline=dict(auc_tt=base[0], auc_all=base[1]), n_fit=a.n_fit,
             max_iter=a.max_iter, rows=rows), indent=1))
    print("\n=== teacher-derived families, ranked by gain vs tt ===")
    for r in rows:
        print(f"{r['family']:<30s} {r['n_cols']:3d}  tt {r['auc_tt']:.4f} {r['d_tt']:+.4f}  "
              f"all {r['auc_all']:.4f} {r['d_all']:+.4f}")


if __name__ == "__main__":
    main()
