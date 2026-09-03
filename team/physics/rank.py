"""Rank candidate physics features by how much they buy against tt.

Three stages, all on CPU, all off the existing caches (no parquet re-reads):

  stage 1 (alone)     : AUC of the single raw feature, HH_4b vs tt.
  stage 2 (marginal)  : gradient-boosted trees on B1e_16p's 11 event features,
                        with and without the candidate feature -- the honest
                        "does it add anything" test, since half of these
                        correlate strongly with HT and the leading pTs.
  stage 3 (greedy)    : forward selection on top of the 11, so correlated
                        features do not all get credit for the same handle.

The GBDT is a stand-in for rho(), not the final model: it answers "is the
information there" at a few seconds per fit instead of a few minutes.

  python rank.py alone
  python rank.py marginal
  python rank.py greedy --k 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import load_cache, GROUP_ID, EVENT_FEATURES   # noqa: E402
import features as ft                                   # noqa: E402

HERE = Path(__file__).resolve().parent
FCACHE = HERE / "cache"
TT, QCD, W = GROUP_ID["tt"], GROUP_ID["QCD"], GROUP_ID["Wjets"]


def get_features(tag: str):
    """Physics features for a cache tag, computed once and memoized on disk."""
    FCACHE.mkdir(exist_ok=True)
    npy, js = FCACHE / f"{tag}_phys.npy", FCACHE / f"{tag}_phys.json"
    X, F, y, g, meta = load_cache(tag)
    if npy.exists() and js.exists():
        names = json.loads(js.read_text())
        P = np.load(npy)
        if len(P) == len(X):
            return names, P, F, y, g
    t0 = time.perf_counter()
    names, P = ft.compute_chunked(X, verbose=True)
    print(f"  computed {P.shape} physics features in {time.perf_counter() - t0:.1f}s", flush=True)
    np.save(npy, P)
    js.write_text(json.dumps(names, indent=1))
    return names, P, F, y, g


def auc_vs(scores, y, g, group_id):
    sel = (y == 1) | (g == group_id)
    return float(roc_auc_score(y[sel], scores[sel]))


def oriented(scores, y, g, group_id):
    """AUC of the best-oriented single feature (sign is a free parameter)."""
    a = auc_vs(scores, y, g, group_id)
    return max(a, 1.0 - a), ("+" if a >= 0.5 else "-")


def gbdt(Xtr, ytr, Xev, seed=0, max_iter=150):
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
        early_stopping=False, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xev)[:, 1]


def stage_alone(args):
    names, P, F, y, g = get_features(args.eval_tag)
    rows = []
    for i, n in enumerate(names):
        v = P[:, i].astype(np.float64)
        a_tt, sgn = oriented(v, y, g, TT)
        a_all = roc_auc_score(y, v if sgn == "+" else -v)
        a_qcd, _ = oriented(v, y, g, QCD)
        a_w, _ = oriented(v, y, g, W)
        rows.append(dict(feature=n, dir=sgn, auc_tt=a_tt, auc_all=float(a_all),
                         auc_qcd=a_qcd, auc_w=a_w))
    # the 11 incumbents, for scale
    for i, n in enumerate(EVENT_FEATURES):
        v = F[:, i].astype(np.float64)
        a_tt, sgn = oriented(v, y, g, TT)
        a_all = roc_auc_score(y, v if sgn == "+" else -v)
        rows.append(dict(feature=f"[base] {n}", dir=sgn, auc_tt=a_tt, auc_all=float(a_all),
                         auc_qcd=oriented(v, y, g, QCD)[0], auc_w=oriented(v, y, g, W)[0]))
    rows.sort(key=lambda r: -r["auc_tt"])
    print(f"\n{'feature':<18s} {'dir':>3s} {'AUC vs tt':>10s} {'vs QCD':>8s} {'vs W':>8s} {'vs all':>8s}")
    for r in rows:
        print(f"{r['feature']:<18s} {r['dir']:>3s} {r['auc_tt']:10.4f} "
              f"{r['auc_qcd']:8.4f} {r['auc_w']:8.4f} {r['auc_all']:8.4f}")
    (HERE / "stage1_alone.json").write_text(json.dumps(rows, indent=1))


def _data(args):
    names, Ptr, Ftr, ytr, gtr = get_features(args.train_tag)
    _, Pev, Fev, yev, gev = get_features(args.eval_tag)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ytr))[:args.n_fit]
    return (names, Ptr[idx], Ftr[idx], ytr[idx], Pev, Fev, yev, gev)


def _score(Ftr, ytr, Fev, yev, gev, cols_tr=None, cols_ev=None, max_iter=150):
    A = Ftr if cols_tr is None else np.hstack([Ftr, cols_tr])
    B = Fev if cols_ev is None else np.hstack([Fev, cols_ev])
    s = gbdt(A, ytr, B, max_iter=max_iter)
    return (auc_vs(s, yev, gev, TT), float(roc_auc_score(yev, s)),
            auc_vs(s, yev, gev, QCD), auc_vs(s, yev, gev, W))


def stage_marginal(args):
    names, Ptr, Ftr, ytr, Pev, Fev, yev, gev = _data(args)
    t0 = time.perf_counter()
    base = _score(Ftr, ytr, Fev, yev, gev, max_iter=args.max_iter)
    print(f"baseline (11 event features): AUC vs tt {base[0]:.4f}  overall {base[1]:.4f}  "
          f"({time.perf_counter() - t0:.1f}s/fit)", flush=True)
    rows = []
    for i, n in enumerate(names):
        r = _score(Ftr, ytr, Fev, yev, gev, Ptr[:, i:i + 1], Pev[:, i:i + 1], args.max_iter)
        rows.append(dict(feature=n, auc_tt=r[0], auc_all=r[1], auc_qcd=r[2], auc_w=r[3],
                         d_tt=r[0] - base[0], d_all=r[1] - base[1]))
        print(f"  +{n:<18s} tt {r[0]:.4f} ({r[0]-base[0]:+.4f})  "
              f"all {r[1]:.4f} ({r[1]-base[1]:+.4f})", flush=True)
    rows.sort(key=lambda r: -r["d_tt"])
    out = dict(baseline=dict(auc_tt=base[0], auc_all=base[1], auc_qcd=base[2], auc_w=base[3]),
               n_fit=args.n_fit, max_iter=args.max_iter, rows=rows)
    (HERE / "stage2_marginal.json").write_text(json.dumps(out, indent=1))
    print("\n=== ranked by gain in AUC vs tt, on top of the 11 baseline features ===")
    print(f"{'feature':<18s} {'AUC tt':>8s} {'d tt':>8s} {'AUC all':>8s} {'d all':>8s}")
    for r in rows:
        print(f"{r['feature']:<18s} {r['auc_tt']:8.4f} {r['d_tt']:+8.4f} "
              f"{r['auc_all']:8.4f} {r['d_all']:+8.4f}")


def stage_greedy(args):
    names, Ptr, Ftr, ytr, Pev, Fev, yev, gev = _data(args)
    pool = list(range(len(names)))
    if args.pool_from:
        # forward selection over every feature is 39*k fits; restricting the pool
        # to the best of stage 2 costs nothing real -- a feature that adds nothing
        # on its own to the 11 does not suddenly add something next to another one.
        top = [r["feature"] for r in json.loads((HERE / "stage2_marginal.json").read_text())["rows"]]
        keep = set(top[:args.pool_from])
        pool = [i for i, n in enumerate(names) if n in keep]
        print(f"greedy pool ({len(pool)}): {[names[i] for i in pool]}", flush=True)
    chosen, hist = [], []
    cur = _score(Ftr, ytr, Fev, yev, gev, max_iter=args.max_iter)
    print(f"start: tt {cur[0]:.4f}  all {cur[1]:.4f}", flush=True)
    for step in range(args.k):
        best = None
        for i in pool:
            if i in chosen:
                continue
            n = names[i]
            cols = chosen + [i]
            r = _score(Ftr, ytr, Fev, yev, gev, Ptr[:, cols], Pev[:, cols], args.max_iter)
            if best is None or r[0] > best[1][0]:
                best = (i, r)
        i, r = best
        chosen.append(i)
        hist.append(dict(step=step + 1, added=names[i], auc_tt=r[0], auc_all=r[1],
                         auc_qcd=r[2], auc_w=r[3], d_tt=r[0] - cur[0], d_all=r[1] - cur[1]))
        print(f"step {step+1}: + {names[i]:<18s} tt {r[0]:.4f} ({r[0]-cur[0]:+.4f})  "
              f"all {r[1]:.4f} ({r[1]-cur[1]:+.4f})", flush=True)
        cur = r
    (HERE / "stage3_greedy.json").write_text(json.dumps(
        dict(n_fit=args.n_fit, max_iter=args.max_iter, selected=[names[i] for i in chosen],
             history=hist), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["alone", "marginal", "greedy"])
    ap.add_argument("--train-tag", default="train300k")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--n-fit", type=int, default=300_000)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--pool-from", type=int, default=0,
                    help="greedy: only consider the top-N features of stage2_marginal.json")
    a = ap.parse_args()
    {"alone": stage_alone, "marginal": stage_marginal, "greedy": stage_greedy}[a.stage](a)


if __name__ == "__main__":
    main()
