"""Are `pdgId` and `dxysig` worth adding to the candidate input?

COLUMNS.md found that we use 4 of the 14 `L1T_PUPPIPart` subfields, and two of
the unused ten look like they answer questions this lane spent the day
approximating: `pdgId` flags electrons and muons directly (our best event
feature, `iso_lead_pt`, is a hand-built proxy for a lepton), and `dxysig` is the
impact-parameter *significance*, the variable a real b-tagger uses, where we
feed the network raw `dxy`.

Same protocol as rank.py stage 2: trees on the 11 incumbent event features, then
on those plus a block, differenced. Reads parquet directly because these fields
are not in the cache.

  python newfields.py --n 60000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from data import DATA_ROOT, SIGNAL, BACKGROUND, GROUP_ID, event_features, LABEL_HH  # noqa: E402
from rank import gbdt, auc_vs, TT, QCD, W                                            # noqa: E402

COLS = [f"L1T_PUPPIPart.{f}" for f in
        ("pt", "eta", "phi", "dxy", "dxysig", "pdgId")] + ["label"]
P = 16
DXYSIG_CLIP = 20.0     # float16 overflow: |dxysig| runs to inf, p99 is in the thousands


def grab(directory: Path, n: int):
    out = []
    seen = 0
    for path in sorted(directory.glob(f"{directory.name}_*.parquet")):
        for b in pq.ParquetFile(path).iter_batches(batch_size=20_000, columns=COLS):
            a = ak.from_arrow(b)
            c = a["L1T_PUPPIPart"]
            f = {k: ak.to_numpy(ak.fill_none(ak.pad_none(c[k], P, clip=True), 0)).astype(np.float64)
                 for k in ("pt", "eta", "phi", "dxy", "dxysig", "pdgId")}
            f["label"] = ak.to_numpy(a["label"]).astype(np.int8)
            take = min(len(f["pt"]), n - seen)
            out.append({k: v[:take] for k, v in f.items()})
            seen += take
            if seen >= n:
                return out
    return out


def blocks(f):
    """(baseline 11, dxysig block, pdgId block) for one chunk."""
    pt, eta, phi, dxy = f["pt"], f["eta"], f["phi"], f["dxy"]
    mask = pt > 0
    base = event_features(pt.astype(np.float32), eta.astype(np.float32),
                          phi.astype(np.float32), dxy.astype(np.float32))

    ds = np.abs(np.nan_to_num(f["dxysig"], nan=0.0, posinf=DXYSIG_CLIP, neginf=DXYSIG_CLIP))
    ds = np.clip(ds, 0.0, DXYSIG_CLIP) * mask
    srt = np.sort(ds, axis=1)[:, ::-1]
    ht = np.maximum((pt * mask).sum(1), 1e-6)
    d_names = ["dsig_sum", "dsig_max", "dsig_mean", "dsig_ord2", "dsig_ord3", "dsig_ord4",
               "n_dsig_gt2", "n_dsig_gt3", "n_dsig_gt5", "dsig_ptw"]
    D = np.stack([ds.sum(1), srt[:, 0], ds.sum(1) / 16.0, srt[:, 1], srt[:, 2], srt[:, 3],
                  (ds > 2).sum(1), (ds > 3).sum(1), (ds > 5).sum(1),
                  (pt * ds).sum(1) / ht], axis=1)

    pid = np.abs(f["pdgId"]).astype(int)
    is_e, is_mu = (pid == 11) & mask, (pid == 13) & mask
    lep = is_e | is_mu
    lep_pt = (pt * lep).max(1)
    p_names = ["n_ele", "n_mu", "n_lep", "lead_lep_pt", "lep_pt_frac",
               "n_photon", "n_charged", "n_neutral", "charged_frac"]
    Pb = np.stack([is_e.sum(1), is_mu.sum(1), lep.sum(1), lep_pt, lep_pt / ht,
                   ((pid == 22) & mask).sum(1), ((pid == 211) & mask).sum(1),
                   ((pid == 0) & mask).sum(1),
                   (pt * ((pid == 211) | is_e | is_mu) * mask).sum(1) / ht], axis=1)
    return base, (d_names, D), (p_names, Pb)


def load(split: str, n_sig: int, n_bkg: int):
    Bs, Ds, Ps, ys, gs = [], [], [], [], []
    budgets = [(p, n_sig) for p in SIGNAL] + \
              [(p, int(round(n_bkg * p.weight / 3.0))) for p in BACKGROUND]
    for proc, budget in budgets:
        for f in grab(DATA_ROOT / split / proc.directory, budget):
            b, (dn, D), (pn, Pb) = blocks(f)
            Bs.append(b); Ds.append(D); Ps.append(Pb)
            ys.append((f["label"] == LABEL_HH).astype(np.float32))
            gs.append(np.full(len(b), GROUP_ID[proc.group], dtype=np.int8))
        print(f"  {split}/{proc.directory[:36]:<38s} done", flush=True)
    return (np.concatenate(Bs), (dn, np.concatenate(Ds)), (pn, np.concatenate(Ps)),
            np.concatenate(ys), np.concatenate(gs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60_000, help="events per process budget unit")
    ap.add_argument("--max-iter", type=int, default=150)
    a = ap.parse_args()

    Btr, (dn, Dtr), (pn, Ptr), ytr, gtr = load("train", a.n, a.n)
    Bev, (_, Dev), (_, Pev), yev, gev = load("eval", a.n // 2, a.n // 2)
    print(f"train {Btr.shape} eval {Bev.shape}", flush=True)

    def run(tr, ev, label):
        s = gbdt(tr, ytr, ev, max_iter=a.max_iter)
        r = (float(roc_auc_score(yev, s)), auc_vs(s, yev, gev, TT),
             auc_vs(s, yev, gev, QCD), auc_vs(s, yev, gev, W))
        print(f"  {label:<34s} all {r[0]:.4f}  tt {r[1]:.4f}  qcd {r[2]:.4f}  w {r[3]:.4f}",
              flush=True)
        return r

    base = run(Btr, Bev, "11 event features")
    out = {"baseline": base, "rows": []}
    for label, tr, ev in (
            ("+ dxysig block (10)", np.hstack([Btr, Dtr]), np.hstack([Bev, Dev])),
            ("+ pdgId block (9)", np.hstack([Btr, Ptr]), np.hstack([Bev, Pev])),
            ("+ both (19)", np.hstack([Btr, Dtr, Ptr]), np.hstack([Bev, Dev, Pev]))):
        r = run(tr, ev, label)
        out["rows"].append(dict(block=label, auc_all=r[0], auc_tt=r[1], auc_qcd=r[2],
                                auc_w=r[3], d_all=r[0] - base[0], d_tt=r[1] - base[1]))
    # single-feature AUC vs tt for the new columns
    print("\n  single-feature AUC vs tt:")
    sel = (yev == 1) | (gev == TT)
    singles = {}
    for name, col in list(zip(dn, Dev.T)) + list(zip(pn, Pev.T)):
        v = roc_auc_score(yev[sel], col[sel])
        singles[name] = max(v, 1 - v)
        print(f"    {name:<14s} {singles[name]:.4f}")
    out["singles"] = singles
    (HERE / "stage7_newfields.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
