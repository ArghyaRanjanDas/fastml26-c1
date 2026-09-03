"""Split tt by decay mode and ask where the baseline actually loses.

The cached mixture lumps the three tt directories (hadronic, semi-leptonic,
fully leptonic) into one "tt" group, which hides the thing that matters: two of
the three carry a real lepton and a real neutrino.  This builds a small
mode-labelled eval sample with the team loader and reports, per mode,

  * the AUC of the frozen B1e_16p_1M baseline, and
  * the AUC of each physics feature on its own,

so the ranking table can say *which* tt it is beating.  It also states the
mixture: data.py gives each tt directory 1/9 of the background budget, i.e. the
sampled tt is 1/3 hadronic, 1/3 semi-leptonic, 1/3 fully leptonic, whereas
nature gives roughly 46 / 44 / 10.  Anything that keys on the lepton is
therefore *over*-rewarded here; the per-mode split is what makes that visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import DATA_ROOT, stream_process   # noqa: E402
from models import DeepSetPlus               # noqa: E402
import features as ft                        # noqa: E402

MODES = {
    "tt_hadronic": "tt0123j_5f_ckm_LO_MLM_hadronic",
    "tt_semilep": "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
    "tt_leptonic": "tt0123j_5f_ckm_LO_MLM_leptonic",
    "QCD": "QCD_HT250toInf",
    "Wjets": "WJetsToLNu_13TeV-madgraphMLM-pythia8",
}


def grab(directory: str, n: int, split="eval"):
    chunks = list(stream_process(DATA_ROOT / split / directory, n))
    X = np.concatenate([c[0] for c in chunks])
    F = np.concatenate([c[1] for c in chunks])
    return X, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30_000, help="events per process")
    ap.add_argument("--run", default="B1e_16p_1M")
    a = ap.parse_args()

    print(f"loading {a.n} eval events per process ...", flush=True)
    Xs, Fs = grab("HH_4b", a.n)
    data = {k: grab(v, a.n) for k, v in MODES.items()}

    summ = json.loads((HERE.parent / "runs" / f"{a.run}_summary.json").read_text())
    model = DeepSetPlus(n_features=5, n_event_features=11,
                        phi_dims=tuple(summ["phi"]), rho_dims=tuple(summ["rho"]),
                        use_event_features=summ["use_event_features"],
                        event_scale=summ["event_scale"], pool_norm=summ["pool_norm"])
    model.load_state_dict(torch.load(HERE.parent / "runs" / f"{a.run}_best.pt",
                                     map_location="cpu"))
    model.eval()

    @torch.no_grad()
    def score(X, F):
        out = []
        for i in range(0, len(X), 8192):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + 8192]),
                                           torch.from_numpy(F[i:i + 8192]))).numpy())
        return np.concatenate(out)

    s_sig = score(Xs, Fs)
    names, P_sig = ft.compute_chunked(Xs)
    mode_scores = {}

    print(f"\n=== {a.run} (frozen) vs each background process, {a.n} events each ===")
    rows = {}
    feat_tbl = {}
    for name, (Xb, Fb) in data.items():
        s_bkg = score(Xb, Fb)
        s = np.concatenate([s_sig, s_bkg])
        y = np.concatenate([np.ones(len(s_sig)), np.zeros(len(s_bkg))])
        mode_scores[name] = s_bkg
        rows[name] = float(roc_auc_score(y, s))
        print(f"  vs {name:<12s} AUC {rows[name]:.4f}")
        _, P_b = ft.compute_chunked(Xb)
        col = {}
        for i, fn in enumerate(names):
            v = np.concatenate([P_sig[:, i], P_b[:, i]]).astype(np.float64)
            auc = roc_auc_score(y, v)
            col[fn] = max(auc, 1.0 - auc)
        feat_tbl[name] = col

    # Re-weight tt to the Standard Model branching fractions.  data.py samples the
    # three tt directories 1/3 each; nature gives 45.7% all-hadronic, 43.8%
    # semi-leptonic, 10.5% di-leptonic.  Leptonic tt is the *easiest* mode for the
    # baseline, so the evenly-mixed number is optimistic -- this is how optimistic.
    BR = {"tt_hadronic": 0.457, "tt_semilep": 0.438, "tt_leptonic": 0.105}
    rng = np.random.default_rng(0)
    # largest total that keeps every mode within the events actually loaded
    n_tot = min(len(mode_scores[m]) / w for m, w in BR.items())
    parts = [rng.choice(mode_scores[m], size=int(round(n_tot * w)), replace=False)
             for m, w in BR.items()]
    s_tt_phys = np.concatenate(parts)
    s_tt_even = np.concatenate([mode_scores[m] for m in BR])
    for label, s_tt in (("even (as sampled)", s_tt_even), ("SM branching fractions", s_tt_phys)):
        yy = np.concatenate([np.ones(len(s_sig)), np.zeros(len(s_tt))])
        a = float(roc_auc_score(yy, np.concatenate([s_sig, s_tt])))
        rows[f"tt [{label}]"] = a
        print(f"  vs tt, {label:<24s} AUC {a:.4f}")

    order = sorted(names, key=lambda n: -np.mean([feat_tbl[m][n] for m in
                                                  ("tt_hadronic", "tt_semilep", "tt_leptonic")]))
    hdr = " ".join(f"{m:>12s}" for m in MODES)
    print(f"\n=== single-feature AUC per process ===\n{'feature':<18s}{hdr}")
    for n in order:
        print(f"{n:<18s}" + " ".join(f"{feat_tbl[m][n]:12.4f}" for m in MODES))

    (HERE / "ttmodes.json").write_text(json.dumps(
        dict(run=a.run, n_per_process=a.n, model_auc=rows, feature_auc=feat_tbl,
             tt_branching_fractions=BR), indent=1))


if __name__ == "__main__":
    main()
