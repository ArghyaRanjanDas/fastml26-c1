"""Render the c2 tt-study section of RESULTS.md from the stage JSON files.

Regenerating instead of hand-writing keeps the numbers in RESULTS.md identical
to the numbers the scripts produced.  Writes to stdout; the caller appends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import COST   # noqa: E402


def cost(name):
    return COST.get(name.replace("[base] ", ""), "event")


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    s1, s2, s3, tt = load("stage1_alone.json"), load("stage2_marginal.json"), \
        load("stage3_greedy.json"), load("ttmodes.json")
    out = []
    A = out.append

    A("# Task 4 (c2): the tt background\n")
    A("`B1e_16p_1M` is 0.930 vs QCD and 0.972 vs W+jets but **0.759 vs tt**. Everything below")
    A("asks one question: what else is in the 16 PUPPI candidates that separates HH→4b from tt?\n")
    A("All of it runs on CPU off the existing `team/cache` tensors — the cached `X` is an")
    A("invertible transform of (pt, η, φ, dxy), so `physics/features.py:decode()` recovers")
    A("physical units without re-reading a single parquet file. 39 candidate features,")
    A("`physics/rank.py` for the ranking, eval slice = `eval100k` throughout.\n")

    if tt:
        A("## The mixture, and which tt actually hurts\n")
        A("`data.py` gives each of the three tt directories 1/9 of the background budget, so the")
        A("sampled tt is **1/3 hadronic, 1/3 semi-leptonic, 1/3 fully leptonic**. Nature gives")
        A("roughly 45.7 / 43.8 / 10.5. Splitting the frozen baseline by mode "
          f"({tt['n_per_process']:,} eval events each):\n")
        A("| background | `B1e_16p_1M` AUC |")
        A("|---|---|")
        for k in ("tt_hadronic", "tt_semilep", "tt_leptonic"):
            A(f"| {k} | {tt['model_auc'][k]:.4f} |")
        for k in tt["model_auc"]:
            if k.startswith("tt ["):
                A(f"| **{k}** | **{tt['model_auc'][k]:.4f}** |")
        for k in ("QCD", "Wjets"):
            A(f"| {k} | {tt['model_auc'][k]:.4f} |")
        A("")
        A("The baseline is worst against **all-hadronic tt**, the mode with no lepton and no")
        A("neutrino — exactly the tt that looks like four b-jets. Because the sampled mixture")
        A("over-represents the *easy* leptonic mode by 3×, the headline 0.759 is optimistic:")
        A("re-weighted to Standard Model branching fractions it drops (row above). Worth saying")
        A("out loud on Friday, and worth remembering when reading every gain below.\n")

    if tt and s1:
        A("### Which feature works on which tt\n")
        A("Single-feature AUC, signal vs that process alone. Read the first three columns")
        A("together: `iso_lead_pt` is a *lepton* tag (0.59 → 0.82 across the modes), while the")
        A("dxy features are flat across modes and are the only handle that works on the")
        A("hadronic mode. Anything that beats hadronic tt has to come from b-content, jet")
        A("counting or mass structure, not from the lepton.\n")
        modes = ("tt_hadronic", "tt_semilep", "tt_leptonic", "QCD", "Wjets")
        A("| feature | cost | " + " | ".join(modes) + " |")
        A("|---" * (len(modes) + 2) + "|")
        top = [r["feature"] for r in s1 if not r["feature"].startswith("[base]")][:12]
        for n in top:
            A(f"| `{n}` | {cost(n)} | " +
              " | ".join(f"{tt['feature_auc'][m][n]:.4f}" for m in modes) + " |")
        A("")

    if s1:
        A("## Stage 1 — each feature alone (HH→4b vs tt, eval slice)\n")
        A("Sign is a free parameter, so the AUC is reported oriented (`dir` = whether signal")
        A("sits high or low). The 11 incumbent event features are included for scale.\n")
        A("| feature | cost | dir | AUC vs tt | vs QCD | vs W+jets | vs all bkg |")
        A("|---|---|---|---|---|---|---|")
        for r in s1[:20]:
            A(f"| `{r['feature']}` | {cost(r['feature'])} | {r['dir']} | {r['auc_tt']:.4f} | "
              f"{r['auc_qcd']:.4f} | {r['auc_w']:.4f} | {r['auc_all']:.4f} |")
        A(f"\n<sub>Full table of all {len(s1)} rows: `physics/stage1_alone.json`.</sub>\n")

    if s2:
        b = s2["baseline"]
        A("## Stage 2 — what each feature *adds* to the 11 baseline features\n")
        A("A single feature scoring 0.67 alone is worthless if HT already carries it. So: fit")
        A("gradient-boosted trees on B1e_16p's 11 event features, then on the 11 + one candidate,")
        A(f"and difference. ({s2['n_fit']:,} train events, {s2['max_iter']} trees; the GBDT is a")
        A("stand-in for ρ() at 20 s/fit instead of 20 min.)\n")
        A(f"**Baseline (11 event features, no particle branch): AUC vs tt {b['auc_tt']:.4f}, "
          f"overall {b['auc_all']:.4f}.**\n")
        A("`cost` is what the feature needs in firmware: **event** = O(16) reductions,")
        A("essentially free; **pairwise** = the 16x16 ΔR table (~512 multiplies, ~4% of φ());")
        A("**jets** = 6 sequential cone-clustering passes, expensive in latency.\n")
        A("| + feature | cost | AUC vs tt | Δ vs tt | AUC overall | Δ overall |")
        A("|---|---|---|---|---|---|")
        for r in s2["rows"]:
            A(f"| `{r['feature']}` | {cost(r['feature'])} | {r['auc_tt']:.4f} | {r['d_tt']:+.4f} | "
              f"{r['auc_all']:.4f} | {r['d_all']:+.4f} |")
        A("")

    if s3:
        A("## Stage 3 — greedy forward selection\n")
        A("The top of stage 2 is three flavours of the same handle (an isolated hard candidate),")
        A("so ranking alone over-counts it. Forward selection, each step keeping the feature that")
        A("adds most against tt:\n")
        A("| step | added | cost | AUC vs tt | Δ | AUC overall | Δ |")
        A("|---|---|---|---|---|---|---|")
        for h in s3["history"]:
            A(f"| {h['step']} | `{h['added']}` | {cost(h['added'])} | {h['auc_tt']:.4f} | "
              f"{h['d_tt']:+.4f} | {h['auc_all']:.4f} | {h['d_all']:+.4f} |")
        A("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
