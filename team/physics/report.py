"""Render the c2 tt-study section of RESULTS.md from the stage JSON files.

Regenerating instead of hand-writing keeps the numbers in RESULTS.md identical
to the numbers the scripts produced.  Writes to stdout; the caller appends.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features import COST   # noqa: E402


def cost(name):
    return COST.get(name.replace("[base] ", ""), "event")


START = "<!-- c2:tt-study:start -->"
END = "<!-- c2:tt-study:end -->"


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def load_greedy():
    """stage3_greedy.json, or the log if the search was stopped early on purpose."""
    j = load("stage3_greedy.json")
    if j:
        return j
    log = HERE / "greedy.log"
    if not log.exists():
        return None
    pat = re.compile(r"^step (\d+): \+ (\S+)\s+tt ([\d.]+) \(([-+][\d.]+)\)\s+"
                     r"all ([\d.]+) \(([-+][\d.]+)\)")
    hist = []
    for line in log.read_text().splitlines():
        m = pat.match(line)
        if m:
            hist.append(dict(step=int(m[1]), added=m[2], auc_tt=float(m[3]),
                             d_tt=float(m[4]), auc_all=float(m[5]), d_all=float(m[6])))
    return dict(history=hist, selected=[h["added"] for h in hist], truncated=True) if hist else None


RUNS = HERE.parent / "runs"

STUDENTS = [
    ("c2_base_cpu", "control: 11 event features, 5 channels/candidate", "—"),
    ("c2_meanmax", "+ max-pool alongside mean", "comparators, no DSP"),
    ("c2_pair4", "+ 24 leading-4 pair scalars (ln ΔR / kT / z / m²)", "24 event scalars"),
    ("c2_rich", "+ 6 teacher per-candidate channels (φ sees 11)", "+24% φ MACs"),
    ("c2_rich_mm", "+ those channels AND max-pool", "+24% φ MACs"),
    ("c2_ttfeat", "+ c2's 3 tt features", "see above"),
]


def run_row(tag):
    p = RUNS / f"{tag}_summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d


def main():
    s1, s2, s3, tt = load("stage1_alone.json"), load("stage2_marginal.json"), \
        load_greedy(), load("ttmodes.json")
    s4 = load("stage4_derived.json")
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
        if s3.get("truncated"):
            A("<sub>Stopped after step 3: the three features it had picked are the three that")
            A("go into `data.py`, and the CPU was better spent on the student runs below.</sub>\n")
        A("| step | added | cost | AUC vs tt | Δ | AUC overall | Δ |")
        A("|---|---|---|---|---|---|---|")
        for h in s3["history"]:
            A(f"| {h['step']} | `{h['added']}` | {cost(h['added'])} | {h['auc_tt']:.4f} | "
              f"{h['d_tt']:+.4f} | {h['auc_all']:.4f} | {h['d_all']:+.4f} |")
        A("")

    if s4:
        b = s4["baseline"]
        A("## Stage 4 — the teacher's derived quantities, priced\n")
        A("`team/teacher/common.py` gives the teacher two blocks the student does not have:")
        A("6 derived quantities per candidate (`rich`), and ParT's 4 quantities per *pair*.")
        A("First, a correction worth having on the record: **the teacher that produced the")
        A("published soft targets (`ds_big_s0`, 0.9151 overall / 0.8261 vs tt) uses no pairwise")
        A("features at all.** It is `BigDeepSet(rich=True)` — per-candidate channels, mean+max")
        A("pooling, 72k parameters, 40 epochs on 2M events. `pair_features()` belongs to")
        A("`ParTLite`, and no ParT run has been published. So the 0.826 − 0.759 gap is not yet")
        A("evidence about relational information.\n")
        A("Same GBDT protocol as stage 2, one row per *family* of columns:\n")
        A(f"**Baseline (11 event features): AUC vs tt {b['auc_tt']:.4f}, overall {b['auc_all']:.4f}.**\n")
        A("| + family | cols | cost for the student | AUC vs tt | Δ vs tt | AUC overall | Δ overall |")
        A("|---|---|---|---|---|---|---|")
        for r in s4["rows"]:
            A(f"| `{r['family']}` | {r['n_cols']} | {r['cost']} | {r['auc_tt']:.4f} | "
              f"{r['d_tt']:+.4f} | {r['auc_all']:.4f} | {r['d_all']:+.4f} |")
        A("")

    rows = [(t, d, c, run_row(t)) for t, d, c in STUDENTS]
    if any(r[3] for r in rows):
        A("## Stage 5 — the same questions asked of the real 2k student\n")
        A("The GBDT above has no particle branch, so it over-states anything φ() already")
        A("computes. These are the actual B1e_16p architecture (φ 32-16-8, ρ 32-16, pool BN,")
        A("event scale 0.2, 25 epochs, seed 0, `train300k`), CPU, one input change per row.\n")
        A("The FPGA lane has just priced the baseline at ap_fixed<22,10>: **319k LUT / 1,724 DSP,")
        A("91% of one SLR** on both. There is no headroom, so where a feature lands matters as much")
        A("as what it is worth. Multiply-accumulates per event, at φ 32-16-8 / ρ 32-16:\n")
        A("| block | MACs/event | note |")
        A("|---|---|---|")
        A("| φ, 5 input channels (baseline) | 12,800 | 16 × 800, replicated per candidate — the FPGA bill |")
        A("| φ, 11 input channels (teacher `rich`) | 15,872 | **+3,072 (+24%)**, straight onto the block that is already at 91% |")
        A("| ρ, 19 inputs (baseline) | 1,136 | once per event |")
        A("| ρ, +24 pair scalars | 1,904 | +768, i.e. +6% of φ — after the pool, so it never replicates |")
        A("| ρ, +8 from max-pooling | 1,392 | +256; the max itself is comparators, no DSP |")
        A("| 16×16 ΔR table (isolation) | ~512 mults | cos Δφ = c_i c_j + s_i s_j from inputs already there |")
        A("")
        A("| run | change | φ cost | params | AUC (eval) | vs tt | vs QCD | vs W+jets |")
        A("|---|---|---|---|---|---|---|---|")
        for tag, desc, cost_s, d in rows:
            if not d:
                continue
            g = d["per_background_auc"]
            A(f"| `{tag}` | {desc} | {cost_s} | {d['params']:,} | {d['eval_auc']:.5f} | "
              f"{g['tt']:.4f} | {g['QCD']:.4f} | {g['Wjets']:.4f} |")
        A("")

    text = "\n".join(out)
    if "--write" not in sys.argv:
        print(text)
        return

    # Own exactly one block of RESULTS.md, delimited by markers, so regenerating
    # never touches anyone else's rows.
    md = HERE.parent / "RESULTS.md"
    body = md.read_text()
    block = f"{START}\n{text}\n{END}\n"
    if START in body and END in body:
        head, rest = body.split(START, 1)
        body = head + block + rest.split(END, 1)[1].lstrip("\n")
    else:
        body = body.rstrip("\n") + "\n\n---\n\n" + block
    md.write_text(body)
    print(f"wrote {len(text.splitlines())} lines into {md}")


if __name__ == "__main__":
    main()
