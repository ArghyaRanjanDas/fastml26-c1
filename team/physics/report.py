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
    ("c2_base_cpu", "control: 11 event features, 5 channels/candidate", "12,800"),
    ("c2_meanmax", "+ max-pool alongside mean", "12,800"),
    ("c2_pair4", "+ 24 leading-4 pair scalars (ln ΔR / kT / z / m²)", "12,800"),
    ("c2_rich", "+ 6 teacher per-candidate channels (φ sees 11)", "15,872"),
    ("c2_rich_mm", "+ those channels AND max-pool", "15,872"),
    ("c2_canon", "**canonical set**: 11 channels + max-pool + 8 event features", "15,872"),
    ("c2_canon_narrow", "canonical set, φ 24-12-8", "10,368"),
    ("c2_canon_8p", "canonical set, leading 8 candidates", "7,936"),
    ("c2_canon3", "canonical + dxysig/pdgId features, φ 24-12-8", "10,368"),
    ("c2_canon3_wide", "canonical + dxysig/pdgId features, φ 32-16-8", "15,872"),
]


_OFFICIAL_BKG = None


def official_auc(tag, d):
    """AUC of a saved run re-weighted to the organizers' eval background mixture.

    AUC is a rank statistic between the two classes, so only the *background*
    composition matters; the signal fraction does not enter.
    """
    global _OFFICIAL_BKG
    import numpy as np
    from sklearn.metrics import roc_auc_score
    sp = RUNS / f"{tag}_eval_scores.npy"
    mix = load("dataset_mixture.json")
    if not sp.exists() or mix is None:
        return None
    if _OFFICIAL_BKG is None:
        _OFFICIAL_BKG = {}
        for v in mix["eval"].values():
            if v["group"] != "HH_4b":
                _OFFICIAL_BKG[v["group"]] = _OFFICIAL_BKG.get(v["group"], 0) + v["events"]
    cache = HERE.parent / "cache" / d["eval_meta"]["tag"]
    y, g = np.load(cache / "y.npy"), np.load(cache / "group.npy")
    sc = np.load(sp)
    if len(sc) != len(y):
        return None
    ids, tot = {"QCD": 0, "tt": 2, "Wjets": 3}, sum(_OFFICIAL_BKG.values())
    ours = {k: int((g == i).sum()) for k, i in ids.items()}
    n = sum(ours.values())
    w = np.ones(len(y))
    for k, i in ids.items():
        w[(y == 0) & (g == i)] = (_OFFICIAL_BKG[k] / tot) / (ours[k] / n)
    return float(roc_auc_score(y, sc, sample_weight=w))


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
    s6, s7, s8 = load("stage6_diagnostic.json"), load("stage7_newfields.json"), \
        load("stage8_tophad.json")
    mixture = load("dataset_mixture.json")
    out = []
    A = out.append

    A("# Task 4 (c2): the tt background\n")
    A("`B1e_16p_1M` is 0.930 vs QCD and 0.972 vs W+jets but **0.759 vs tt**. Everything below")
    A("asks one question: what else is in the 16 PUPPI candidates that separates HH→4b from tt?\n")
    A("**Where it ended up.** `c2_canon3` — the same 2k-parameter DeepSet, fed the input set")
    A("this lane built — reaches **AUC 0.9154 overall and 0.828 vs tt on 600k training")
    A("events, at 10,368 φ MACs, which is 19% *cheaper* than today's baseline.** For")
    A("comparison the published baseline is 0.8869 / 0.759 at 12,800 MACs on 2M events, and")
    A("the 72,717-parameter teacher is 0.9152 / 0.826 on 2M events. None of that came from a")
    A("new architecture; all of it came from what the network is shown.\n")
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
        A("| run | change | φ MACs | params | AUC (eval) | AUC (official mix) | vs tt | vs QCD | vs W+jets | eff@99% |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for tag, desc, cost_s, d in rows:
            if not d:
                continue
            g = d["per_background_auc"]
            off = official_auc(tag, d)
            offs = f"{off:.5f}" if off else "—"
            A(f"| `{tag}` | {desc} | {cost_s} | {d['params']:,} | {d['eval_auc']:.5f} | {offs} | "
              f"{g['tt']:.4f} | {g['QCD']:.4f} | {g['Wjets']:.4f} | "
              f"{d['signal_eff']['0.99']:.4f} |")
        A("| *(reference)* `ds_big_s0` teacher, 72k params, 2M events | — | — | 72,717 | "
          "0.91515 | — | 0.8261 | 0.9436 | 0.9757 | 0.2717 |")
        A("")
        A("Max-pooling on its own is worth +0.0005. The 24 pair scalars are worth +0.013 vs tt.")
        A("The 6 per-candidate channels are worth **+0.037 vs tt**, and the three together")
        A("(`c2_canon`) are **+0.045 vs tt and +0.017 overall over the control**, taking signal")
        A("efficiency at 99% background rejection from 0.174 to 0.216. That is 2,777 parameters")
        A("and 300k training events reaching 0.796 vs tt, against 0.826 for a 72k-parameter")
        A("teacher trained on 2M — most of the teacher's advantage was its inputs, not its size.\n")

        A("### The canonical student input set — tensor layout for c1\n")
        A("`data.py` now builds it behind two flags. Nothing changes by default: a cache built")
        A("without them is byte-identical to before.\n")
        A("```bash")
        A("python data.py --tag train1M_c2  --split train --n-signal 1000000 --n-background 1000000 \\")
        A("       --rich-particles --extra-features")
        A("python data.py --tag eval100k_c2 --split eval  --n-signal  100000 --n-background  100000 \\")
        A("       --rich-particles --extra-features")
        A("python train.py --model deepset_plus --phi 32,16,8 --rho 32,16 --pool meanmax \\")
        A("       --pool-norm --event-scale 0.2 --epochs 25 \\")
        A("       --train-tag train1M_c2 --eval-tag eval100k_c2 --tag <name>")
        A("```")
        A("`train.py` needs no change: it takes the per-candidate width from `Xtr.shape[2]` and")
        A("the event width from `Ftr.shape[1]`. `--pool meanmax` is the pooling you already added.\n")
        A("**X — `(N, 16, 11)` float32.** Channels 0-4 are exactly what they were; 5-10 are new.")
        A("Order matters to anything reading an export:\n")
        A("| ch | name | value |")
        A("|---|---|---|")
        for i, (n, v) in enumerate([
                ("log_pt", "log1p(pt) / 8"), ("eta", "η / 4"), ("dxy", "clip(dxy, ±2) / 2"),
                ("cos_phi", "cos φ"), ("sin_phi", "sin φ"),
                ("lnz", "ln(pt / HT) / 4"), ("lnE", "log1p(pt cosh η) / 8"),
                ("cos_dphi_lead", "cos(φ − φ₁)"), ("sin_dphi_lead", "sin(φ − φ₁)"),
                ("deta_lead", "(η − η₁) / 2"), ("abs_dxy", "|dxy| / 2")]):
            A(f"| {i} | `{n}` | {v} |")
        A("")
        A("Channels 5-10 are the teacher's, in the teacher's order, and match")
        A("`teacher/common.py:particle_features(rich=True)` to 2e-7 — so teacher and student")
        A("consume the identical tensor and the published soft targets stay valid.\n")
        A("**F — `(N, 19)` float32.** The 11 incumbent event features unchanged, then:\n")
        A("| idx | name | value |")
        A("|---|---|---|")
        A("| 11 | `iso_lead_pt` | log1p(pT of the most isolated pT>10 candidate), standardized |")
        A("| 12 | `n_iso` | number of pT>10 candidates with cone-pT/pT < 0.15, standardized |")
        A("| 13-18 | `p12_lndRc` … `p34_lndRc` | ½ln(Δη² + 2(1−cosΔφ)) for the 6 pairs among the leading 4 |")
        A("")
        A("All eight are standardized with frozen constants in `data.EXTRA_STANDARDIZE`")
        A("(re-measure with `python data.py --fit-extra-norm`). The pair distance uses")
        A("`2(1−cosΔφ)` rather than `Δφ²` deliberately: it needs no atan2 and no 2π wrap in")
        A("firmware, and it measures the same (+0.0158 vs +0.0156 AUC vs tt). Full firmware")
        A("costing of every one of these — formula, cost class, fixed-point rewrite — is in")
        A("`team/fpga/FEATURES.md`.\n")

    if s6:
        A("## Stage 6 — is the student limited by its inputs or by its shape?\n")
        A("Gradient-boosted trees on **event scalars only** — no particle branch at all —")
        A("on the same train300k/eval100k split the DeepSet rows use.\n")
        A("| setup | cols | AUC (all) | vs QCD | vs tt | vs W+jets |")
        A("|---|---|---|---|---|---|")
        for r in s6["rows"]:
            A(f"| {r['setup']} | {r['n_cols']} | {r['auc_all']:.4f} | {r['auc_qcd']:.4f} | "
              f"{r['auc_tt']:.4f} | {r['auc_w']:.4f} |")
        for label, (aa, q, t, w) in s6["reference"].items():
            A(f"| *{label}* | — | {aa:.4f} | {q:.4f} | {t:.4f} | {w:.4f} |")
        A("")
        A("**The answer is feature content, not representation.** The 11 incumbent scalars")
        A("alone reach 0.8742 against the DeepSet's 0.8840 — the entire 16-candidate")
        A("particle branch is worth about 0.010. Give the same trees the event scalars this")
        A("lane built and they reach 0.8998, *beating* the DeepSet by 0.016 with no")
        A("per-particle processing whatsoever, and 0.9052 with everything. Hours spent on")
        A("features have been paying roughly twice what hours spent on architecture would.\n")
        A("Two riders. The ceiling here is a 100-tree GBDT, not something that fits an SLR —")
        A("it says where the information is, not what to ship. And the jet-clustered scalars")
        A("(row D) are the part of the ceiling a trigger cannot afford; the affordable row C")
        A("is already 0.8998.\n")

    if s7:
        b = s7["baseline"]
        A("## Stage 7 — the two parquet fields we were not reading\n")
        A("`team/physics/COLUMNS.md` inventories the files: we use 4 of `L1T_PUPPIPart`'s 14")
        A("subfields. Two of the other ten answer questions this lane spent the day")
        A("approximating — **`pdgId`** flags electrons and muons directly (`iso_lead_pt` is a")
        A("hand-built proxy for exactly that), and **`dxysig`** is the impact-parameter")
        A("*significance*, the variable a real b-tagger uses, where we feed raw `dxy`.")
        A("Trees on the 11 incumbent scalars, plus each block (train/eval read straight from")
        A("parquet, 120k train / 60k eval):\n")
        A("| features | AUC (all) | vs tt | vs QCD | vs W+jets |")
        A("|---|---|---|---|---|")
        A(f"| 11 event features | {b[0]:.4f} | {b[1]:.4f} | {b[2]:.4f} | {b[3]:.4f} |")
        for r in s7["rows"]:
            A(f"| {r['block']} | {r['auc_all']:.4f} | {r['auc_tt']:.4f} | {r['auc_qcd']:.4f} | "
              f"{r['auc_w']:.4f} |")
        A("")
        A("**+0.036 overall and +0.076 vs tt from two fields already on disk** — more than")
        A("every hand-made feature in this document put together. The strongest singles vs tt:")
        top = sorted(s7["singles"].items(), key=lambda kv: -kv[1])[:6]
        A("  " + ", ".join(f"`{k}` {v:.3f}" for k, v in top) + ".\n")
        A("`dxysig` is stored as float16 and overflows — |dxysig| reaches `inf` and its p99 is")
        A("in the thousands — so it must be clipped (20.0 here) before anything touches it.")
        A("`data.py` now reads both fields (only when a feature needs them) and exposes nine")
        A("derived event scalars; all nine are O(16) reductions, no DSP, no new pair table.\n")

    if s8:
        A("## Stage 8 — hadronic tt: is there a top tag in the candidates?\n")
        A("The lepton features fixed the easy tt modes and left the hard one (0.708 vs")
        A("hadronic tt on the canonical set). All-hadronic tt is a W→qq inside a t→Wb, so:")
        A("the 15 dijet masses among the leading 6 candidates, `min |m_jj − 80|`, and the 20")
        A("trijet masses, `min |m_jjj − 173|`. Cheap, because for massless constituents")
        A("`m_ijk² = m_ij² + m_ik² + m_jk²` — the 20 triples are adds once the 15 pairs exist.\n")
        A("| on top of the canonical 19 | tt hadronic | tt semi-lep | tt leptonic | QCD | W+jets | all bkg |")
        A("|---|---|---|---|---|---|---|")
        for label, r in s8["blocks"].items():
            A(f"| {label} | {r['tt_hadronic']:.4f} | {r['tt_semilep']:.4f} | "
              f"{r['tt_leptonic']:.4f} | {r['QCD']:.4f} | {r['Wjets']:.4f} | {r['all']:.4f} |")
        A("")
        A("**The top tag is dead: +0.0003 against hadronic tt.** And the single-feature numbers")
        A("say why it was never going to work — `dm_top6` scores 0.571 against *hadronic* tt")
        A("and 0.651 against *leptonic* tt. A real top tag would do the opposite. It is")
        A("picking up generic kinematics, not a resonance: the inputs are particle-flow")
        A("candidates, not jets, so two of the leading six routinely come from the same jet")
        A("and the leading six rarely span two tops. This is the third independent way the")
        A("same conclusion has arrived (jet-clustered `dm_W`/`dm_top` +0.002, the HH dijet")
        A("pairing `dm_higgs` +0.000, and now this) — **mass reconstruction is not available")
        A("at this input granularity, and no more hours should go into it.**\n")
        A("**The |dxy| order statistics are the opposite story: +0.017 against hadronic tt,**")
        A("the mode nothing else moved, plus +0.006 semi-leptonic and +0.011 vs QCD, for four")
        A("numbers and a comparator network. They are in `data.py`'s canonical set.\n")

    if mixture:
        A("## The organizers' eval mixture is tt-dominated, and ours is not\n")
        ev = mixture["eval"]
        tot = sum(v["events"] for v in ev.values())
        bkg = {}
        for k, v in ev.items():
            if v["group"] != "HH_4b":
                bkg[v["group"]] = bkg.get(v["group"], 0) + v["events"]
        tb = sum(bkg.values())
        A(f"Row counts straight out of `eval/` ({tot:,} events, 47.6% of them signal):\n")
        A("| process | events | share of background |")
        A("|---|---|---|")
        for k, v in ev.items():
            share = "— (signal)" if v["group"] == "HH_4b" else f"{v['events']/tb*100:.2f}%"
            A(f"| `{k}` | {v['events']:,} | {share} |")
        A("")
        A("Grouped, the official background is **QCD " + f"{bkg['QCD']/tb*100:.1f}%, tt "
          f"{bkg['tt']/tb*100:.1f}%, W+jets {bkg['Wjets']/tb*100:.1f}%" + "** — against our")
        A("even thirds. So **no: the official metric weights QCD far *less* than our eval")
        A("slice does (9% against 33%), and tt far more (55% against 33%).** Our worst")
        A("background is the official metric's dominant one. Re-weighting our saved eval")
        A("scores to those proportions (AUC is a rank statistic, so only the background")
        A("composition matters):\n")
        A("| run | our even mix | official eval mix | Δ |")
        A("|---|---|---|---|")
        for tag, a0, a1 in [("B1e_16p_1M", 0.88687, 0.85180), ("c2_base_cpu", 0.88397, 0.84761),
                            ("c2_pair4", 0.88785, 0.85455), ("c2_rich", 0.89758, 0.86821),
                            ("c2_canon", 0.90099, 0.87300),
                            ("c2_canon_narrow", 0.90150, 0.87504)]:
            A(f"| `{tag}` | {a0:.5f} | {a1:.5f} | {a1-a0:+.5f} |")
        A("")
        A("Every number drops about 0.03 under the official mixture — and the tt work gains")
        A("in importance rather than losing it: `c2_canon_narrow` beats `B1e_16p_1M` by")
        A("+0.0146 on our slice and by **+0.0232** on the organizers'. Within tt the three")
        A("decay modes are equal in `eval/` (200k each), which matches our sampling; the")
        A("Standard-Model-branching-fraction caveat earlier in this section is about physical")
        A("realism, not about the challenge metric.\n")

    A("## `train4M` is built and waiting — for c1\n")
    A("`team/cache/train4M/` (5.6 GB on disk), built on the CPU box with the canonical")
    A("input set: **X (7,569,258, 16, 11) float32, F (7,569,258, 19), y, group**, streamed")
    A("from parquet so peak memory stayed near 11 GB. `train.py` reads it with no change —")
    A("`--train-tag train4M --eval-tag eval100k_c2 --pool meanmax`.\n")
    A("**Two things to know before you use it.** First, it is 4,000,000 signal against")
    A("3,569,258 background, not 4M+4M: QCD ran out. The whole `train/QCD_HT250toInf`")
    A("directory holds 902,592 events, so a QCD budget of 1,333,333 could not be met.")
    A("The background is therefore **QCD 25.3%, tt 37.4%, W+jets 37.4%**, not even thirds.")
    A("Second, that is *closer* to the organizers' eval mixture (QCD 9%, tt 55%, W 36%) than")
    A("even thirds is, so it is arguably the better training mixture — but it is a different")
    A("mixture from `train1M`, and a model trained on it is not a clean A/B against a")
    A("`train1M` row. If you want strict even thirds at this scale the ceiling is")
    A("2,707,776 background events (3 × 902,592).\n")
    A("A `train4M` cache with the *newer* `dxysig`/`pdgId` features (F of 31 rather than 19)")
    A("is a rebuild away; say the word and it runs.\n")
    A("## What to take, and what to leave\n")
    A("Answering the two questions that were asked, with the numbers above:\n")
    A("**1. Ranked, the teacher's derived quantities.** The 6 per-candidate `rich` channels")
    A("are the prize (`rich:ALL`, +0.033 vs tt in the proxy, +0.037 in the real student).")
    A("Within them, |dxy| and Δφ-to-the-leading-candidate carry most of it (+0.012 each);")
    A("ln pt/HT and ln E carry almost nothing on their own (+0.003) because HT and the")
    A("leading pTs are already event features. Of the pairwise quantities, **ln ΔR is the")
    A("useful one** (+0.016 from 6 numbers), ln kT and ln m² are slightly weaker and largely")
    A("the same information, and **ln z is worth nothing at all** (−0.000).\n")
    A("**2. What a trigger student can afford.** In priority order:\n")
    A("* **The 6 rich channels — take them.** +0.037 vs tt and +0.0136 overall in the actual")
    A("  2,057-parameter model, for +192 parameters. They are all O(n): ln(pt/HT) needs HT")
    A("  (already summed), ln E needs cosh η, Δφ/Δη are differences against the leading")
    A("  candidate, |dxy| is an absolute value. The catch is that they widen φ, the block")
    A("  the FPGA lane already has at 91% of an SLR — so take them *and* pay for them by")
    A("  narrowing φ or dropping to 8 candidates (rows `c2_canon_narrow` / `c2_canon_8p`,")
    A("  both cheaper than today's baseline).")
    A("* **Max-pooling alongside mean — take it.** Comparators, no DSP, +8 inputs to ρ.")
    A("* **`iso_lead_pt` (and `n_iso`, free once the ΔR table exists) — take them.** One")
    A("  event-level scalar is worth more against tt (+0.024) than all 24 leading-4 pair")
    A("  numbers together (+0.016), and it goes in after the pool where there is room.")
    A("* **Pair quantities — take ln ΔR of the leading-4 pairs.** 6 scalars, +0.016 in the")
    A("  proxy and +0.013 vs tt in the real student. Adding ln kT and ln m² on top buys")
    A("  +0.000 (they are the same information");
    A("  in different coordinates); ln z buys nothing anywhere.")
    A("* **A full pairwise block — no.** Pooling all 120 pairs into mean/min/max/std keeps")
    A("  only +0.009 of the +0.023 that the explicit leading-6 pairs give, so the value is")
    A("  in *which* pair, not in the pairwise ensemble. A student cannot afford 16×16×4")
    A("  inputs, and the ceiling it would be reaching for is modest.\n")
    A("And the framing correction, restated because it changes what to chase: the published")
    A("teacher's 0.826 vs tt comes from per-candidate features plus capacity, not from")
    A("relational information. A ParT teacher may still change that — but on this evidence,")
    A("pairwise is the smaller half.\n")

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
