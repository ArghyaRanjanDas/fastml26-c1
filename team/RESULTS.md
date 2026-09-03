# Challenge 1 — HH→4b vs. background, results log

Metric: **binary ROC AUC** of the network score (signal = `HH_4b`, background = QCD + tt
+ W+jets pooled), on a held-out slice built from `eval/` — never from a file used for
training. Train = `train300k` (300k signal + 300k background, background split evenly
over the three groups). Eval = 100k + 100k, same mixture. Inputs: the 16 leading-pT
`L1T_PUPPIPart` candidates × 5 features (log-pT, η, dxy, cos φ, sin φ).

FPGA columns are filled in by `team/fpga/synth.py`; see `team/fpga/RESULTS-fpga.md`.

| model | params | AUC (eval) | train events | quant | LUT | FF | DSP | BRAM | latency | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSet φ64-32-16 ρ256-128-32 (round-1 baseline) | 44,401 | 0.88261 | 600k | float32 | — | — | — | — | 0.21 µs/evt GPU | dropout 0.1/0.2, 20 ep. Reference point. |
| ─ **task 1: event-level features** ─ | | | | | | | | | | |
| + 11 event features, [0,1] squash | 47,217 | 0.87877 | 600k | float32 | — | — | — | — | — | **−0.0034 vs baseline.** Higher *train* loss too → underfitting, not overfitting. |
| + 11 event features, standardized | 47,217 | 0.87806 | 600k | float32 | — | — | — | — | — | Standardizing the features did not recover it. |
| + event features, ×0.2 at concat | 47,217 | 0.87849 | 600k | float32 | — | — | — | — | — | Rescale to fix the drive imbalance. |
| + event features, 60 epochs | 47,217 | 0.88014 | 600k | float32 | — | — | — | — | — | Longer training closes part of the gap. |
| + event features + pool BatchNorm | 47,217 | 0.88027 | 600k | float32 | — | — | — | — | — | |
| + event features + pool BN + ×0.2 | 47,217 | 0.88036 | 600k | float32 | — | — | — | — | — | Best *with* features. Still below the control. |
| **control: no event features + pool BatchNorm** | 44,401 | **0.88436** | 600k | float32 | — | — | — | — | — | **Best of task 1.** The BatchNorm is the win, not the features. |
| probe: 11 event features alone (MLP, no particles) | 3,393 | 0.87236 | 600k | float32 | — | — | — | — | — | Diagnostic only — shows the features are informative but redundant. |

## Task 1 conclusion: the event features give **no gain** (−0.004)

Requested features (HT, leading-4 pT, n_cand, sum/max/mean |dxy|, m2, m4) were built and
verified — `ht`/`lead_pt` recomputed from the cached particle tensor match the cached
features to 0.00000, and `m2` matches the analytic back-to-back value. They are genuinely
informative: **on their own** they reach AUC 0.872, i.e. almost the whole baseline. But
concatenated after pooling they cost ~0.004 AUC at matched architecture (0.88036 vs
0.88436).

Two findings explain it:

1. **Redundancy.** Mean-pooling a learned per-particle φ already extracts essentially the
   same information, so the features add nothing the network did not have.
2. **A drive imbalance that actively hurts.** The mean-pooled vector sits at |h| ≈ 0.11
   while standardized event features sit at |f| ≈ 0.67, so they drove the first ρ layer
   **5.5× harder** (0.0714 vs 0.0131) and the φ branch under-trained. Measured, not guessed.

The useful by-product: **a BatchNorm on the pooled vector is worth +0.002 AUC and is free
in firmware** (at inference it is a fixed per-channel affine that folds exactly into the
next Linear — `export.py` does the fold, so nothing extra is synthesized).

`n_cand` is a dead input in this dataset (identically 16 everywhere, std exactly 0); kept
for spec compliance and because it revives if `n_particles` drops. See `PIPELINE.md`.

---

# Task 2 — size sweep (AUC vs. params, and vs. the thing that actually costs LUTs)

All rows: dropout 0, pooled BatchNorm on, 25 epochs, `train300k`. Families A/B/C/D
per `sweep_size.sh` and `sweep_narrow.sh`.

**φ MACs/event = n_particles × Σ(in×out over φ layers).** φ runs once per particle, so
this — not the parameter count — is what sets DSP/LUT. The `~LUT`/`~DSP` columns scale
the one measured Vitis point (φ64-32-16 × 16p → 524,961 LUT / 4,836 DSP at
`ap_fixed<16,6>`, reuse 1) linearly in φ MACs. They are an extrapolation for ranking
candidates, not a synthesis result; real numbers go in `fpga/RESULTS-fpga.md`.

| tag | φ | ρ | particles | evt feats | params | φ MACs/evt | **AUC (eval)** | AUC vs tt | ~LUT | ~DSP | fits SLR? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `A1_40k` | 64-32-16 | 256-128 | 16 | no | 40,401 | 46,080 | **0.88755** | 0.75917 | 524,961 | 4,836 | ❌ |
| `A2_9k` | 64-32-16 | 96-40 | 16 | no | 8,577 | 46,080 | **0.88476** | 0.75171 | 524,961 | 4,836 | ❌ |
| `A3_3k` | 64-32-16 | 16-8 | 16 | no | 3,441 | 46,080 | **0.88319** | 0.74825 | 524,961 | 4,836 | ❌ |
| `A2e_9k` | 64-32-16 | 96-40 | 16 | yes | 9,633 | 46,080 | **0.88343** | 0.75465 | 524,961 | 4,836 | ❌ |
| `A3e_3k` | 64-32-16 | 16-8 | 16 | yes | 3,617 | 46,080 | **0.88618** | 0.75665 | 524,961 | 4,836 | ❌ |
| `B1_16p` | 32-16-8 | 32-16 | 16 | no | 1,705 | 12,800 | **0.88222** | 0.74642 | 145,822 | 1,343 | ✅ |
| `B1e_16p` | 32-16-8 | 32-16 | 16 | yes | 2,057 | 12,800 | **0.88378** | 0.74932 | 145,822 | 1,343 | ✅ |
| `B2_16p` | 32-16-8 | 64-32 | 16 | no | 3,561 | 12,800 | **0.88184** | 0.74468 | 145,822 | 1,343 | ✅ |
| `C1_8p` | 32-16-8 | 32-16 | 8 | no | 1,705 | 6,400 | **0.85568** | 0.71496 | 72,911 | 671 | ✅ |
| `C1e_8p` | 32-16-8 | 32-16 | 8 | yes | 2,057 | 6,400 | **0.88011** | 0.74712 | 72,911 | 671 | ✅ |
| `C3e_8p` | 32-16-8 | 64-32 | 8 | yes | 4,265 | 6,400 | **0.87446** | 0.73793 | 72,911 | 671 | ✅ |
| `C2_8p` | 16-8 | 32-16 | 8 | no | 1,081 | 1,664 | **0.85197** | 0.70977 | 18,956 | 174 | ✅ |
| `C2e_8p` | 16-8 | 32-16 | 8 | yes | 1,433 | 1,664 | **0.87155** | 0.73303 | 18,956 | 174 | ✅ |
| `D1_8p` | 24-12-8 | 32-16 | 8 | no | 1,397 | 4,032 | **0.85581** | 0.71595 | 45,934 | 423 | ✅ |
| `D1e_8p` | 24-12-8 | 32-16 | 8 | yes | 1,749 | 4,032 | **0.87272** | 0.73288 | 45,934 | 423 | ✅ |
| `D2e_16p` | 24-12-8 | 32-16 | 16 | yes | 1,749 | 8,064 | **0.87334** | 0.73594 | 91,868 | 846 | ✅ |

## What the sweep says

**1. Parameters are the wrong axis.** `A1_40k` (40,401 params) and `A3_3k` (3,441) have
*identical* FPGA cost — same φ, same particles — and differ by only 0.004 AUC. Shrinking ρ
is nearly free in AUC and buys nothing in hardware. Shrinking φ is what matters.

**2. Narrowing φ is almost free in AUC.** φ 64-32-16 → 32-16-8 at 16 particles costs
**0.001 AUC** (0.88476 → 0.88378) and cuts φ MACs **3.6×**, taking the extrapolated cost
from 525k LUT / 4,836 DSP (over budget on both) to ~146k / ~1,343 — inside one SLR at
16-bit, before any quantization work.

**3. Halving particles is *not* free — unless you add the event features.** This is the
sweep's main result and it reverses the task-1 conclusion:

| | 16 particles | 8 particles | cost of halving |
|---|---|---|---|
| no event features | 0.88222 | 0.85568 | **−0.0265** |
| with event features | 0.88378 | 0.88011 | **−0.0037** |

At 16 particles the event features are worth +0.0016 (noise-level, consistent with task 1).
At 8 particles they are worth **+0.0244**. Once the particle branch can no longer see the
whole event, the event-level summaries stop being redundant and start carrying the
information the truncation threw away — and they are computed **once per event, from all
16 candidates**, so they cost nothing in the 8×-replicated φ block. Same story at φ16-8
(+0.0196) and φ24-12-8 (+0.0169).

So the task-1 finding stands as stated (no gain at full width) and is *superseded* in
exactly the regime the FPGA budget pushes us into.

**4. Widening ρ never paid.** `B2_16p` (ρ64-32) < `B1_16p` (ρ32-16); `C3e_8p` (ρ64-32) <
`C1e_8p` (ρ32-16). ρ32-16 is enough everywhere tested.

---

# Task 3 — export for FPGA (`team/export/`)

Per the contract in `team/fpga/README.md`. Three models exported; **`model_2041` is the
primary** — the best AUC among candidates that fit one SLR.

| file | run | φ | ρ | particles | params | AUC (eval) | why |
|---|---|---|---|---|---|---|---|
| `model_2041.{pt,json}` + `eval_sample.npz` | `B1e_16p_1M` | 32-16-8 | 32-16 | 16 | 2,041 | **0.88687** | **primary** — best AUC that fits (~146k LUT / ~1,343 DSP) |
| `model_2041_8p.{pt,json}` + `eval_sample_8p.npz` | `C1e_8p_1M` | 32-16-8 | 32-16 | 8 | 2,041 | **0.88113** | the requested 8-particle variant — half the φ bill (~73k LUT / ~671 DSP) |
| `model_3585.{pt,json}` + `eval_sample_3617.npz` | `A3e_3k` | 64-32-16 | 16-8 | 16 | 3,585 | 0.88618 | best AUC ≤10k params, but φ64-32-16 is the width Vitis already rejected |

Each `.json` carries the contract keys (`phi`, `rho`, `n_features`, `n_particles`,
`n_event_features`) plus activations, pooling, the feature list and every normalization
constant. Each `eval_sample*.npz` holds 5000 preprocessed eval events: `X` (also as
`particles`), `F` (also as `event`), `y`, and `scores`.

**Two things done so the firmware is not silently wrong:**

*The BatchNorm is folded away.* The trained models carry a BatchNorm on the pooled vector.
`synth.py` maps `*.weight` keys positionally onto Keras Conv1D/Dense, so a BatchNorm in the
state_dict would both break that mapping and ask the firmware to synthesize something it
does not need. At inference BatchNorm is a fixed per-channel affine, so `export.py` folds
it exactly into the first ρ Linear (and folds the `event_scale` multiplier into the same
weights). Verified: max|folded − original| ≈ 1.7e-06. The exported state_dict is
BatchNorm-free — hence 2,041 params rather than the trained 2,057.

*The mapping is verified, not assumed.* `verify_export.py` reimplements `synth.py`'s exact
torch→Keras mapping (`sd[k].numpy().T`, `W[None,:,:]` for Conv1D) in pure numpy and runs the
5000 exported events through it. All three exports reproduce the stored scores to <2e-06
and the identical AUC. This catches a transpose or layer-order slip before synthesis, and
needs neither hls4ml nor TensorFlow:

```bash
python verify_export.py --json export/model_2041.json \
       --weights export/model_2041.pt --sample export/eval_sample.npz   # -> PASS
```

---

# Round 3 — distillation, c2's rich inputs, and the deployable student

## Distillation (task 1)

Teacher trained here: `teacher_1M`, φ128-64-32 / ρ256-128-64, mean+max pool, 71,905 params,
40 epochs on 2M events — **eval AUC 0.89950** (tt 0.78927). Students are the deployable
φ32-16-8 / ρ32-16 shape. Binary KD: soft target `sigmoid(z_t/T)`, student scored at `z_s/T`,
KD term scaled by `T²`, mixed with hard BCE at weight `alpha`. Teacher logits are cached once,
so distillation costs the same per epoch as ordinary training.

| student | teacher | T | alpha | params | AUC (eval) | AUC vs tt |
|---|---|---|---|---|---|---|
| from scratch (`B1e_16p_1M`) | — | — | — | 2,057 | 0.88687 | 0.75869 |
| `kd_T2_a07` | teacher_1M (0.8995) | 2 | 0.7 | 2,057 | **0.88997** | 0.76740 |
| `kd_T3_a07` | teacher_1M (0.8995) | 3 | 0.7 | 2,057 | 0.88876 | 0.76398 |

**Distillation is worth +0.0031** at identical cost (0.88687 → 0.88997), and +0.0087 against tt.
T=2 beat T=3. The sweep was cut short at two points — the team soft targets landed and
superseded it, and the box became CPU-contended.

## c2's rich inputs — the big win

Taking exactly what c2's study priced as worth its cost (`make_student_cache.py`):
per-candidate X 5 → **11 channels** (+ ln(pt/HT), ln E, cos/sin Δφ_lead, Δη_lead, |dxy|),
**mean+max pooling**, and F 11 → **19 event features** (+ `iso_lead_pt`, `n_iso`, and ln ΔR of
the 6 leading-4 pairs). Deliberately skipped: ln kT / ln m² / ln z pair quantities and any full
pairwise block — c2 measured those as redundant or worthless.

| run | inputs | teacher | T | alpha | params | φ MACs | **AUC (eval)** | AUC vs tt |
|---|---|---|---|---|---|---|---|---|
| `B1e_16p_1M` | 5ch / 11 evt | — | — | — | 2,057 | 12,800 | 0.88687 | 0.75869 |
| `rich_16p` | **11ch / 19 evt** | — | — | — | 2,777 | 15,872 | 0.90638 | 0.80822 |
| `rkd_T3_a07` | 11ch / 19 evt | ds_big_s0 (0.91515) | 3 | 0.7 | 2,777 | 15,872 | 0.90689 | 0.80752 |
| `rkd_T2_a09` | 11ch / 19 evt | ds_big_s0 | 2 | 0.9 | 2,777 | 15,872 | 0.90779 | 0.80913 |
| `rkd_T2_a07` | 11ch / 19 evt | ds_big_s0 | 2 | 0.7 | 2,777 | 15,872 | 0.90787 | 0.80960 |
| **`rkd_T2_a05`** | 11ch / 19 evt | ds_big_s0 | 2 | 0.5 | 2,777 | 15,872 | **0.90901** | **0.81188** |

**+0.0221 overall and +0.053 against tt, for +720 parameters and +24% φ MACs.** c2's predicted
+0.0136 overall / +0.037 vs tt was, if anything, conservative. Distillation on top of the rich
inputs adds a further +0.0026, and lower alpha is better here (0.5 > 0.7 > 0.9) — with strong
inputs the hard labels still carry information the teacher's soft targets do not.

Exported as `export/model_2777_rich.{pt,json}` + `eval_sample_rich.npz`, with a full
`input_spec` in the json giving every derived channel's formula and frozen constant.

---

# Confirmation run at 1M + 1M events

Same config as `B1e_16p`, trained on `train1M` (1M signal + 1M background, same even
group mixture), 25 epochs. Eval slice unchanged.

| run | train events | params | AUC (eval) | vs QCD | vs tt | vs W+jets | Δ vs 300k |
|---|---|---|---|---|---|---|---|
| `B1e_16p` (16 particles) | 600k | 2,057 | 0.88378 | 0.93109 | 0.74932 | 0.97059 | — |
| **`B1e_16p_1M`** (16 particles) | **2M** | 2,057 | **0.88687** | 0.93027 | 0.75869 | 0.97163 | **+0.0031** |
| `C1e_8p` (8 particles) | 600k | 2,057 | 0.88011 | — | 0.74712 | — | — |
| **`C1e_8p_1M`** (8 particles) | **2M** | 2,057 | **0.88113** | — | 0.75017 | — | **+0.0010** |

More data is worth about as much as the entire ρ sweep, at zero hardware cost. The 2,041-param
model trained on 2M events (**0.88687**) now essentially matches the 40,401-param `A1_40k`
trained on 600k (0.88755) — while `A1_40k` needs ~525k LUT / 4,836 DSP and this one needs
~146k / ~1,343. Most of the gain is against tt (0.74932 → 0.75869), the weak background.

**The exports in `team/export/` are the 1M models** (`B1e_16p_1M` and `C1e_8p_1M`); the
table in Task 3 above lists the architectures, which are unchanged.

## Next (per FPGA feedback #2: LUT is the binding constraint)

Reuse factor alone did not save the big model (DSP 4,836 → 1,624, but LUT stayed 538k and
latency hit ~220 cycles). The exported φ32-16-8 models cut φ MACs 3.6–7.2× at ≈0.001–0.004
AUC, which should clear LUT on its own; **QAT at 8-bit/6-bit in `~/hlsenv` is the next
lever** and is where the remaining margin comes from. Not started — flagged as follow-up.

## Reproduce

```bash
cd ~/fastml26-hackathon/team
./sweep_size.sh && ./sweep_narrow.sh        # the table above
python export.py --run B1e_16p              # the primary export
python verify_export.py --json export/model_2041.json \
       --weights export/model_2041.pt --sample export/eval_sample.npz
```

---

# Teacher lane (Purdue AF A100, agent `hh4b`) — soft targets for distillation

Unconstrained teachers trained on `train1M` (1M + 1M, 10 % held out for model selection),
evaluated on the same `eval100k` slice as every row above. Inputs are *exactly* the
student's cache tensors; the teacher derives extra per-candidate and pairwise quantities
from them on the fly (`team/teacher/common.py`). Soft targets = float32 logits in cache
row order: `team/teacher/soft_targets_{train1M,train300k,eval100k}.npy`
(`soft_targets_meta.json` says which run they come from). Training: AdamW, warm-up + cosine,
label smoothing 0.05, bf16, EMA of weights; the run is selected on validation AUC.

| run | model | params | train events | epochs | **AUC (eval)** | vs QCD | vs tt | vs W+jets | eff@99 % rej | train-slice AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `B1e_16p_1M` (student, for reference) | DeepSet φ32-16-8 ρ32-16 + evt | 2,057 | 2M | 25 | 0.88687 | 0.93027 | 0.75869 | 0.97163 | — | — | the distillation target shape |
| **`ds_big_s0`** | BigDeepSet φ128-64-32, mean+max, ρ256-128-64, rich feats | 72,717 | 2M | 40 (best 23) | **0.91515** | 0.94363 | **0.82612** | 0.97569 | 0.272 | 0.92046 | **first soft targets (pushed)**. +0.028 overall, +0.067 vs tt over the student. Train/val gap 0.005 → in-sample logits are fine as targets. |

---

<!-- c2:tt-study:start -->
# Task 4 (c2): the tt background

`B1e_16p_1M` is 0.930 vs QCD and 0.972 vs W+jets but **0.759 vs tt**. Everything below
asks one question: what else is in the 16 PUPPI candidates that separates HH→4b from tt?

All of it runs on CPU off the existing `team/cache` tensors — the cached `X` is an
invertible transform of (pt, η, φ, dxy), so `physics/features.py:decode()` recovers
physical units without re-reading a single parquet file. 39 candidate features,
`physics/rank.py` for the ranking, eval slice = `eval100k` throughout.

## The mixture, and which tt actually hurts

`data.py` gives each of the three tt directories 1/9 of the background budget, so the
sampled tt is **1/3 hadronic, 1/3 semi-leptonic, 1/3 fully leptonic**. Nature gives
roughly 45.7 / 43.8 / 10.5. Splitting the frozen baseline by mode (30,000 eval events each):

| background | `B1e_16p_1M` AUC |
|---|---|
| tt_hadronic | 0.7234 |
| tt_semilep | 0.7593 |
| tt_leptonic | 0.7954 |
| **tt [even (as sampled)]** | **0.7594** |
| **tt [SM branching fractions]** | **0.7470** |
| QCD | 0.9300 |
| Wjets | 0.9752 |

The baseline is worst against **all-hadronic tt**, the mode with no lepton and no
neutrino — exactly the tt that looks like four b-jets. Because the sampled mixture
over-represents the *easy* leptonic mode by 3×, the headline 0.759 is optimistic:
re-weighted to Standard Model branching fractions it drops (row above). Worth saying
out loud on Friday, and worth remembering when reading every gain below.

### Which feature works on which tt

Single-feature AUC, signal vs that process alone. Read the first three columns
together: `iso_lead_pt` is a *lepton* tag (0.59 → 0.82 across the modes), while the
dxy features are flat across modes and are the only handle that works on the
hadronic mode. Anything that beats hadronic tt has to come from b-content, jet
counting or mass structure, not from the lepton.

| feature | cost | tt_hadronic | tt_semilep | tt_leptonic | QCD | Wjets |
|---|---|---|---|---|---|---|
| `iso_lead_pt` | pairwise | 0.5888 | 0.7333 | 0.8221 | 0.5442 | 0.5121 |
| `n_dxy_p05` | event | 0.6831 | 0.6721 | 0.6593 | 0.8914 | 0.9250 |
| `ptw_dxy` | event | 0.6697 | 0.6706 | 0.6701 | 0.8870 | 0.9435 |
| `ht4_frac` | event | 0.5528 | 0.6726 | 0.7734 | 0.5739 | 0.5710 |
| `mt_lep_mpt` | pairwise | 0.5474 | 0.6745 | 0.7570 | 0.5033 | 0.5429 |
| `dxy_lead4` | event | 0.6500 | 0.6406 | 0.6292 | 0.8109 | 0.8246 |
| `n_jets15` | jets | 0.6632 | 0.6476 | 0.6135 | 0.6295 | 0.8542 |
| `n_bjets` | jets | 0.6442 | 0.6430 | 0.6405 | 0.8318 | 0.8763 |
| `n_iso` | pairwise | 0.5954 | 0.6352 | 0.6738 | 0.5627 | 0.5794 |
| `n_jets30` | jets | 0.6339 | 0.6290 | 0.6116 | 0.5524 | 0.8643 |
| `m6` | event | 0.5832 | 0.6224 | 0.6578 | 0.5453 | 0.6144 |
| `n_dxy_p20` | event | 0.6310 | 0.6233 | 0.6148 | 0.7783 | 0.8159 |

## Stage 1 — each feature alone (HH→4b vs tt, eval slice)

Sign is a free parameter, so the AUC is reported oriented (`dir` = whether signal
sits high or low). The 11 incumbent event features are included for scale.

| feature | cost | dir | AUC vs tt | vs QCD | vs W+jets | vs all bkg |
|---|---|---|---|---|---|---|
| `iso_lead_pt` | pairwise | - | 0.7146 | 0.5441 | 0.5858 | 0.5282 |
| `n_dxy_p05` | event | + | 0.6685 | 0.8911 | 0.9248 | 0.8281 |
| `ptw_dxy` | event | + | 0.6673 | 0.8869 | 0.9397 | 0.8313 |
| `ht4_frac` | event | - | 0.6651 | 0.5720 | 0.6618 | 0.5251 |
| `[base] lead_pt1` | event | - | 0.6613 | 0.5276 | 0.7762 | 0.4709 |
| `mt_lep_mpt` | pairwise | - | 0.6580 | 0.5037 | 0.5355 | 0.5396 |
| `[base] mean_abs_dxy` | event | + | 0.6468 | 0.8819 | 0.9402 | 0.8230 |
| `[base] sum_abs_dxy` | event | + | 0.6468 | 0.8819 | 0.9402 | 0.8230 |
| `dxy_lead4` | event | + | 0.6404 | 0.8138 | 0.8318 | 0.7620 |
| `n_jets15` | jets | - | 0.6400 | 0.6304 | 0.8535 | 0.3854 |
| `n_bjets` | jets | + | 0.6393 | 0.8310 | 0.8743 | 0.7815 |
| `n_iso` | pairwise | - | 0.6360 | 0.5630 | 0.6093 | 0.4879 |
| `[base] lead_pt2` | event | - | 0.6315 | 0.5084 | 0.8660 | 0.4247 |
| `[base] m4` | event | - | 0.6291 | 0.5421 | 0.6317 | 0.5132 |
| `[base] ht` | event | - | 0.6269 | 0.5057 | 0.8957 | 0.4085 |
| `n_jets30` | jets | - | 0.6243 | 0.5516 | 0.8755 | 0.3991 |
| `[base] m2` | event | - | 0.6239 | 0.5090 | 0.5326 | 0.5275 |
| `[base] max_abs_dxy` | event | + | 0.6237 | 0.8680 | 0.9314 | 0.8077 |
| `m6` | event | - | 0.6217 | 0.5454 | 0.6392 | 0.5093 |
| `n_dxy_p20` | event | + | 0.6196 | 0.7762 | 0.8113 | 0.7357 |

<sub>Full table of all 50 rows: `physics/stage1_alone.json`.</sub>

## Stage 2 — what each feature *adds* to the 11 baseline features

A single feature scoring 0.67 alone is worthless if HT already carries it. So: fit
gradient-boosted trees on B1e_16p's 11 event features, then on the 11 + one candidate,
and difference. (300,000 train events, 150 trees; the GBDT is a
stand-in for ρ() at 20 s/fit instead of 20 min.)

**Baseline (11 event features, no particle branch): AUC vs tt 0.7377, overall 0.8732.**

`cost` is what the feature needs in firmware: **event** = O(16) reductions,
essentially free; **pairwise** = the 16x16 ΔR table (~512 multiplies, ~4% of φ());
**jets** = 6 sequential cone-clustering passes, expensive in latency.

| + feature | cost | AUC vs tt | Δ vs tt | AUC overall | Δ overall |
|---|---|---|---|---|---|
| `iso_lead_pt` | pairwise | 0.7619 | +0.0242 | 0.8808 | +0.0075 |
| `mt_lep_mpt` | pairwise | 0.7529 | +0.0152 | 0.8780 | +0.0048 |
| `n_iso` | pairwise | 0.7494 | +0.0117 | 0.8767 | +0.0034 |
| `ht_jets4` | jets | 0.7474 | +0.0097 | 0.8765 | +0.0032 |
| `n_jets15` | jets | 0.7468 | +0.0091 | 0.8757 | +0.0025 |
| `jet1_m` | jets | 0.7467 | +0.0090 | 0.8767 | +0.0035 |
| `m_jj_maxpt` | jets | 0.7463 | +0.0086 | 0.8765 | +0.0033 |
| `n_bjets` | jets | 0.7457 | +0.0080 | 0.8773 | +0.0041 |
| `m_4jet` | jets | 0.7446 | +0.0069 | 0.8753 | +0.0020 |
| `n_jets30` | jets | 0.7442 | +0.0065 | 0.8750 | +0.0018 |
| `n_dxy_p05` | event | 0.7429 | +0.0052 | 0.8768 | +0.0036 |
| `iso_min` | pairwise | 0.7427 | +0.0050 | 0.8748 | +0.0016 |
| `mpt_over_ht` | event | 0.7421 | +0.0044 | 0.8746 | +0.0014 |
| `dxy_lead4` | event | 0.7419 | +0.0042 | 0.8762 | +0.0030 |
| `mpt` | event | 0.7415 | +0.0038 | 0.8744 | +0.0012 |
| `mpt_sig` | event | 0.7414 | +0.0037 | 0.8744 | +0.0011 |
| `ptw_dxy` | event | 0.7414 | +0.0037 | 0.8746 | +0.0014 |
| `dR_j12` | jets | 0.7407 | +0.0030 | 0.8747 | +0.0015 |
| `centrality` | event | 0.7405 | +0.0028 | 0.8748 | +0.0016 |
| `dm_Wtop` | jets | 0.7402 | +0.0025 | 0.8742 | +0.0010 |
| `jet_m_max` | jets | 0.7401 | +0.0024 | 0.8742 | +0.0010 |
| `dm_top` | jets | 0.7400 | +0.0024 | 0.8742 | +0.0010 |
| `ht4_frac` | event | 0.7399 | +0.0022 | 0.8743 | +0.0010 |
| `sphericity_T` | event | 0.7398 | +0.0021 | 0.8751 | +0.0019 |
| `jet_ptdxy_max` | jets | 0.7396 | +0.0019 | 0.8742 | +0.0010 |
| `max_pt_dxy` | event | 0.7393 | +0.0017 | 0.8740 | +0.0007 |
| `m_bb1` | jets | 0.7391 | +0.0014 | 0.8737 | +0.0005 |
| `eta_spread` | event | 0.7388 | +0.0012 | 0.8738 | +0.0006 |
| `n_dxy_p20` | event | 0.7388 | +0.0011 | 0.8742 | +0.0010 |
| `dm_W` | jets | 0.7387 | +0.0010 | 0.8737 | +0.0005 |
| `dm_pair` | jets | 0.7385 | +0.0008 | 0.8734 | +0.0002 |
| `m_bb2` | jets | 0.7384 | +0.0008 | 0.8734 | +0.0002 |
| `min_dphi_mpt` | event | 0.7383 | +0.0006 | 0.8733 | +0.0001 |
| `deta_j12` | jets | 0.7383 | +0.0006 | 0.8737 | +0.0004 |
| `m16` | event | 0.7381 | +0.0004 | 0.8735 | +0.0003 |
| `m8` | event | 0.7380 | +0.0003 | 0.8734 | +0.0002 |
| `dm_higgs` | jets | 0.7379 | +0.0002 | 0.8733 | +0.0001 |
| `m6` | event | 0.7378 | +0.0001 | 0.8732 | -0.0001 |
| `pt_ratio_41` | event | 0.7376 | -0.0001 | 0.8731 | -0.0001 |

## Stage 3 — greedy forward selection

The top of stage 2 is three flavours of the same handle (an isolated hard candidate),
so ranking alone over-counts it. Forward selection, each step keeping the feature that
adds most against tt:

<sub>Stopped after step 3: the three features it had picked are the three that
go into `data.py`, and the CPU was better spent on the student runs below.</sub>

| step | added | cost | AUC vs tt | Δ | AUC overall | Δ |
|---|---|---|---|---|---|---|
| 1 | `iso_lead_pt` | pairwise | 0.7619 | +0.0242 | 0.8808 | +0.0075 |
| 2 | `n_iso` | pairwise | 0.7740 | +0.0120 | 0.8848 | +0.0040 |

## Stage 4 — the teacher's derived quantities, priced

`team/teacher/common.py` gives the teacher two blocks the student does not have:
6 derived quantities per candidate (`rich`), and ParT's 4 quantities per *pair*.
First, a correction worth having on the record: **the teacher that produced the
published soft targets (`ds_big_s0`, 0.9151 overall / 0.8261 vs tt) uses no pairwise
features at all.** It is `BigDeepSet(rich=True)` — per-candidate channels, mean+max
pooling, 72k parameters, 40 epochs on 2M events. `pair_features()` belongs to
`ParTLite`, and no ParT run has been published. So the 0.826 − 0.759 gap is not yet
evidence about relational information.

Same GBDT protocol as stage 2, one row per *family* of columns:

**Baseline (11 event features): AUC vs tt 0.7377, overall 0.8732.**

| + family | cols | cost for the student | AUC vs tt | Δ vs tt | AUC overall | Δ overall |
|---|---|---|---|---|---|---|
| `c2:iso + rich:ALL + pair4:ALL` | 55 | mixed | 0.7851 | +0.0474 | 0.8932 | +0.0200 |
| `c2:iso + rich:ALL` | 31 | mixed | 0.7824 | +0.0447 | 0.8921 | +0.0189 |
| `rich:ALL + pair4:ALL` | 54 | phi-width (+24% phi MACs, O(n)) | 0.7760 | +0.0383 | 0.8901 | +0.0169 |
| `rich:ALL` | 30 | phi-width (+24% phi MACs, O(n)) | 0.7711 | +0.0334 | 0.8883 | +0.0151 |
| `c2:iso_lead_pt` | 1 | mixed | 0.7619 | +0.0242 | 0.8808 | +0.0075 |
| `pair6:ALL` | 60 | 60 event scalars (15 pairs x 4) | 0.7602 | +0.0225 | 0.8803 | +0.0071 |
| `pair4:ALL` | 24 | 24 event scalars (6 pairs x 4) | 0.7538 | +0.0161 | 0.8783 | +0.0051 |
| `pair4:lndR` | 6 | 24 event scalars (6 pairs x 4) | 0.7537 | +0.0160 | 0.8783 | +0.0051 |
| `pair4:lnkt` | 6 | 24 event scalars (6 pairs x 4) | 0.7519 | +0.0143 | 0.8776 | +0.0044 |
| `pair4:lnm2` | 6 | 24 event scalars (6 pairs x 4) | 0.7512 | +0.0135 | 0.8772 | +0.0040 |
| `rich:abs_dxy` | 5 | phi-width (+24% phi MACs, O(n)) | 0.7498 | +0.0121 | 0.8798 | +0.0066 |
| `rich:cos_dphi_lead` | 5 | phi-width (+24% phi MACs, O(n)) | 0.7495 | +0.0118 | 0.8771 | +0.0039 |
| `pairfull:pooled` | 16 | full 16x16 table (not trigger-affordable) | 0.7462 | +0.0085 | 0.8780 | +0.0048 |
| `rich:sin_dphi_lead` | 5 | phi-width (+24% phi MACs, O(n)) | 0.7446 | +0.0069 | 0.8751 | +0.0019 |
| `rich:deta_lead` | 5 | phi-width (+24% phi MACs, O(n)) | 0.7432 | +0.0055 | 0.8747 | +0.0015 |
| `rich:lnz` | 5 | phi-width (+24% phi MACs, O(n)) | 0.7403 | +0.0026 | 0.8747 | +0.0014 |
| `rich:lnE` | 5 | phi-width (+24% phi MACs, O(n)) | 0.7403 | +0.0026 | 0.8748 | +0.0015 |
| `pair4:lnz` | 6 | 24 event scalars (6 pairs x 4) | 0.7375 | -0.0002 | 0.8729 | -0.0003 |

## Stage 5 — the same questions asked of the real 2k student

The GBDT above has no particle branch, so it over-states anything φ() already
computes. These are the actual B1e_16p architecture (φ 32-16-8, ρ 32-16, pool BN,
event scale 0.2, 25 epochs, seed 0, `train300k`), CPU, one input change per row.

The FPGA lane has just priced the baseline at ap_fixed<22,10>: **319k LUT / 1,724 DSP,
91% of one SLR** on both. There is no headroom, so where a feature lands matters as much
as what it is worth. Multiply-accumulates per event, at φ 32-16-8 / ρ 32-16:

| block | MACs/event | note |
|---|---|---|
| φ, 5 input channels (baseline) | 12,800 | 16 × 800, replicated per candidate — the FPGA bill |
| φ, 11 input channels (teacher `rich`) | 15,872 | **+3,072 (+24%)**, straight onto the block that is already at 91% |
| ρ, 19 inputs (baseline) | 1,136 | once per event |
| ρ, +24 pair scalars | 1,904 | +768, i.e. +6% of φ — after the pool, so it never replicates |
| ρ, +8 from max-pooling | 1,392 | +256; the max itself is comparators, no DSP |
| 16×16 ΔR table (isolation) | ~512 mults | cos Δφ = c_i c_j + s_i s_j from inputs already there |

| run | change | φ MACs | params | AUC (eval) | vs tt | vs QCD | vs W+jets | eff@99% |
|---|---|---|---|---|---|---|---|---|
| `c2_base_cpu` | control: 11 event features, 5 channels/candidate | 12,800 | 2,057 | 0.88397 | 0.7514 | 0.9292 | 0.9712 | 0.1737 |
| `c2_meanmax` | + max-pool alongside mean | 12,800 | 2,329 | 0.88451 | 0.7510 | 0.9306 | 0.9720 | 0.1747 |
| `c2_pair4` | + 24 leading-4 pair scalars (ln ΔR / kT / z / m²) | 12,800 | 2,825 | 0.88785 | 0.7646 | 0.9280 | 0.9709 | 0.1793 |
| `c2_rich` | + 6 teacher per-candidate channels (φ sees 11) | 15,872 | 2,249 | 0.89758 | 0.7886 | 0.9328 | 0.9713 | 0.1988 |
| `c2_rich_mm` | + those channels AND max-pool | 15,872 | 2,521 | 0.89879 | 0.7913 | 0.9327 | 0.9724 | 0.2017 |
| `c2_canon` | **canonical set**: 11 channels + max-pool + 8 event features | 15,872 | 2,777 | 0.90099 | 0.7961 | 0.9339 | 0.9729 | 0.2163 |
| *(reference)* `ds_big_s0` teacher, 72k params, 2M events | — | — | 72,717 | 0.91515 | 0.8261 | 0.9436 | 0.9757 | 0.2717 |

Max-pooling on its own is worth +0.0005. The 24 pair scalars are worth +0.013 vs tt.
The 6 per-candidate channels are worth **+0.037 vs tt**, and the three together
(`c2_canon`) are **+0.045 vs tt and +0.017 overall over the control**, taking signal
efficiency at 99% background rejection from 0.174 to 0.216. That is 2,777 parameters
and 300k training events reaching 0.796 vs tt, against 0.826 for a 72k-parameter
teacher trained on 2M — most of the teacher's advantage was its inputs, not its size.

### The canonical student input set — tensor layout for c1

`data.py` now builds it behind two flags. Nothing changes by default: a cache built
without them is byte-identical to before.

```bash
python data.py --tag train1M_c2  --split train --n-signal 1000000 --n-background 1000000 \
       --rich-particles --extra-features
python data.py --tag eval100k_c2 --split eval  --n-signal  100000 --n-background  100000 \
       --rich-particles --extra-features
python train.py --model deepset_plus --phi 32,16,8 --rho 32,16 --pool meanmax \
       --pool-norm --event-scale 0.2 --epochs 25 \
       --train-tag train1M_c2 --eval-tag eval100k_c2 --tag <name>
```
`train.py` needs no change: it takes the per-candidate width from `Xtr.shape[2]` and
the event width from `Ftr.shape[1]`. `--pool meanmax` is the pooling you already added.

**X — `(N, 16, 11)` float32.** Channels 0-4 are exactly what they were; 5-10 are new.
Order matters to anything reading an export:

| ch | name | value |
|---|---|---|
| 0 | `log_pt` | log1p(pt) / 8 |
| 1 | `eta` | η / 4 |
| 2 | `dxy` | clip(dxy, ±2) / 2 |
| 3 | `cos_phi` | cos φ |
| 4 | `sin_phi` | sin φ |
| 5 | `lnz` | ln(pt / HT) / 4 |
| 6 | `lnE` | log1p(pt cosh η) / 8 |
| 7 | `cos_dphi_lead` | cos(φ − φ₁) |
| 8 | `sin_dphi_lead` | sin(φ − φ₁) |
| 9 | `deta_lead` | (η − η₁) / 2 |
| 10 | `abs_dxy` | |dxy| / 2 |

Channels 5-10 are the teacher's, in the teacher's order, and match
`teacher/common.py:particle_features(rich=True)` to 2e-7 — so teacher and student
consume the identical tensor and the published soft targets stay valid.

**F — `(N, 19)` float32.** The 11 incumbent event features unchanged, then:

| idx | name | value |
|---|---|---|
| 11 | `iso_lead_pt` | log1p(pT of the most isolated pT>10 candidate), standardized |
| 12 | `n_iso` | number of pT>10 candidates with cone-pT/pT < 0.15, standardized |
| 13-18 | `p12_lndRc` … `p34_lndRc` | ½ln(Δη² + 2(1−cosΔφ)) for the 6 pairs among the leading 4 |

All eight are standardized with frozen constants in `data.EXTRA_STANDARDIZE`
(re-measure with `python data.py --fit-extra-norm`). The pair distance uses
`2(1−cosΔφ)` rather than `Δφ²` deliberately: it needs no atan2 and no 2π wrap in
firmware, and it measures the same (+0.0158 vs +0.0156 AUC vs tt). Full firmware
costing of every one of these — formula, cost class, fixed-point rewrite — is in
`team/fpga/FEATURES.md`.

## What to take, and what to leave

Answering the two questions that were asked, with the numbers above:

**1. Ranked, the teacher's derived quantities.** The 6 per-candidate `rich` channels
are the prize (`rich:ALL`, +0.033 vs tt in the proxy, +0.037 in the real student).
Within them, |dxy| and Δφ-to-the-leading-candidate carry most of it (+0.012 each);
ln pt/HT and ln E carry almost nothing on their own (+0.003) because HT and the
leading pTs are already event features. Of the pairwise quantities, **ln ΔR is the
useful one** (+0.016 from 6 numbers), ln kT and ln m² are slightly weaker and largely
the same information, and **ln z is worth nothing at all** (−0.000).

**2. What a trigger student can afford.** In priority order:

* **The 6 rich channels — take them.** +0.037 vs tt and +0.0136 overall in the actual
  2,057-parameter model, for +192 parameters. They are all O(n): ln(pt/HT) needs HT
  (already summed), ln E needs cosh η, Δφ/Δη are differences against the leading
  candidate, |dxy| is an absolute value. The catch is that they widen φ, the block
  the FPGA lane already has at 91% of an SLR — so take them *and* pay for them by
  narrowing φ or dropping to 8 candidates (rows `c2_canon_narrow` / `c2_canon_8p`,
  both cheaper than today's baseline).
* **Max-pooling alongside mean — take it.** Comparators, no DSP, +8 inputs to ρ.
* **`iso_lead_pt` (and `n_iso`, free once the ΔR table exists) — take them.** One
  event-level scalar is worth more against tt (+0.024) than all 24 leading-4 pair
  numbers together (+0.016), and it goes in after the pool where there is room.
* **Pair quantities — take ln ΔR of the leading-4 pairs.** 6 scalars, +0.016 in the
  proxy and +0.013 vs tt in the real student. Adding ln kT and ln m² on top buys
  +0.000 (they are the same information
  in different coordinates); ln z buys nothing anywhere.
* **A full pairwise block — no.** Pooling all 120 pairs into mean/min/max/std keeps
  only +0.009 of the +0.023 that the explicit leading-6 pairs give, so the value is
  in *which* pair, not in the pairwise ensemble. A student cannot afford 16×16×4
  inputs, and the ceiling it would be reaching for is modest.

And the framing correction, restated because it changes what to chase: the published
teacher's 0.826 vs tt comes from per-candidate features plus capacity, not from
relational information. A ParT teacher may still change that — but on this evidence,
pairwise is the smaller half.

<!-- c2:tt-study:end -->
