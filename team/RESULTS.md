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

Unconstrained teachers, trained on `train1M` (1M signal + 1M background, 10 % held out for
checkpoint selection) and evaluated on the same `eval100k` slice as every row above, with the
same AUC definitions as `team/train.py` (AUC vs a background group = signal vs that group only).

Both teachers see **exactly the student's inputs** — the cache tensors `X` (16 candidates × 5
features) and `F` (11 event features). Nothing outside them is used; the teacher only derives
extra quantities from them on the fly (`team/teacher/common.py`), so the logits are valid soft
targets for a student that consumes those same tensors.

Training: AdamW, 2-epoch warm-up then cosine, label smoothing 0.05, bf16 autocast, EMA of
weights and BatchNorm buffers, gradient clipping at 1.0, batch 2048.

| run | model | params | epochs (best) | **AUC (eval)** | vs QCD | **vs tt** | vs W+jets | eff@99 % rej | eff@99.9 % rej |
|---|---|---|---|---|---|---|---|---|---|
| `B1e_16p_1M` *(student, for reference)* | DeepSet φ32-16-8 ρ32-16 + evt | 2,057 | 25 | 0.88687 | 0.93027 | 0.75869 | 0.97163 | — | — |
| `ds_big_s0` | BigDeepSet φ128-64-32, mean+max, ρ256-128-64 | 72,717 | 40 (23) | 0.91515 | 0.94363 | 0.82612 | 0.97569 | 0.2717 | 0.0747 |
| `part_s0` | ParT-lite d128, 4 blocks × 8 heads | 1,334,013 | 50 (19) | 0.92392 | 0.94543 | 0.85042 | 0.97591 | 0.3182 | 0.0926 |
| `part_s1` | ParT-lite, seed 1 | 1,334,013 | 50 (18) | 0.92385 | 0.94553 | 0.85022 | 0.97580 | 0.3108 | 0.0903 |
| `part_e25_s2` | ParT-lite, 25 ep, dropout 0.15 | 1,334,013 | 25 (18) | 0.92364 | 0.94542 | 0.84985 | 0.97566 | 0.3163 | 0.0954 |
| `part_e25_d20_s3` | ParT-lite, 25 ep, dropout 0.2, wd 0.1 | 1,334,013 | 25 (23) | 0.92358 | 0.94523 | 0.85007 | 0.97545 | 0.3124 | 0.0897 |
| `ens_part2` | mean logit of `part_s0` + `part_s1` | 2,668,026 | — | 0.92450 | 0.94604 | 0.85130 | 0.97615 | 0.3171 | 0.0951 |
| `ens_part4_ds` | the 4 ParT seeds **+ `ds_big_s0`** | 5,408,769 | — | 0.92468 | 0.94713 | 0.85010 | 0.97681 | 0.3195 | 0.0982 |
| **`ens_part4`** ← **published** | mean logit of the 4 ParT-lite seeds | 5,336,052 | — | **0.92480** | 0.94636 | **0.85181** | 0.97622 | 0.3186 | 0.1013 |

**Headline: the teacher is +0.0379 AUC overall and +0.0931 vs tt over the student it will teach**
(0.92480 vs 0.88687; 0.85181 vs 0.75869). Almost all of the headroom distillation can transfer is
against tt — the background the student is worst at — which is exactly where round 3 wanted it.

## What the sweep says

**1. ParT-lite beats the big DeepSet by +0.0097, and by +0.0243 vs tt** (0.92392 vs 0.91515;
0.85042 vs 0.82612). The pairwise (ΔR, ln kT, ln z, ln m²) attention bias is doing real work: a
DeepSet sees candidates independently and only ever pools them, so it cannot represent the
two-body mass structure that separates HH→4b (two ~125 GeV pairs) from tt̄ (a 80 GeV W plus a
173 GeV top). The bias hands that structure to the attention directly.

*This answers c2's open question directly.* c2's tt study concluded that the teacher's vs-tt
number "comes from per-candidate features plus capacity, not from relational information", with
"pairwise is the smaller half", and flagged that a ParT teacher might change that. It does: the
BigDeepSet and ParT-lite here are trained on the same rows with the same per-candidate features,
and the only structural difference is the pairwise attention bias. It is worth **+0.0243 vs tt**
(0.82612 → 0.85042) — more than the whole per-seed and regularization spread combined. On this
evidence relational information is the larger half against tt, not the smaller one.

**2. The ParT-lite configuration is saturated.** Four runs spanning seeds, 25 vs 50 epochs, dropout
0.1–0.2 and weight decay 0.05–0.1 span **0.00034 AUC** (0.92358–0.92392). Neither a shorter cosine
schedule matched to the peak epoch nor heavier regularization beat the default, so the remaining
gap to a perfect teacher is not a tuning problem at this width.

**3. Ensembling is worth +0.0009**, two seeds giving +0.0006 and four +0.0009 over the best single
seed — a normal, small, real gain for averaged logits.

**4. Adding the DeepSet to the ensemble hurts the headline number.** `ens_part4_ds` gains vs QCD
(+0.0008) and vs W+jets (+0.0006) but loses vs tt (−0.0017), netting −0.0001 overall. The DeepSet
is 0.0087 weaker and its errors against tt are not decorrelated enough to pay for that. **So the
published targets are the 4 ParT seeds only**; the DeepSet targets are kept beside them as
`soft_targets_*_dsbig.npy` for anyone who wants to compare or blend.

## The soft targets

| file | rows | content |
|---|---|---|
| `team/teacher/soft_targets_train1M.npy` | 2,000,000 | float32 `ens_part4` logits, `team/cache/train1M` row order |
| `team/teacher/soft_targets_train300k.npy` | 599,999 | same, `train300k` |
| `team/teacher/soft_targets_eval100k.npy` | 200,000 | same, `eval100k` |
| `team/teacher/soft_targets_*_dsbig.npy` | same | the `ds_big_s0` DeepSet logits, for comparison |
| `team/teacher/soft_targets_meta.json` | — | source run, members, params, eval AUCs, usage note |

Teacher AUC **on the files themselves**: 0.93040 on `train1M`, 0.93111 on `train300k`, 0.92480 on
`eval100k` (the train caches read higher because 90 % of their rows are in-sample; see below).

**Two things c1 needs when consuming these.**

*Use the logits, not probabilities.* Student score = `sigmoid(logit)`; for KD at temperature T use
`sigmoid(logit / T)`. The teacher trained with **label smoothing 0.05**, so its probabilities
saturate near 0.975 / 0.025 rather than 1 / 0 — the targets are already softened before any T.

*The in-sample rows are safe to use.* Each teacher trained on 90 % of `train1M`, so most soft
targets are in-sample. Measured, not assumed: the teacher's AUC on rows it trained on is only
**+0.0057** above rows it did not (0.93044 vs 0.92477 for `part_s0`), and its confidence is
**identical** on both (mean |logit| 1.996 in-sample vs 1.995 held-out). It is not memorizing, so
cross-fitting the targets is unnecessary and the full-cache logits can be used directly.

## Not done

- **A 32-candidate privileged teacher** (the optional item) is not possible on this pod: the raw
  parquet dataset is not mounted here, and the caches contain 16 candidates only.
- **Bigger and deeper ParT variants** (d192 × 6 blocks, 8 blocks, and a tt-upweighted run) were
  launched and cut at 5–11 of 30 epochs when the shared A100 was handed back. They were tracking
  the baseline, not beating it (val AUC 0.9213 at epoch 5 vs 0.9248 final), and finding 3 above
  says the configuration is saturated anyway.

## Phase 2 — a 4-class teacher, and what it adds

The same ParT-lite trunk with a **softmax head over the four processes** (`QCD, HH_4b, tt, Wjets`
— the column order is `team/data.py` `GROUP_ID`, so the class target *is* the cache's `group.npy`),
trained with cross-entropy and label smoothing 0.05. Its binary number is derived exactly, with no
refitting: `logit(HH) − logsumexp(logit(background))` is the log-odds of the softmax's HH
probability against the pooled background, so it is directly comparable to the binary head's logit.
Verified against `log(p/(1−p))` of the softmax and shown invariant to a constant shift of all four
logits.

`part4c_s0` uses seed 0 and every other hyperparameter of `part_s0`, so the head and the loss are
the only intended differences. It ran 25 epochs rather than 50; `part_s0`'s best epoch was 19 and
the 25-epoch binary variants landed within 0.0003 of it, so the epoch budget is not the confound.

| run | head | params | **AUC (eval)** | vs QCD | **vs tt** | vs W+jets | eff@99 % | eff@99.9 % |
|---|---|---|---|---|---|---|---|---|
| `part_s0` | binary (BCE) | 1,334,013 | 0.92392 | 0.94543 | 0.85042 | 0.97591 | 0.3182 | 0.0926 |
| **`part4c_s0`** | 4-class (CE) | 1,334,208 | **0.92460** | **0.94675** | **0.85110** | **0.97594** | 0.3161 | 0.0955 |
| Δ | | +195 | **+0.00068** | +0.00132 | +0.00068 | +0.00003 | | |

**The 4-class head is a small free win.** It beats the binary head on the pooled AUC and on every
background separately, for 195 extra parameters and half the epochs. Telling the network *which*
background it is looking at is a richer training signal than "not signal", and the gain is largest
against QCD (+0.0013) — the class the binary head can most afford to be lazy about, since QCD is
already the easiest background.

**Where the teacher actually fails**, from the 4-class confusion matrix on the eval slice (rows are
the true class, normalized; predicted class = argmax over the four logits):

| true \ predicted | QCD | HH_4b | tt | Wjets |
|---|---|---|---|---|
| QCD | 0.769 | 0.145 | 0.036 | 0.049 |
| **HH_4b** | 0.037 | **0.890** | 0.051 | 0.022 |
| **tt** | 0.065 | **0.398** | 0.508 | 0.028 |
| Wjets | 0.088 | 0.058 | 0.023 | 0.831 |

Overall 4-class accuracy is 0.796. The tt row is the whole story of this challenge: **40 % of tt
events are called HH outright**, against 15 % for QCD and 6 % for W+jets. tt is not merely the
weakest background in AUC, it is the one the teacher genuinely confuses with signal at the decision
point — a hadronic tt̄ event really does contain four b-ish jets. This is the diagnostic version of
the vs-tt AUC gap, and it is what any tt-specific feature has to move.

### Two more seeds at a larger batch

With the A100 free, `part_b4k_s6/s7` repeat `part_s0` at **batch 4096** with a √2-scaled learning
rate (1.4e-3), 30 epochs.

| run | batch | AUC (eval) | vs tt |
|---|---|---|---|
| `part_b4k_s6` | 4096 | 0.92338 | 0.84896 |
| `part_b4k_s7` | 4096 | 0.92318 | 0.84872 |
| *(batch-2048 seeds, for comparison)* | 2048 | 0.92358 – 0.92392 | 0.84985 – 0.85042 |

**The larger batch cost about 0.0004 AUC** and did not pay for itself: both batch-4096 seeds land
below all four batch-2048 seeds. √2 LR scaling under-compensates here, so batch 2048 stays the
recommended setting. The seeds are still useful as ensemble members.

### Ensembles, including the 4-class member

A 4-class member contributes its HH-vs-background log-odds, which is the same quantity the binary
head emits, so averaging across head types is meaningful rather than a scale mismatch.

| ensemble | members | AUC (eval) | vs QCD | vs tt | vs W+jets |
|---|---|---|---|---|---|
| `ens_part4` ← **currently published** | 4 binary seeds | 0.92480 | 0.94636 | 0.85181 | 0.97622 |
| `ens_part6` | + the 2 batch-4096 seeds | 0.92485 | 0.94650 | 0.85173 | 0.97632 |
| `ens_part6_4c` | those 6 + `part4c_s0` | 0.92508 | 0.94678 | 0.85205 | 0.97640 |
| **`ens_part4_4c`** | 4 binary seeds + `part4c_s0` | **0.92511** | 0.94676 | **0.85223** | 0.97634 |

The 4-class model is the most valuable single addition (+0.0003 over `ens_part4`, more than the two
extra binary seeds contribute), because it is the only member whose errors come from a different
training objective. The two weaker batch-4096 seeds dilute it slightly, so the best combination is
the four batch-2048 seeds plus the 4-class model.

**`soft_targets_*.npy` is deliberately NOT updated to `ens_part4_4c`.** The c3 and c1 distillation
jobs in `team/A100_QUEUE.md` are reading `soft_targets_train1M.npy` right now, and `c3-2` is
explicitly measuring run-to-run spread across four runs started at different times. Swapping the
teacher underneath them would make those numbers incomparable, for a teacher gain of +0.0003 that is
worth ~nothing to students sitting 0.012–0.019 below the teacher. The better logits are on disk at
`team/teacher/runs/ens_part4_4c/logits_*.npy` and the swap can happen once the distillation lanes
are quiescent — say the word.

### 4-class soft targets

| file | shape | content |
|---|---|---|
| `team/teacher/soft_targets4_train1M.npy` | (2,000,000, 4) | float32 class logits, `train1M` row order |
| `team/teacher/soft_targets4_train300k.npy` | (599,999, 4) | same, `train300k` |
| `team/teacher/soft_targets4_eval100k.npy` | (200,000, 4) | same, `eval100k` |
| `team/teacher/soft_targets4_meta.json` | — | source run, class order, AUCs, usage |

Columns are `['QCD', 'HH_4b', 'tt', 'Wjets']`. Class probabilities are `softmax(logits, axis=1)`;
the binary score is `logits[:,1] - logsumexp(logits[:,[0,2,3]], axis=1)`; for KD at temperature T
use `softmax(logits / T, axis=1)`. The binary `soft_targets_*.npy` files are untouched and remain
byte-identical to what was published in phase 1 — verified against `origin/main`.

These are the targets to use for a **student with a 4-class head**, which is worth trying: the
confusion matrix says the tt-vs-HH boundary is where the loss is, and a student trained to name the
background gets a gradient that says so.

## Reproduce

```bash
cd team/teacher
PY=/work/users/das214/fastml26/venv/bin/python
$PY train_teacher.py --model part --tag part_s0 --epochs 50 --ema 0.999 --compile --seed 0
$PY train_teacher.py --model part --tag part_s1 --epochs 50 --ema 0.999 --compile --seed 1
$PY ensemble.py --runs part_s0 part_s1 part_e25_s2 part_e25_d20_s3 --name ens_part4 --publish
# phase 2: the 4-class teacher (publishes soft_targets4_*, never touches the binary files)
$PY train_teacher.py --model part --tag part4c_s0 --n-classes 4 --epochs 25 --ema 0.999 \
    --compile --seed 0 --publish
$PY ensemble.py --runs part_s0 part_s1 part_e25_s2 part_e25_d20_s3 part4c_s0 --name ens_part4_4c
```

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
| `c2_canon_narrow` | canonical set, φ 24-12-8 | 10,368 | 2,421 | 0.90150 | 0.8008 | 0.9316 | 0.9721 | 0.2075 |
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

## Stage 6 — is the student limited by its inputs or by its shape?

Gradient-boosted trees on **event scalars only** — no particle branch at all —
on the same train300k/eval100k split the DeepSet rows use.

| setup | cols | AUC (all) | vs QCD | vs tt | vs W+jets |
|---|---|---|---|---|---|
| A  11 event features only | 11 | 0.8742 | 0.9173 | 0.7392 | 0.9661 |
| B  A + |dxy| order statistics | 15 | 0.8813 | 0.9279 | 0.7472 | 0.9689 |
| C  A + affordable event scalars | 56 | 0.8998 | 0.9327 | 0.7948 | 0.9718 |
| D  C + jet-clustered scalars | 74 | 0.9038 | 0.9359 | 0.8030 | 0.9724 |
| E  D + rich summaries + pair distances | 128 | 0.9052 | 0.9361 | 0.8064 | 0.9730 |
| *B1e_16p_1M (2,057 params, 2M events, GPU)* | — | 0.8869 | 0.9303 | 0.7587 | 0.9716 |
| *c2_base_cpu (same net, 600k events, CPU)* | — | 0.8840 | 0.9292 | 0.7514 | 0.9712 |
| *c2_canon (canonical inputs, 600k, CPU)* | — | 0.9010 | 0.9339 | 0.7961 | 0.9729 |

**The answer is feature content, not representation.** The 11 incumbent scalars
alone reach 0.8742 against the DeepSet's 0.8840 — the entire 16-candidate
particle branch is worth about 0.010. Give the same trees the event scalars this
lane built and they reach 0.8998, *beating* the DeepSet by 0.016 with no
per-particle processing whatsoever, and 0.9052 with everything. Hours spent on
features have been paying roughly twice what hours spent on architecture would.

Two riders. The ceiling here is a 100-tree GBDT, not something that fits an SLR —
it says where the information is, not what to ship. And the jet-clustered scalars
(row D) are the part of the ceiling a trigger cannot afford; the affordable row C
is already 0.8998.

## Stage 7 — the two parquet fields we were not reading

`team/physics/COLUMNS.md` inventories the files: we use 4 of `L1T_PUPPIPart`'s 14
subfields. Two of the other ten answer questions this lane spent the day
approximating — **`pdgId`** flags electrons and muons directly (`iso_lead_pt` is a
hand-built proxy for exactly that), and **`dxysig`** is the impact-parameter
*significance*, the variable a real b-tagger uses, where we feed raw `dxy`.
Trees on the 11 incumbent scalars, plus each block (train/eval read straight from
parquet, 120k train / 60k eval):

| features | AUC (all) | vs tt | vs QCD | vs W+jets |
|---|---|---|---|---|
| 11 event features | 0.8710 | 0.7314 | 0.9150 | 0.9666 |
| + dxysig block (10) | 0.8819 | 0.7470 | 0.9285 | 0.9700 |
| + pdgId block (9) | 0.8865 | 0.7711 | 0.9199 | 0.9685 |
| + both (19) | 0.9065 | 0.8076 | 0.9371 | 0.9746 |

**+0.036 overall and +0.076 vs tt from two fields already on disk** — more than
every hand-made feature in this document put together. The strongest singles vs tt:
  `dsig_ptw` 0.687, `dsig_sum` 0.672, `dsig_mean` 0.672, `n_dsig_gt5` 0.672, `n_dsig_gt3` 0.671, `n_dsig_gt2` 0.671.

`dxysig` is stored as float16 and overflows — |dxysig| reaches `inf` and its p99 is
in the thousands — so it must be clipped (20.0 here) before anything touches it.
`data.py` now reads both fields (only when a feature needs them) and exposes nine
derived event scalars; all nine are O(16) reductions, no DSP, no new pair table.

## Stage 8 — hadronic tt: is there a top tag in the candidates?

The lepton features fixed the easy tt modes and left the hard one (0.708 vs
hadronic tt on the canonical set). All-hadronic tt is a W→qq inside a t→Wb, so:
the 15 dijet masses among the leading 6 candidates, `min |m_jj − 80|`, and the 20
trijet masses, `min |m_jjj − 173|`. Cheap, because for massless constituents
`m_ijk² = m_ij² + m_ik² + m_jk²` — the 20 triples are adds once the 15 pairs exist.

| on top of the canonical 19 | tt hadronic | tt semi-lep | tt leptonic | QCD | W+jets | all bkg |
|---|---|---|---|---|---|---|
| A  canonical 19 | 0.7083 | 0.7877 | 0.8420 | 0.9158 | 0.9723 | 0.8452 |
| B  + top6 (6 cols) | 0.7086 | 0.7877 | 0.8418 | 0.9159 | 0.9726 | 0.8453 |
| C  + dxy order stats (4 cols) | 0.7252 | 0.7936 | 0.8412 | 0.9271 | 0.9740 | 0.8522 |
| D  + both (10 cols) | 0.7247 | 0.7933 | 0.8407 | 0.9270 | 0.9739 | 0.8519 |

**The top tag is dead: +0.0003 against hadronic tt.** And the single-feature numbers
say why it was never going to work — `dm_top6` scores 0.571 against *hadronic* tt
and 0.651 against *leptonic* tt. A real top tag would do the opposite. It is
picking up generic kinematics, not a resonance: the inputs are particle-flow
candidates, not jets, so two of the leading six routinely come from the same jet
and the leading six rarely span two tops. This is the third independent way the
same conclusion has arrived (jet-clustered `dm_W`/`dm_top` +0.002, the HH dijet
pairing `dm_higgs` +0.000, and now this) — **mass reconstruction is not available
at this input granularity, and no more hours should go into it.**

**The |dxy| order statistics are the opposite story: +0.017 against hadronic tt,**
the mode nothing else moved, plus +0.006 semi-leptonic and +0.011 vs QCD, for four
numbers and a comparator network. They are in `data.py`'s canonical set.

## The organizers' eval mixture is tt-dominated, and ours is not

Row counts straight out of `eval/` (2,102,226 events, 47.6% of them signal):

| process | events | share of background |
|---|---|---|
| `HH_4b` | 1,000,384 | — (signal) |
| `QCD_HT250toInf` | 100,482 | 9.12% |
| `tt0123j_5f_ckm_LO_MLM_hadronic` | 200,327 | 18.18% |
| `tt0123j_5f_ckm_LO_MLM_leptonic` | 200,111 | 18.16% |
| `tt0123j_5f_ckm_LO_MLM_semiLeptonic` | 200,110 | 18.16% |
| `WJetsToLNu_13TeV-madgraphMLM-pythia8` | 200,281 | 18.18% |
| `WJetsToQQ_13TeV-madgraphMLM-pythia8` | 200,531 | 18.20% |

Grouped, the official background is **QCD 9.1%, tt 54.5%, W+jets 36.4%** — against our
even thirds. So **no: the official metric weights QCD far *less* than our eval
slice does (9% against 33%), and tt far more (55% against 33%).** Our worst
background is the official metric's dominant one. Re-weighting our saved eval
scores to those proportions (AUC is a rank statistic, so only the background
composition matters):

| run | our even mix | official eval mix | Δ |
|---|---|---|---|
| `B1e_16p_1M` | 0.88687 | 0.85180 | -0.03507 |
| `c2_base_cpu` | 0.88397 | 0.84761 | -0.03636 |
| `c2_pair4` | 0.88785 | 0.85455 | -0.03330 |
| `c2_rich` | 0.89758 | 0.86821 | -0.02937 |
| `c2_canon` | 0.90099 | 0.87300 | -0.02799 |
| `c2_canon_narrow` | 0.90150 | 0.87504 | -0.02646 |

Every number drops about 0.03 under the official mixture — and the tt work gains
in importance rather than losing it: `c2_canon_narrow` beats `B1e_16p_1M` by
+0.0146 on our slice and by **+0.0232** on the organizers'. Within tt the three
decay modes are equal in `eval/` (200k each), which matches our sampling; the
Standard-Model-branching-fraction caveat earlier in this section is about physical
realism, not about the challenge metric.

## `train4M` is built and waiting — for c1

`team/cache/train4M/` (5.6 GB on disk), built on the CPU box with the canonical
input set: **X (7,569,258, 16, 11) float32, F (7,569,258, 19), y, group**, streamed
from parquet so peak memory stayed near 11 GB. `train.py` reads it with no change —
`--train-tag train4M --eval-tag eval100k_c2 --pool meanmax`.

**Two things to know before you use it.** First, it is 4,000,000 signal against
3,569,258 background, not 4M+4M: QCD ran out. The whole `train/QCD_HT250toInf`
directory holds 902,592 events, so a QCD budget of 1,333,333 could not be met.
The background is therefore **QCD 25.3%, tt 37.4%, W+jets 37.4%**, not even thirds.
Second, that is *closer* to the organizers' eval mixture (QCD 9%, tt 55%, W 36%) than
even thirds is, so it is arguably the better training mixture — but it is a different
mixture from `train1M`, and a model trained on it is not a clean A/B against a
`train1M` row. If you want strict even thirds at this scale the ceiling is
2,707,776 background events (3 × 902,592).

A `train4M` cache with the *newer* `dxysig`/`pdgId` features (F of 31 rather than 19)
is a rebuild away; say the word and it runs.

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

## Teacher quality is not the bottleneck — student capacity is

The team teacher was upgraded mid-round from `ds_big_s0` (AUC 0.91515, tt 0.82612) to
`ens_part4`, a 4-member ParT ensemble (AUC 0.92480, tt 0.85181). Same student shape, same
recipe, re-run against the stronger targets:

| student | teacher | teacher AUC | T | alpha | AUC (eval) | AUC vs tt |
|---|---|---|---|---|---|---|
| `rkd_T2_a05` | ds_big_s0 | 0.91515 | 2 | 0.5 | **0.90901** | 0.81188 |
| `pkd_T2_a07` | ens_part4 | 0.92480 | 2 | 0.7 | 0.90870 | **0.81367** |
| `pkd_T2_a05` | ens_part4 | 0.92480 | 2 | 0.5 | 0.90805 | 0.81253 |
| `pkd_T2_a03` | ens_part4 | 0.92480 | 2 | 0.3 | 0.90779 | 0.81121 |

**A teacher +0.0096 better produced a student −0.0003 different.** At 2,777 parameters the
student is capacity-limited, not supervision-limited, so effort is better spent on inputs
(c2's rich channels bought +0.022) or on more data than on a stronger teacher.

## Targeting tt in the loss (screened on `train300k_s`, 30 epochs)

| config | overall | vs QCD | **vs tt** | vs W+jets |
|---|---|---|---|---|
| baseline | 0.90398 | 0.93626 | 0.80280 | 0.97288 |
| tt ×2 in the hard term | 0.90409 | 0.93349 | 0.80681 | 0.97195 |
| tt ×3 | 0.90294 | 0.93003 | 0.80946 | 0.96933 |
| disagreement-weighted KD | 0.90479 | 0.93615 | 0.80544 | 0.97279 |
| **tt ×2 + disagreement** | **0.90488** | 0.93437 | **0.80901** | 0.97126 |

**tt upweighting is close to zero-sum**: ×2 buys +0.0040 on tt and pays −0.0028 on QCD and
−0.0009 on W (net +0.0001 overall); ×3 buys +0.0067 on tt and pays −0.0062/−0.0036 (net
−0.0010). It moves the operating point rather than improving the model.

**Disagreement weighting is a genuine free gain** — weighting each event by
`1 + |z_teacher − z_student| / mean` adds +0.0008 overall *and* +0.0026 on tt with no
measurable QCD/W cost, because it spends capacity on the events the student actually gets
wrong rather than on a class label. The two compose: together +0.0062 on tt for −0.0019 on
QCD, the best overall point of the five. Promoted to `train1M_s` as `rich_1M_w2dis` (40 epochs), which confirms it at scale:

| student | overall | vs QCD | **vs tt** | vs W+jets | sig eff @99.9% |
|---|---|---|---|---|---|
| `rkd_T2_a05` (current export) | **0.90901** | 0.93109 | 0.81188 | 0.97059 | 0.0541 |
| `rich_1M_w2dis` (tt ×2 + disagree) | 0.90864 | 0.93578 | **0.81814** | 0.97200 | **0.0628** |

Statistically tied overall (−0.0004) but **+0.0063 on tt**, the background that limits us,
and +0.009 signal efficiency at 99.9% background rejection — the trigger-relevant number.
Exported as `model_2777_rich_tt` alongside the incumbent so both can be synthesized.

## HGQ2 QAT (Lane 2)

**Toy gate passed first**, as the brief required: a 2-layer mean-pool + concat DeepSet in
`hgq.layers` converts through `hls4ml` (Vitis, xcu200) with csim matching Keras to 1e-3 —
and so does the mean+max variant our student actually uses, which was the real risk.
`hgq/toy_gate.py`, run before spending any GPU time.

Environment note: `hls4ml[hgq2]` needs a Keras 3 backend and the venv had none. Rather than
install a second CUDA stack, the venv was rebuilt with `--system-site-packages` and Keras
runs on the **existing CUDA torch** (`KERAS_BACKEND=torch`) — no JAX, no TensorFlow.

| run | beta0 | epochs | AUC (eval) | vs tt | EBOPs | vs float |
|---|---|---|---|---|---|---|
| float reference (`rkd_T2_a05`) | — | — | 0.90901 | 0.81188 | — | — |
| `qat_b1e-6` | 1e-6 | 12 | 0.89044 | 0.78791 | 853k → **360k** | −0.0186 |

EBOPs — the quantity that drives DSP, now the binding resource at 1,692/1,900 — fell **2.4×**,
but the AUC cost is far too high: 0.890 is below the 0.895 bar and well below what the PTQ
path already achieves in real Vitis (0.906). Two causes, both fixable:

* **Val AUC peaked at epoch 6 (0.89477) and then decayed** under rising EBOPs pressure, and
  the script was saving the *final* epoch. `qat_hgq.py` now keeps the best-val epoch.
* **12 epochs is far too short.** Warm-started pre-QAT AUC is 0.548 — the learned bit widths
  start from nothing, so a large part of training is just the quantizers settling.

Lower betas at 30 epochs (1e-7, 3e-7) are running; the long runs are queued for the A100 as
job `c1-1`. `hgq/convert.py` does the required convert → `compile()` → closure (overall and
per background) → `write()` → tar into `fpga/projects/<tag>.tar.gz`, with
`Strategy=distributed_arithmetic`, `RF=1` and **no manual precision** (HGQ2 carries its own
per-layer widths; overriding them would throw away what QAT trained).

**Not done this round:** the 4-class softmax head. Flagged, not attempted.
