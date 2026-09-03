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

## Reproduce

```bash
cd team/teacher
PY=/work/users/das214/fastml26/venv/bin/python
$PY train_teacher.py --model part --tag part_s0 --epochs 50 --ema 0.999 --compile --seed 0
$PY train_teacher.py --model part --tag part_s1 --epochs 50 --ema 0.999 --compile --seed 1
$PY ensemble.py --runs part_s0 part_s1 part_e25_s2 part_e25_d20_s3 --name ens_part4 --publish
```
