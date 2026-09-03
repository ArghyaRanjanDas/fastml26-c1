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

## Reproduce

```bash
cd ~/fastml26-hackathon/team
python train.py --model deepset_plus --rho 256,128,32 --dropout 0.1 --pool-norm \
                --no-event-features --tag ctl_pn
```
