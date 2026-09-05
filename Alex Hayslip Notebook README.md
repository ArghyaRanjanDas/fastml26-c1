# HH→4b Signal Classification — Trigger-Level Event Tagging

**San Diego Data Science Alliance Hackathon — Challenge 1: Di-Higgs (HH→4b) Discrimination**
Author: Alex Hayslip · 2nd Place

## Overview

This notebook builds and compares machine learning models to classify simulated particle-collider
events as either **HH→4b signal** (di-Higgs production decaying to four bottom quarks) or
**background** (QCD multi-jet, ttbar, or W+jets), using low-level Level-1 trigger (L1T) particle
candidates. The task mirrors a real trigger-system problem: identifying rare signal events fast
enough, and accurately enough, to decide what's worth recording at the HL-LHC.

Performance is evaluated with **ROC-AUC**, matching the challenge's official metric, since a
trigger's operating threshold is tuned after training rather than fixed at 0.5.

## Dataset

- **Source:** `L1T_PUPPIPart` — Level-1 trigger PUPPI (Pileup Per Particle Identification)
  particle-flow candidates, one entry per reconstructed particle per event.
- **Samples used** (3,000 events each, 21,000 events total):

  | Sample | Label | Class |
  |---|---|---|
  | `HH_4b` | 1 | Signal |
  | `QCD_HT250toInf` | 0 | Background |
  | `tt0123j_5f_ckm_LO_MLM_hadronic` | 2 | Background |
  | `tt0123j_5f_ckm_LO_MLM_leptonic` | 2 | Background |
  | `tt0123j_5f_ckm_LO_MLM_semiLeptonic` | 2 | Background |
  | `WJetsToLNu_13TeV-madgraphMLM-pythia8` | 3 | Background |
  | `WJetsToQQ_13TeV-madgraphMLM-pythia8` | 3 | Background |

- **Per-candidate fields used:** `pt`, `eta`, `phi`, `dxy`, `dxysig`, `pdgId`, `charge`,
  `pt_weighted`, `puppi_weight`, `e`, `mass`, `dz`, `error_dz` (dropped during cleaning),
  `funique_id` (dropped — pure identifier, no physics content).
- **Final classical-ML dataset:** 10,370,440 particle-level rows after cleaning.
- **Binary target:** `is_signal = 1` if `label == 1` (HH_4b), else `0`. Class imbalance handled
  throughout via `class_weight="balanced"` / explicit `compute_class_weight`.

## Pipeline

### 1. Schema exploration
Parsed the nested/jagged Parquet schema (Awkward Array) to understand the event → particle
structure before flattening anything into a DataFrame.

### 2. Data cleaning
- Dropped `funique_id` (bookkeeping only, no physics signal).
- Dropped `error_dz` — every value was exactly `0.0` (a sentinel/non-informative column).
- Found **3,212 rows (0.031%)** where `dxysig` was `±inf`, caused by division by the
  now-dropped zero-valued `error_dz`. These rows were **dropped**.
- Found and dropped **10 exact duplicate rows**.
- Verified no null values anywhere in the cleaned candidate table.

### 3. Exploratory Data Analysis
- Per-class summary statistics (`pt`, `eta`, `e`, `mass`, `dxy`, `dz`).
- Correlation heatmap across kinematic features — found `dxy`/`dxysig` near-perfectly
  correlated (1.00), and `pt`/`pt_weighted` strongly correlated (0.87).
- Derived `Subatomic_Charge_classification` (Negative/Neutral/Positive) and
  `Has_Dummy_Particle` flags for candidate-level breakdowns.
- **K-Means clustering** (`pt`, `mass`, k=3) to test whether kinematics alone separate
  particles by charge. Result: cleanly isolates **neutral vs. charged** particles
  (91% purity on the neutral cluster) but **cannot** separate positive from negative charge —
  consistent with the physics, since charge sign requires curvature information not present
  in `pt`/`mass` alone.

### 4. Feature engineering: particle-level → event-level
Initial models trained on raw per-particle rows performed at chance level (**AUC ≈ 0.52**),
since a single particle carries almost no information about which physics process produced
the whole event. Aggregating ~22 features per event (sums, means, standard deviations of
`pt`, `e`, `mass`, `dxysig`, charge counts, PUPPI weights, dummy-particle fraction, etc.)
was the single largest driver of model performance in the whole project.

### 5. Model comparison
Nine classifiers benchmarked on the 22 event-level features (5-fold stratified CV,
class-balanced, 80/20 train/test split):

| Model | CV AUC | Test AUC |
|---|---|---|
| **HistGradientBoosting** | 0.7880 ± 0.0120 | **0.8003** |
| Gradient Boosting | 0.7899 ± 0.0115 | 0.7981 |
| Random Forest | 0.7859 ± 0.0101 | 0.7952 |
| LightGBM | 0.7729 ± 0.0084 | 0.7866 |
| SVM | 0.7708 ± 0.0115 | 0.7783 |
| Logistic Regression | 0.7307 ± 0.0127 | 0.7222 |
| KNN | 0.6864 ± 0.0096 | 0.7051 |
| Gaussian Naive Bayes | 0.6850 ± 0.0115 | 0.6755 |
| Decision Tree | 0.5707 ± 0.0101 | 0.5581 |

### 6. Hyperparameter tuning
`RandomizedSearchCV` (50 candidates × 5-fold CV) on `HistGradientBoostingClassifier`.
Tuning barely moved AUC (**0.7994** tuned vs. **0.8003** default) — the untuned model was
already near-optimal, indicating the feature set, not the model or its hyperparameters,
was the limiting factor.

### 7. Feature selection via permutation importance
Ranked all 22 features by permutation importance on the held-out test set. `dxysig_std`
(the spread of displacement significance within an event) dominated, at roughly 3.5× the
importance of the next-best feature (`pt_max`) — a plausible low-level proxy for b-jet
identification, since B hadrons travel a measurable distance before decaying.

Trimming to the top 13 features and refitting `HistGradientBoostingClassifier` (default
hyperparameters) gave the **best result in the notebook: Test AUC = 0.8022**. Re-tuning
this trimmed feature set brought it slightly *down* to 0.7990, confirming the defaults
generalized better than an over-tuned fit on this smaller feature space.

### 8. Particle Transformer (raw per-particle input)
To move beyond hand-engineered aggregates, a custom `ParticleTransformer` (PyTorch) was
trained directly on the padded per-particle tensor (16 particles × 4 features: `pt`, `eta`,
`phi`, `dxy`), using:
- A learned `[CLS]` token for permutation-invariant event-level pooling
- A boolean padding mask so the model never attends to zero-padded slots
- A 3-layer Transformer encoder with masked self-attention
- Class-weighted `BCEWithLogitsLoss`, trained ~140 epochs with AdamW

**Result: Test AUC = 0.7868** — meaningfully below the tuned/trimmed HistGradientBoosting
result (0.8022), likely limited by the small feature set fed to the transformer (only 4 raw
kinematic fields vs. 13 engineered features) and the modest training scale (21,000 events).

## Results Summary

| Approach | Test AUC |
|---|---|
| **HistGradientBoosting, top-13 features (untuned)** | **0.8022** ⭐ best |
| HistGradientBoosting, all 22 features (untuned) | 0.8003 |
| HistGradientBoosting, all 22 features (tuned) | 0.7994 |
| HistGradientBoosting, top-13 features (tuned) | 0.7990 |
| Gradient Boosting | 0.7981 |
| Random Forest | 0.7952 |
| Particle Transformer (raw per-particle input) | 0.7868 |
| LightGBM | 0.7866 |
| SVM | 0.7783 |
| Logistic Regression | 0.7222 |
| KNN | 0.7051 |
| Gaussian Naive Bayes | 0.6755 |
| Decision Tree | 0.5581 |
| *(Particle-level, unaggregated baseline)* | *0.518 (≈ random)* |

## Key Takeaways

1. **Data quality matters as much as modeling.** A silent division-by-zero bug (`dxysig`
   from a zero `error_dz`) and duplicate rows were found and removed before any modeling.
2. **Framing beats brute-force tuning.** Reframing the problem from particle-level to
   event-level aggregation improved AUC from ~0.52 to ~0.80 — far more than any amount of
   hyperparameter search on either representation.
3. **Tree ensembles were the strongest classical approach** on the engineered feature set,
   clustering tightly around 0.78–0.80 regardless of the specific algorithm.
4. **A single feature (`dxysig_std`) dominated feature importance**, plausibly acting as a
   coarse b-tagging proxy — a promising direction for further feature engineering.
5. **The Transformer under-performed the best classical model here**, likely due to its
   limited raw feature set (4 fields) and modest dataset scale (21K events) — with more
   events, more per-particle features, and longer training, this is the architecture most
   likely to eventually surpass the classical ceiling.

## Requirements

```
awkward
pandas
numpy
pyarrow
matplotlib
seaborn
scikit-learn
scipy
lightgbm
torch
```

## Notes on Reproducing

- `seed = 0` is set for NumPy and PyTorch throughout.
- `N_EVENTS_PER_SAMPLE = 3000` and `N_PARTICLES_MAX = 16` are deliberately small for fast
  iteration; increasing both (using more of the available fragments/events) is the most
  direct way to test whether the Transformer's underperformance is a data-scale issue.
- Data is expected at `hack-data/C1_HH4b/train/<sample_name>/<sample_name>_*.parquet`.
