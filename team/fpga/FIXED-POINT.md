# Fixed-point range analysis for `model_2041` (B1e_16p_1M)

Tools (both pure numpy + torch — they run in the training pod, no hls4ml or Vitis needed):

* `team/diagnose_range.py` — per-tensor max|weight|, max|bias|, max|pre-activation| on a sample.
* `team/quantsim.py` — simulates `ap_fixed<W,I>` through an export and reports the AUC.
  hls4ml's default is AP_TRN/**AP_WRAP**, so an out-of-range value wraps sign rather than
  clipping. `--overflow sat` switches to saturation, `--per-layer` gives each tensor its own
  integer width (what `granularity="name"` already allows in `synth.py`).

`quantsim.py` reproduces the measured Vitis closure numbers, so it can be used to iterate
locally instead of occupying the synthesis box:

| precision | Vitis (measured) | `quantsim.py` |
|---|---|---|
| `ap_fixed<16,6>` | 0.807 | 0.80973 |
| `ap_fixed<18,8>` | 0.871 | 0.88132 |
| `ap_fixed<22,10>` | 0.883 | 0.88409 |
| `ap_fixed<28,12>` | 0.885 | 0.88468 |

## What actually overflows

Float AUC 0.88471. `ap_fixed<16,6>` is ±32:

| tensor | max\|·\| | fits ±32? |
|---|---|---|
| input particles | 1.00 | ok |
| input event features | 5.00 | ok |
| phi0 weight / preact | 15.91 / 15.98 | ok |
| phi1 preact | **34.19** | **OVER** |
| phi2 preact | **114.64** | **OVER** |
| pooled / concat | 1.40 / 5.00 | ok |
| **rho0 weight** | **184.01** | **OVER** |
| rho1, out | ≤ 10.76 | ok |

**The event features were already standardized** (zero mean, unit variance, constants frozen
in `data.py:EVENT_STANDARDIZE` and written into every export json) — they are not the
problem, and the normalization has not been changed. Measured on the export: each of the 11
sits at mean ≈ 0.0, std ≈ 0.99, clipped to ±5; after the ×0.2 that the folded `rho0` weights
apply, max|f| = 1.00.

The two real sources are:

1. **`rho0 weight` = 184** — my own BatchNorm fold. The BN scale `γ/√(var+ε)` explodes for
   pooled channels whose variance is near zero. Those channels are nearly dead, so the
   *product* weight×activation stays small and float accuracy is unaffected — but the weight
   itself cannot be represented.
2. **`phi2 preact` = 115** — the φ activations grow independently of the fold.

## What was tried

| approach | `<16,6>` AUC | note |
|---|---|---|
| as exported, global type (AP_WRAP) | 0.80973 | the reported failure |
| as exported, saturating instead of wrapping | **0.87906** | most of the loss is *wrap*, not precision |
| per-layer integer widths, 16-bit total | **0.87364** | `granularity="name"`; −0.011 vs float |
| cross-layer equalization, then global type | 0.62426 | **worse** — see below |

**Cross-layer equalization did not work here** and is off by default (`export.py --equalize`
to opt in). It is function-preserving (verified to 4e-06, AUC unchanged) and it does fix the
range — max|W| 184 → 8.97, max|preact| 115 → 8.97, everything inside ±16. But it fixes range
by *moving* magnitude, and the magnitude lands as underflow: `rho0`'s median weight drops from
0.108 to 0.0038 and **25.5% of its weights fall below the `<16,6>` quantization step**, i.e.
truncate to zero. Balancing the two ends of a 20-bit dynamic range into a 15-bit format is not
possible by rescaling alone.

One trap worth recording: the first implementation bounded *post*-ReLU activations and looked
like it had succeeded, while `phi2` still sat at −33.5. A large negative pre-activation is
invisible after ReLU but still has to fit the accumulator, and under AP_WRAP it wraps to a
large *positive* value. `equalize.py` now bounds pre-activation magnitude.

## Recommendation

* Short term, `<22,10>` is the validated deployable point (319k LUT, 1,724 DSP, 0.39 µs).
* If a global type must stay 16-bit, **per-layer integer widths** recover 0.874 for free.
* The real fix is **QAT**: post-training rescaling cannot fix a dynamic range the training
  never constrained. Training with quantization in the loop constrains it directly, which is
  the next step. The QAT target is a 16-bit-or-narrower datapath, to restore the ~9% margin
  that `<22,10>` leaves on LUT and DSP.
