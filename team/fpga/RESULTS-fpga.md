# FPGA synthesis results (Vitis HLS 2024.1, xcu200-fsgd2104-2-e, 200 MHz, io_parallel)

Budget (one SLR): ~350k LUT · 700k FF · 1,900 DSP · 25.3 MB BRAM · latency ≤ 1 µs (200 cycles)

| tag | model | params | precision | reuse | LUT | FF | DSP | BRAM | latency (cycles) | fits? |
|---|---|---|---|---|---|---|---|---|---|---|
| dummy10k | DeepSet φ64-32-16 ×16p, +8 evt, ρ64-32 (random weights) | 6,705 | ap_fixed<16,6> | 1 | 524,961 | 374,954 | 4,836 | 1 | 71–74 (0.37 µs) | ❌ LUT 1.5×, DSP 2.5× |
| r4_16p | same shape, reuse 4 (random weights) | 6,705 | ap_fixed<16,6> | 4 | 538,341 | 397,548 | 1,624 | 1 | 219–222 (~1.1 µs @5 ns; 0.8 µs @ est. 3.6 ns) | ❌ LUT 1.5× (DSP now fits) |
| r8_16p | same shape, reuse 8 | 6,705 | ap_fixed<16,6> | 8 | 530,001 | 409,538 | 812 | 1 | 426–429 (~2.1 µs) | ❌ LUT, latency |
| small_r1 | **B1e shape**: φ32-16-8 ×16p, +8 evt, ρ32-16 (random w.) | 1,945 | ap_fixed<16,6> | 1 | **221,920** | 187,549 | **1,503** | 1 | **69–72 (0.36 µs)** | ✅ **fits** |
| small_r4 | same, reuse 4 | 1,945 | ap_fixed<16,6> | 4 | 224,307 | 190,598 | 460 | 1 | 219–222 (~0.8–1.1 µs) | ⚠️ latency borderline |
| small_p10 | same, 10-bit | 1,945 | ap_fixed<10,4> | 1 | 209,470 | 118,537 | **0** | 1 | 69–72 (0.36 µs) | ✅ fits, DSP-free |
| small_8p | same, 8 particles | 1,945 | ap_fixed<16,6> | 1 | 139,746 | 106,335 | 1,489 | 1 | 44–47 (0.23 µs) | ✅ fits |

Lesson: φ is replicated once per particle (16×) — that is the whole DSP/LUT bill.
Levers: reuse factor on φ, narrower φ, fewer particles, lower precision / QAT.
**Conclusion of the sweep:** the deployable shape (φ 32-16-8 on 16 particles, ρ 32-16 — exactly c1's B1e_16p) **fits one SLR fully parallel at 16-bit**: ~222k LUT, 1.5k DSP, 0.36 µs. Reuse buys DSPs but not LUTs and costs latency; going to 10-bit removes the DSPs entirely; 8 particles halves LUTs. Real-weight synthesis of the exported models follows in the next rows.

## Real weights (c1's exports, VU9P, 200 MHz, io_parallel, reuse 1)

| export | run | AUC (float) | params | precision | LUT | FF | DSP | latency | fits? | closure (keras vs HLS) |
|---|---|---|---|---|---|---|---|---|---|---|
| model_2041 | B1e_16p_1M | 0.88687 | 2,041 | ap_fixed<16,6> | **228,151** | 182,308 | **1,466** | **71–74 cyc (0.36 µs)** | ✅ | ❌ max diff 0.999 on random inputs — precision not yet tuned |
| model_2041_8p | C1e_8p_1M | 0.88113 | 2,041 | ap_fixed<16,6> | 145,659 | 101,117 | 1,487 | 46–49 cyc (0.23 µs) | ✅ | ❌ max diff 0.21 |
| model_3585 | A3e_3k | 0.88618 | 3,585 | ap_fixed<16,6> | 436,757 | 353,912 | 2,425 | 69–72 cyc | ❌ LUT, DSP | — |

**Resources: the primary model fits with ~35% LUT headroom and 23% DSP headroom.** The closure column is
the open item: with a single global ap_fixed<16,6> the HLS model does not reproduce the float scores, which
means the synthesized AUC cannot be quoted yet. Next: closure check on `export/eval_sample.npz` (real inputs),
per-layer precision from hls4ml profiling, then re-synthesize at the minimal precision that closes.

## Closure vs fixed-point precision — model_2041 (B1e_16p_1M), real inputs (`export/eval_sample.npz`, 5,000 events)

| precision (global) | max\|Δ\| keras–HLS | mean\|Δ\| | AUC keras | **AUC HLS** |
|---|---|---|---|---|
| ap_fixed<16,6>  | 0.994 | 0.161 | 0.8847 | 0.8066 ❌ |
| ap_fixed<18,8>  | 0.534 | 0.118 | 0.8847 | 0.8708 |
| ap_fixed<22,10> | 0.224 | 0.028 | 0.8847 | **0.8830** |
| ap_fixed<28,12> | 0.016 | 0.002 | 0.8847 | 0.8846 ✅ |

Diagnosis: the loss is **integer-range overflow** (6 → 10 integer bits recovers it), not rounding. Fix on the
training side: standardize the event features and bound activations to ~±8 before export (asked of c1);
then 16-bit (or QAT ≤8-bit) closes. Meanwhile the primary model is being synthesized at <22,10> and <20,10>
to price the wider datapath.

## Primary model at a precision that closes — **the slide number**

| precision | AUC HLS (sample) | LUT | FF | DSP | latency | fits one SLR? |
|---|---|---|---|---|---|---|
| ap_fixed<20,10> | 0.8708 | 268,087 | 223,809 | 1,468 | 72–75 cyc (0.37 µs) | ✅ but AUC loss −0.014 |
| **ap_fixed<22,10>** | **0.8830** (float 0.8847) | **319,251** | 311,532 | **1,724** | **75–78 cyc (0.39 µs)** | ✅ **fits, closes** |

**B1e_16p_1M (2,041 params) synthesized at ap_fixed<22,10>: AUC 0.883, 319k LUT / 1.7k DSP / 0.39 µs — inside one VU9P SLR
(91% of the LUT budget, 91% of DSP) and within 1 µs by 2.5×.** Per-run Vitis summaries are in `reports/`.
Headroom is thin at this precision; the path to a comfortable margin is bounding the inputs/activations
(asked of c1) so the datapath can drop back toward 16-bit, and then QAT.

## Saturation instead of wrap — model_2041, real Vitis (2026-09-03 afternoon)

c1's quantsim diagnosis (commit 7291b28) said the 16-bit loss is overflow *wrapping*, not precision.
Confirmed on real Vitis HLS: with `AP_SAT` the 16-bit design closes. But saturation applied to **every**
type (weights, accumulators, results) is expensive in logic:

| precision (global, all types) | AUC HLS (sample) | keras | LUT | FF | DSP | latency | fits one SLR? |
|---|---|---|---|---|---|---|---|
| ap_fixed<22,10> (wrap, reference) | 0.8830 | 0.8847 | 319,251 | — | 1,724 | 75–78 cyc (0.39 µs) | ✅ 91 % |
| ap_fixed<16,6,AP_RND,AP_SAT> | 0.8809 | 0.8847 | **566,173** | 216,503 | 1,407 | 113–116 cyc (0.58 µs) | ❌ LUT 162 % |
| ap_fixed<18,8,AP_RND,AP_SAT> | 0.8829 | 0.8847 | 628,259 | 262,193 | 1,476 | 114–117 cyc (0.59 µs) | ❌ |
| ap_fixed<16,8,AP_RND,AP_SAT> (closure only) | 0.8672 | 0.8847 | | | | | fractional bits matter more than headroom |
| ap_fixed<16,6,AP_TRN,AP_SAT> (closure only) | 0.8691 | 0.8847 | | | | | rounding matters (+0.012 for AP_RND) |

Take-aways: (1) saturation on every multiply-accumulate costs ~250k LUT and 38 cycles — it is not free;
(2) the cheap variant is a wide *wrap* accumulator (`ap_fixed<24,12>`) with saturation only at the
16-bit layer-output cast, plus per-layer integer bits for weights (rho0 needs 9) — being synthesized as
`tsat_16_6` (synth.py now takes `--accum`, `--result`, `--auto-weights N`); (3) the real fix is to bound
the pre-activations in training (QAT / activation clipping) so plain wrap 16-bit closes with no saturation.
Summaries: `reports/sum_sat_*.json`.

## Targeted saturation — **new best fitting design** (model_2041, real Vitis, 2026-09-03 16:40 PDT)

Wide *wrap* accumulators, saturation only at the 16-bit layer-output cast, per-layer weight integer bits
from max|W| (`synth.py --accum 'ap_fixed<24,12>' --result 'ap_fixed<16,6,AP_RND,AP_SAT>' --auto-weights 16`):

| design | AUC HLS (sample) | LUT | FF | DSP | latency | one SLR (350k LUT / 1,900 DSP)? |
|---|---|---|---|---|---|---|
| ap_fixed<22,10> wrap (previous headline) | 0.8830 | 319,251 | — | 1,724 | 75–78 cyc (0.39 µs) | ✅ 91 % LUT |
| **tsat_16_6**: 16-bit data, accum <24,12> wrap, output SAT, auto weight ints | **0.8815** | **219,316** | 149,434 | 1,411 | 76–79 cyc (0.39 µs) | ✅ **63 % LUT, 74 % DSP** |
| twrap_16_6_a24: same but no saturation anywhere (closure only) | 0.8794 | | | | | wide accumulators alone recover almost everything |
| tsat_16_6_w18: same as tsat with 18-bit weights (closure only) | 0.8828 | | | | | +0.001 for 2 more weight bits |

So: −31 % LUT and −18 % DSP versus the 22-bit design at −0.0015 AUC. The headroom (~130k LUT, ~500 DSP)
is what pays for c2's six extra per-candidate channels (+24 % φ MACs) in the next student.
Summary: `reports/sum_tsat_16_6.json`. Distributed-arithmetic (da4ml) variants of both designs are synthesizing.

## Rich-feature student on the FPGA — **new headline** (model_2777_rich = `rkd_T2_a05`, 2026-09-03 17:45 PDT)

c1's student with c2's six derived per-candidate channels (16×11 particle tensor), mean+max pooling and
19 event features, distilled from the teacher (float eval AUC 0.909, vs tt 0.809). Same Vitis flow
(VU9P xcu200, 200 MHz, io_parallel, reuse 1); closure on `export/eval_sample_rich.npz` (5,000 events).

| design | AUC keras (sample) | **AUC HLS** | LUT | FF | DSP | latency | one SLR (350k LUT / 1,900 DSP)? |
|---|---|---|---|---|---|---|---|
| **rich, `tsat_16_6`** (16-bit data, accum <24,12> wrap, output SAT, auto weight ints) | 0.9077 | **0.9062** | **253,154** | 176,866 | **1,692** | 81–84 cyc (**0.42 µs**) | ✅ **72 % LUT, 89 % DSP** |
| rich, ap_fixed<22,10> wrap | 0.9077 | 0.9069 | 369,990 | 353,572 | 2,322 | 82–85 cyc | ❌ LUT 106 %, DSP 122 % |
| baseline `tsat_16_6` (model_2041), for reference | 0.8847 | 0.8815 | 219,316 | 149,434 | 1,411 | 76–79 cyc | ✅ 63 % / 74 % |

So the six extra channels cost +34k LUT and +281 DSP and buy **+0.025 AUC on the FPGA** (0.881 → 0.906).
DSP is now the binding resource (89 %); the distributed-arithmetic (da4ml) variant and QAT/pruning are the
levers for the next student. The derived channels are computed once per candidate in a preprocessing
block (formulas and firmware cost classes in `team/fpga/FEATURES.md`); that block is **not** included in
these numbers yet. Summaries: `reports/sum_rich_*.json`.

## Per-background AUC of the synthesized designs (closure sample, 5,000 events: 2,486 HH / 850 QCD / 808 tt / 856 W+jets)

Background labels recovered by matching the sample rows to `team/cache/eval100k` (100 % matched); `synth.py`
prints these whenever the sample carries a `group` array (0 QCD, 1 HH, 2 tt, 3 W+jets).

| design | AUC all (keras → HLS) | vs QCD (keras → HLS) | vs tt (keras → HLS) | vs W+jets (keras → HLS) |
|---|---|---|---|---|
| model_2041 @ ap_fixed<22,10> wrap | 0.8847 → 0.8830 | 0.9267 → 0.9242 | 0.7499 → 0.7521 | 0.9702 → 0.9658 |
| model_2041 @ `tsat_16_6` | 0.8847 → 0.8815 | 0.9267 → 0.9207 | 0.7499 → 0.7533 | 0.9702 → 0.9635 |
| **model_2777_rich @ `tsat_16_6`** | 0.9077 → **0.9062** | 0.9349 → **0.9356** | 0.8085 → **0.8074** | 0.9743 → **0.9703** |

Fixed point costs ≤ 0.006 on any background; the tt gain of the rich student (+0.055) survives quantization intact.

## tt-weighted rich student, and the metric that actually counts (2026-09-03 21:05 PDT)

`model_2777_rich_tt` = `rich_1M_w2dis` (tt ×2 + disagreement-weighted KD, train1M; float eval 0.9086), same `tsat_16_6` flow:

| design | AUC keras → HLS (sample) | vs QCD | vs tt | vs W+jets | LUT | FF | DSP | latency |
|---|---|---|---|---|---|---|---|---|
| rich `tsat_16_6` (plain KD) | 0.9077 → 0.9062 | 0.9356 | 0.8074 | 0.9703 | 253,154 | 176,866 | 1,692 | 81–84 cyc |
| **rich_tt `tsat_16_6`** (tt ×2 + disagreement) | 0.9073 → 0.9048 | 0.9281 | **0.8141** | 0.9674 | 252,894 | 176,373 | 1,651 | 81–84 cyc |

On our even-thirds eval slice the tt-weighted student is −0.0014. **But the organizers' eval parquet is not even thirds.**
Row count of `C1_HH4b/eval` (2,102,226 rows): HH 1,000,384; QCD 100,482; W+jets 400,812 (WJetsToLNu 200,281 + WJetsToQQ 200,531);
tt 600,548 (hadronic 200,327 / semi-leptonic 200,110 / leptonic 200,111). Background fractions: **QCD 9 %, W+jets 36 %, tt 55 %.**
A pooled "HH vs all backgrounds" AUC is the background-fraction-weighted mean of the per-background AUCs, so on the official mixture:

| design | official-mixture AUC (0.09·QCD + 0.36·W + 0.55·tt) |
|---|---|
| model_2041 `tsat_16_6` | 0.09·0.9207 + 0.36·0.9635 + 0.55·0.7533 = **0.844** |
| rich `tsat_16_6` | 0.09·0.9356 + 0.36·0.9703 + 0.55·0.8074 = **0.878** |
| rich_tt `tsat_16_6` | 0.09·0.9281 + 0.36·0.9674 + 0.55·0.8141 = **0.880** |

So on the metric that will be scored, tt is 55 % of the weight, the tt-weighted student is the better one (+0.002), and the
right training/selection mixture is the official one, not even thirds. (Training-set counts: HH 9.0M, QCD 0.90M, W+jets 1.8M+, tt still counting.)

<!-- c3:attention:start -->
## Attention student (c3) — projects awaiting synthesis

Built with `team/attn/synth_attn.py` (hls4ml 1.3.0 + HGQ2, Vitis backend,
`xcu200-fsgd2104-2-e`, 5 ns, **io_parallel**, `Strategy=distributed_arithmetic`,
`ReuseFactor=1`, **no manual precision** — HGQ2 carries a learned bit width per parameter
and per activation, and hls4ml's bit-exact pass puts them in the generated C++).

`io_stream` is not an option for this lane: hls4ml raises `NotImplementedError:
Heterogenous quantization for activations is only supported with IOType=io_parallel`,
and heterogeneous widths are the whole point of HGQ.

**Closure is not an open item here — it is exact.** On the same 5,000-event sample the
DeepSet lane uses (`team/export/eval_sample.npz`, rows matched back to `cache/eval100k`
to recover the background labels, 100 % matched):

| tag | project | EBOPs | max abs diff keras−HLS | AUC official (keras = HLS) | AUC even-3rds | vs QCD | vs tt | vs W+jets |
|---|---|---|---|---|---|---|---|---|
| `s300_b1e-7` | `projects/s300_b1e-7.tar.gz` (0.7 MB) | 1.37M | **0.0** | 0.8710 | 0.9020 | 0.9328 | 0.7967 | 0.9708 |

<sub>The per-background AUCs above are on the 5,000-event closure sample (850 QCD / 808 tt̄ /
856 W+jets), so they are noisier than the `eval100k` numbers in RESULTS.md; on the full
eval slice this checkpoint is official 0.87454 / even-thirds 0.90283.</sub>

**What I need from this one:** it is a deliberately under-regularized calibration point, not
the final design — a short 10-epoch QAT on `train300k` at `beta0=1e-7`. Its purpose is to
measure **LUT (and FF/DSP/latency) per EBOP** for this architecture, so the EBOPs target for
the long runs can be set from a measured number instead of the ~1:1 LUT:EBOPs implied by
arXiv:2510.24784. If it does not fit, that is expected and still the answer I need. The
production runs (`train1M`, 35 epochs, stronger beta, selected on the official mixture) are
training on the A10 and in the A100 queue as job `c3-3`, and will land at a much lower EBOPs
for the same architecture — so **please report the numbers even if it blows the SLR budget.**

Architecture (3,073 synthesized weights): particles (16 × 11 channels) → shared EinsumDense
d=16 relu → 1-head MultiHeadAttention(key_dim 16) + residual → EinsumDense 16→32 relu →32→16
+ residual → mean-pool ‖ max-pool → concat 11 event features → Dense 16 relu → Dense 1.
<!-- c3:attention:end -->
