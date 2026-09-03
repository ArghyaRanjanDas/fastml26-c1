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
