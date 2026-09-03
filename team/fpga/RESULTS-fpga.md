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
