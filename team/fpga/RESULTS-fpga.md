# FPGA synthesis results (Vitis HLS 2024.1, xcu200-fsgd2104-2-e, 200 MHz, io_parallel)

Budget (one SLR): ~350k LUT · 700k FF · 1,900 DSP · 25.3 MB BRAM · latency ≤ 1 µs (200 cycles)

| tag | model | params | precision | reuse | LUT | FF | DSP | BRAM | latency (cycles) | fits? |
|---|---|---|---|---|---|---|---|---|---|---|
| dummy10k | DeepSet φ64-32-16 ×16p, +8 evt, ρ64-32 (random weights) | 6,705 | ap_fixed<16,6> | 1 | 524,961 | 374,954 | 4,836 | 1 | 71–74 (0.37 µs) | ❌ LUT 1.5×, DSP 2.5× |

Lesson: φ is replicated once per particle (16×) — that is the whole DSP/LUT bill.
Levers: reuse factor on φ, narrower φ, fewer particles, lower precision / QAT.
Sweep in progress (see `sweep.sh`).
