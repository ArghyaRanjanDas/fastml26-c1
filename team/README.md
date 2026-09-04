# Team fastml26-c1 — Challenge 1 (HH→4b trigger classifier)

## Results summary (2026-09-03 21:05 PDT; everything below is measured, see `RESULTS.md`, `fpga/RESULTS-fpga.md`)

**Headline: a 2,745-parameter DeepSet student, float AUC 0.909, synthesized on a Xilinx VU9P at HLS AUC 0.906,
253k LUT / 1,692 DSP / 0.42 µs, inside one SLR.** (Morning baseline: 0.883 at 319k LUT / 1,724 DSP.)

| what | model | params | AUC (even-thirds eval) | vs QCD | vs tt | vs W+jets | FPGA |
|---|---|---|---|---|---|---|---|
| baseline student | DeepSet φ32-16-8, mean pool, ρ32-16, 16×5 + 11 event feats | 2,041 | 0.8869 | 0.930 | 0.759 | 0.972 | HLS 0.8815, 219k LUT, 1,411 DSP, 0.39 µs |
| **rich student** | same net over 11 particle channels, mean+max, 19 event feats, distilled | 2,745 | **0.9090** | 0.935 | 0.809 | 0.974 | **HLS 0.9062, 253k LUT, 1,692 DSP, 0.42 µs** |
| rich student, tt-weighted KD | same, tt ×2 + disagreement weighting | 2,745 | 0.9086 | 0.928 | 0.814 | 0.967 | HLS 0.9048, 253k LUT, 1,651 DSP |
| attention student (float) | HGQ2 transformer, 16 tokens, width 16, 2 heads | 3,073 | 0.9086 | 0.940 | 0.814 | 0.973 | quantized synthesis pending |
| attention student (float) | same, 2 blocks | 5,233 | 0.9114 | 0.942 | 0.817 | 0.975 | pending |
| best teacher | ParT-lite ensemble (4 seeds + DeepSet), not FPGA-bound | 5.4M | 0.9247 | 0.947 | 0.850 | 0.977 | n/a |
| four-class teacher | ParT-lite with HH/QCD/tt/W head | 1.33M | 0.9246 | 0.947 | 0.851 | 0.976 | n/a |

Key findings so far:
- **Inputs beat teachers.** Six derived per-candidate channels (+ iso, ln ΔR of leading-4 pairs) gave +0.022 AUC and +0.05 on tt; a 0.01-better teacher moved the student by 0.0003 (capacity-limited at 2.7k params).
- **Fixed point:** 16-bit closes once overflow is handled — wide wrap accumulators + saturation at the layer-output cast (saturating everything costs 2.6× the LUTs). Fixed-point loss ≤ 0.006 on any background.
- **tt is the whole game:** every model separates W+jets at 0.97 and QCD at 0.93–0.95; the student–teacher gap is entirely tt (0.81 vs 0.85) and is relational (pairwise) information.
- **The official eval mixture is QCD 9 % / W+jets 36 % / tt 55 %** (not even thirds), so pooled AUC is tt-dominated: on that mixture the FPGA students score 0.878 (rich) and 0.880 (tt-weighted).
- Distillation: KL(T=2)+BCE, α 0.5–0.7; disagreement weighting is a free +0.001; tt up-weighting trades QCD for tt.

Lanes: `c1` student training/QAT (pod A10), `c2` physics & features (CPU), `c3` attention student (pod A10), `hh4b` teachers + A100 job queue (`A100_QUEUE.md`), orchestrator = Vitis synthesis on a VU9P (xcu200) via hls4ml 1.3 + Vitis HLS 2024.1. Idea scan: `IDEAS.md`. Feature firmware costs: `fpga/FEATURES.md`.

---

# Team workflow — FastML26 Challenge 1 (HH→4b vs QCD, trigger level)

Everything is done by Claude Code agents; humans decide, agents execute.

## The scoreboard
`team/RESULTS.md` is the single source of truth and Friday's slide. Append one row
per experiment; never edit others' rows.

| model | params | AUC (eval) | train events | quant | LUT | FF | DSP | BRAM | latency | notes |

## Conventions
- One branch per person (`<name>/...`), merge to `main` via PR when a row lands in RESULTS.md.
- Scripts live in `team/`: `data.py` (streaming parquet loader, capped events), `models.py`,
  `train.py`. Reuse them; don't fork the loader.
- Never load the full 132 GB. `data.py` caps events per process.
- Commit + push after every milestone: pods restart.

## The two halves of the score
1. **AUC** of the network score, HH_4b vs all backgrounds (QCD, tt, W+jets).
2. **FPGA feasibility**: hls4ml/Vitis HLS estimate for Xilinx VU9P (`xcu200-fsgd2104-2-e`),
   latency ≤ 1 µs, fits ONE SLR (~350k LUT, 700k FF, 1,900 DSP, 25.3 MB BRAM).
   Synthesis runs on Vitis HLS 2024.1 (COS-6PRIME) from an exported model; the pod
   does training + quantization-aware training (QKeras) in `~/hlsenv`.

## Division of labour (proposed)
- physics features: pairwise masses / jet-like clustering of the 16 PUPPI candidates, dxy as a b-proxy
- model-size sweep: DeepSet & MLP widths, AUC-vs-params curve
- quantization + synthesis: QKeras QAT, hls4ml config, Vitis report
