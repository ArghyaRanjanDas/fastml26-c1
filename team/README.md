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
