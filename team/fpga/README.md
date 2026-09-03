# FPGA feasibility path

`synth.py` rebuilds the DeepSet exactly in Keras (Conv1D kernel-1 = per-particle φ,
GlobalAveragePooling1D, optional event-feature Concatenate, Dense ρ), converts with
hls4ml (Vitis backend, io_parallel, ap_fixed<16,6> default) and runs Vitis HLS csynth
for `xcu200-fsgd2104-2-e` at 200 MHz. Output: `hls_<tag>/summary.json` with LUT/FF/DSP/BRAM
and best/worst latency (cycles) — the numbers for the Friday slide.

Runs INSIDE the hackathon pod (the GPU image ships Xilinx tools; hls4ml 1.3 lives in `~/hlsenv`):
```
source /tools/Xilinx/Vivado/2023.1/settings64.sh   # per hackathon README
export XILINX_VITIS=/tools/Xilinx/Vitis/2024.2
~/hlsenv/bin/python team/fpga/synth.py --export team/export/model_9k.json --weights team/export/model_9k.pt --tag m9k
```
Export contract from training (`team/export/`): `model_<params>.json` with keys
`phi`, `rho` (layer widths), `n_features`, `n_particles`, `n_event_features`,
plus `model_<params>.pt` state_dict (Linear layers in phi..rho order) and
`eval_sample.npz` (inputs, labels, scores) for closure checks.
Budget: one SLR ≈ 350k LUT, 700k FF, 1,900 DSP, 25.3 MB BRAM; latency ≤ 1 µs = 200 cycles @200 MHz.
