# FPGA feasibility path

`synth.py` rebuilds the DeepSet exactly in Keras (Conv1D kernel-1 = per-particle φ,
GlobalAveragePooling1D, optional event-feature Concatenate, Dense ρ), converts with
hls4ml (Vitis backend, io_parallel, ap_fixed<16,6> default) and runs Vitis HLS csynth
for `xcu200-fsgd2104-2-e` at 200 MHz. Output: `hls_<tag>/summary.json` with LUT/FF/DSP/BRAM
and best/worst latency (cycles) — the numbers for the Friday slide.

Runs on the hub's **FPGA image** server (profile with FPGA=1, GPUs=0, 4 cores — the only image
that has `/tools/Xilinx`; one FPGA server per team, do not train there). hls4ml 1.3 via `~/hlsenv`:
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

## Fixed-point note (from training side)

The exported event features are standardized (zero mean / unit variance, constants frozen in
`data.py:EVENT_STANDARDIZE` and written into every export json); that normalization is
unchanged. The `ap_fixed<16,6>` closure failure traces to `rho0` weights reaching 184 (the
folded pooled BatchNorm) and phi pre-activations reaching 115 — not to the inputs. Full
analysis, plus `quantsim.py`/`diagnose_range.py` which reproduce the Vitis closure numbers
locally, in `FIXED-POINT.md`.
