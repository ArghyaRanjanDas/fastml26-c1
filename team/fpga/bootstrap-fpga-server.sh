#!/usr/bin/env bash
# Run once on the hub's FPGA-image server (profile FPGA=1, GPUs=0). Dependencies must be
# reinstalled on every new server (hackathon README). Gives: hls4ml + QKeras + TF-CPU in
# ~/hlsenv, the team repo, and the Xilinx env sourced.
set -euo pipefail
cd ~
[ -d fastml26-c1 ] || git clone https://github.com/ArghyaRanjanDas/fastml26-c1.git   # add deploy key / token if private
python3 -m venv --system-site-packages ~/hlsenv
~/hlsenv/bin/pip install -q "hls4ml>=1.3" qkeras tensorflow-cpu onnx qonnx torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || ~/hlsenv/bin/pip install -q "hls4ml>=1.3" qkeras tensorflow-cpu onnx qonnx torch
source /tools/Xilinx/Vivado/2023.1/settings64.sh
export XILINX_VITIS=/tools/Xilinx/Vitis/2024.2
command -v vitis_hls && vitis_hls -version | head -2
echo "ready. synth:  ~/hlsenv/bin/python fastml26-c1/team/fpga/synth.py --export <json> --weights <pt> --tag <tag>"
