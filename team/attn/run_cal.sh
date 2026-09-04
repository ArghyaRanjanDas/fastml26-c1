#!/usr/bin/env bash
# A quick low-EBOPs checkpoint whose only job is to give the synthesis box a
# LUT-per-EBOP calibration point while the long train1M QAT runs.
cd "$(dirname "$0")"
P="$HOME/hlsenv/bin/python"
pids=()
for b in 3e-6 1e-5; do
  KERAS_BACKEND=torch PYTHONUNBUFFERED=1 $P train_attn.py --tag "cal_b$b" --quantized \
     --init-from a_d16 --beta0 "$b" --beta-ramp 3 --train-tag train300k --epochs 14 \
     --lr 1e-3 > "logs_cal_b$b.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "CAL DONE"
for b in 3e-6 1e-5; do tail -n 6 "logs_cal_b$b.log"; done
