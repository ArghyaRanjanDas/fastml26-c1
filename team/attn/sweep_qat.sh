#!/usr/bin/env bash
# Step 4: HGQ2 quantization-aware training, warm-started from the float run.
# beta0 is the EBOPs penalty; it is what sets the bit widths, and through them
# the LUT bill. Ramped from 0 over the first 5 epochs so the warm start is not
# destroyed before the weights have adapted.
set -u
cd "$(dirname "$0")"
P="$HOME/hlsenv/bin/python"
FLOAT="${1:-a_d16}"
for b in 3e-6 1e-5 3e-5 1e-4; do
  tag="q_${FLOAT#a_}_b${b}"
  KERAS_BACKEND=torch $P train_attn.py --tag "$tag" --quantized --init-from "$FLOAT" \
      --beta0 "$b" --beta-ramp 5 --train-tag train1M --epochs 20 --lr 1e-3 \
      > "logs_$tag.log" 2>&1
done
