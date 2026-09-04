#!/usr/bin/env bash
# Production QAT for the deliverable. beta0=3e-6 is the screen's answer: on train300k it
# settles at ~900k EBOPs by epoch 8, and train1M gives 3.4x the gradient steps per epoch,
# so the same beta should land in the few-hundred-k range that fits one SLR.
cd "$(dirname "$0")"
P="$HOME/hlsenv/bin/python"
SEED_RUN="${1:-a_d16}"
TAG="${2:-q_prod}"
BETA="${3:-3e-6}"
KERAS_BACKEND=torch PYTHONUNBUFFERED=1 $P train_attn.py --tag "$TAG" --quantized \
    --init-from "$SEED_RUN" --beta0 "$BETA" --beta-ramp 8 --train-tag train1M \
    --epochs 35 --lr 1e-3 > "logs_$TAG.log" 2>&1
echo "QAT $TAG DONE"
tail -n 12 "logs_$TAG.log"
