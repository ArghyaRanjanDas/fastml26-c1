#!/usr/bin/env bash
# Canonical input set at 8 candidates: phi 32-16-8 on 11 channels x 8 = 7,936 MACs,
# 38% cheaper than today's 12,800-MAC baseline.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
python train.py --model deepset_plus --phi 32,16,8 --rho 32,16 --dropout 0 --pool-norm \
  --event-scale 0.2 --pool meanmax --epochs 25 --seed 0 --n-particles-use 8 \
  --train-tag train300k_c2 --eval-tag eval100k_c2 --tag c2_canon_8p
