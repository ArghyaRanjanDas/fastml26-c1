#!/usr/bin/env bash
# The canonical student input set: 11 per-candidate channels, 19 event features,
# mean+max pooling. Two sizes -- the same phi as today's baseline, and a narrower
# phi that is CHEAPER than today's baseline despite the extra channels.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
common=(--model deepset_plus --rho 32,16 --dropout 0 --pool-norm --event-scale 0.2
        --pool meanmax --epochs 25 --seed 0
        --train-tag train300k_c2 --eval-tag eval100k_c2)
python train.py "${common[@]}" --phi 32,16,8 --tag c2_canon
python train.py "${common[@]}" --phi 24,12,8 --tag c2_canon_narrow
