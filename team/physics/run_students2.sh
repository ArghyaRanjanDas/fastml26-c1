#!/usr/bin/env bash
# Follow-up: the rich per-candidate channels cost +24% phi MACs, and the FPGA
# lane has the baseline at 91% of one SLR. So the real question is not "do they
# help at equal architecture" (they do, +0.037 vs tt) but "do they still help at
# equal *cost*". Both rows below are CHEAPER than the 12,800-MAC baseline.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
common=(--model deepset_plus --rho 32,16 --dropout 0 --pool-norm --event-scale 0.2
        --epochs 25 --seed 0 --train-tag train300k_rich --eval-tag eval100k_rich)

# phi 24-12-8 on 11 channels = 16 x 648 = 10,368 MACs  (baseline: 12,800)
python train.py "${common[@]}" --phi 24,12,8 --tag c2_rich_narrow
# phi 32-16-8 on 11 channels but only the leading 8 candidates = 7,936 MACs
python train.py "${common[@]}" --phi 32,16,8 --n-particles-use 8 --tag c2_rich_8p
