#!/usr/bin/env bash
# Student-side tests of the teacher's derived quantities, all at the B1e_16p
# architecture (phi 32-16-8, rho 32-16, pool BN, event scale 0.2, 25 epochs,
# seed 0, train300k) so the only thing that changes is the inputs.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
common=(--model deepset_plus --phi 32,16,8 --rho 32,16 --dropout 0
        --pool-norm --event-scale 0.2 --epochs 25 --seed 0)

# 6 derived channels per candidate -> phi sees 11 inputs instead of 5
python train.py "${common[@]}" --train-tag train300k_rich --eval-tag eval100k_rich --tag c2_rich
# the same, plus max-pooling alongside mean (comparators, no DSP)
python train.py "${common[@]}" --pool meanmax --train-tag train300k_rich --eval-tag eval100k_rich \
       --tag c2_rich_mm
# ParT pair quantities for the 6 pairs among the leading 4 candidates, as 24
# event-level scalars after the pool
python train.py "${common[@]}" --train-tag train300k_p4 --eval-tag eval100k_p4 --tag c2_pair4
# max-pooling alone, to separate it from the derived channels
python train.py "${common[@]}" --pool meanmax --tag c2_meanmax
