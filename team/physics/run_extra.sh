#!/usr/bin/env bash
# The tt-feature deliverable: build the caches with EXTRA_FEATURES on, then train
# B1e_16p's exact architecture with and without them, on CPU, same seed.
#
# The published B1e_16p numbers were trained on the GPU; retraining the control
# here means the comparison is CPU-vs-CPU, same machine, same seed, and the only
# difference between the two rows is the three extra columns in F.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""

N_SIG=${N_SIG:-300000}
N_BKG=${N_BKG:-300000}
TRAIN_TAG=${TRAIN_TAG:-train300k_x}
EVAL_TAG=${EVAL_TAG:-eval100k_x}
EPOCHS=${EPOCHS:-25}

echo "### building extra-feature caches"
python data.py --tag "$TRAIN_TAG" --split train --n-signal "$N_SIG" --n-background "$N_BKG" \
       --extra-features
python data.py --tag "$EVAL_TAG" --split eval --n-signal 100000 --n-background 100000 \
       --extra-features

common=(--model deepset_plus --phi 32,16,8 --rho 32,16 --dropout 0
        --pool-norm --event-scale 0.2 --epochs "$EPOCHS" --seed 0)

echo "### control: 11 event features (B1e_16p architecture, CPU)"
python train.py "${common[@]}" --train-tag train300k --eval-tag eval100k --tag c2_base_cpu

echo "### with the tt features"
python train.py "${common[@]}" --train-tag "$TRAIN_TAG" --eval-tag "$EVAL_TAG" --tag c2_ttfeat
