#!/usr/bin/env bash
# Distill teacher_1M (71,905 params, mean+max pool, AUC 0.89950) into the
# deployable B1e_16p shape (phi 32-16-8, rho 32-16, 16 particles, 2,057 params).
# Baseline to beat: 0.88687, the same shape trained from scratch on the same 2M events.
set -e
cd "$(dirname "$0")"
d () { local tag=$1 T=$2 a=$3
  echo "##### $tag  T=$T alpha=$a"
  python distill.py --teacher teacher_1M --tag "$tag" --temperature "$T" --alpha "$a" \
    --epochs 30 --train-tag train1M 2>&1 | grep -E "DISTILLED AUC|vs tt"; }
d kd_T2_a07 2 0.7
d kd_T3_a07 3 0.7
d kd_T4_a07 4 0.7
d kd_T3_a09 3 0.9
d kd_T3_a10 3 1.0
