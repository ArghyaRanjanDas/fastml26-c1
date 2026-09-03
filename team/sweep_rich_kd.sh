#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
d () { local tag=$1 T=$2 a=$3
  echo "##### $tag T=$T alpha=$a"
  python distill.py --soft-targets teacher --tag "$tag" --temperature "$T" --alpha "$a" \
    --phi 32,16,8 --rho 32,16 --pool meanmax --gpu-batches --epochs 30 \
    --train-tag train1M_s --eval-tag eval100k_s 2>&1 | grep -E "DISTILLED AUC|vs tt"; }
d rkd_T2_a07 2 0.7
d rkd_T3_a07 3 0.7
d rkd_T2_a05 2 0.5
d rkd_T2_a09 2 0.9
