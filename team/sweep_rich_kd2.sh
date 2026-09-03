#!/usr/bin/env bash
# alpha 0.3-0.5 at T=2 (0.5 beat 0.7 and 0.9), and longer training.
set -e
cd "$(dirname "$0")"
d () { local tag=$1 T=$2 a=$3 e=$4
  echo "##### $tag T=$T alpha=$a epochs=$e"
  python distill.py --soft-targets teacher --tag "$tag" --temperature "$T" --alpha "$a" \
    --phi 32,16,8 --rho 32,16 --pool meanmax --gpu-batches --epochs "$e" \
    --train-tag train1M_s --eval-tag eval100k_s 2>&1 | grep -E "DISTILLED AUC|vs tt"; }
d rkd_T2_a03    2 0.3 30
d rkd_T2_a04    2 0.4 30
d rkd_T2_a05_e60 2 0.5 60
d rkd_T2_a03_e60 2 0.3 60
