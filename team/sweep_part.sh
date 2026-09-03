#!/usr/bin/env bash
# Distill against the ens_part4 teacher (4x ParT ensemble, AUC 0.92480, tt 0.85181),
# which replaced ds_big_s0 (0.91515 / 0.82612) in team/teacher/.
set -e
cd "$(dirname "$0")"
d () { local tag=$1 T=$2 a=$3 e=$4
  echo "##### $tag T=$T alpha=$a epochs=$e"
  python distill.py --soft-targets teacher --tag "$tag" --temperature "$T" --alpha "$a" \
    --phi 32,16,8 --rho 32,16 --pool meanmax --gpu-batches --epochs "$e" \
    --train-tag train1M_s --eval-tag eval100k_s 2>&1 | grep -E "DISTILLED AUC|vs tt"; }
d pkd_T2_a05 2 0.5 30
d pkd_T2_a03 2 0.3 30
d pkd_T2_a07 2 0.7 30
d pkd_T2_a05_e60 2 0.5 60
