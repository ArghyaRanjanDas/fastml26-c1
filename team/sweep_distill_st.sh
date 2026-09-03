#!/usr/bin/env bash
# Distill into the deployable B1e_16p shape against the TEAM soft targets
# (team/teacher/, source_run ds_big_s0, eval AUC 0.91515, tt 0.82612).
# Compare against: 0.88687 from scratch, and the local DeepSet-teacher runs
# (teacher_1M, AUC 0.89950, tt 0.78927) in sweep_distill.sh.
set -e
cd "$(dirname "$0")"
d () { local tag=$1 T=$2 a=$3
  echo "##### $tag  T=$T alpha=$a"
  python distill.py --soft-targets teacher --tag "$tag" --temperature "$T" --alpha "$a" \
    --epochs 30 --train-tag train1M 2>&1 | grep -E "DISTILLED AUC|vs tt"; }
d st_T2_a07 2 0.7
d st_T3_a07 3 0.7
d st_T4_a07 4 0.7
d st_T3_a05 3 0.5
d st_T3_a09 3 0.9
