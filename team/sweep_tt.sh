#!/usr/bin/env bash
# Screening on train300k_s (fast), targeting the tt weakness. Winners get
# promoted to train1M_s. Per-background AUC is reported for all of them so the
# price paid on QCD/W is visible, not just the tt gain.
set -e
cd /home/jovyan/fastml26-hackathon/team
d () { local tag=$1; shift
  echo "##### $tag $*"
  python distill.py --soft-targets teacher --tag "$tag" --temperature 2 --alpha 0.5 \
    --phi 32,16,8 --rho 32,16 --pool meanmax --gpu-batches --epochs 30 \
    --train-tag train300k_s --eval-tag eval100k_s "$@" 2>&1 \
    | grep -E "DISTILLED AUC|vs QCD|vs tt|vs Wjets"; }
d tt_base
d tt_w2  --tt-weight 2
d tt_w3  --tt-weight 3
d tt_dis --disagree-weight
d tt_w2_dis --tt-weight 2 --disagree-weight
