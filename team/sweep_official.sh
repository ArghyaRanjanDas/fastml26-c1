#!/usr/bin/env bash
# Selection metric is now the organizers' mixture (0.0912 QCD + 0.3638 W + 0.5450 tt),
# which is 55% tt. Re-check tt weighting under it, and try training AT that mixture.
set -e
cd /home/jovyan/fastml26-hackathon/team
d () { local tag=$1; shift
  echo "##### $tag $*"
  python distill.py --soft-targets teacher --tag "$tag" --temperature 2 --alpha 0.5 \
    --phi 32,16,8 --rho 32,16 --pool meanmax --gpu-batches --epochs 30 \
    --train-tag train300k_s --eval-tag eval100k_s "$@" 2>&1 \
    | grep -E "OFFICIAL|vs QCD|vs tt|vs Wjets"; }
d of_mix           --mixture official
d of_mix_dis       --mixture official --disagree-weight
d of_mix_w2_dis    --mixture official --tt-weight 2 --disagree-weight
d of_w4_dis        --tt-weight 4 --disagree-weight
d of_w6_dis        --tt-weight 6 --disagree-weight
