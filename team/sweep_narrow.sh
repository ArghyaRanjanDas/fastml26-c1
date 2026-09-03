#!/usr/bin/env bash
# Follow-up sweep driven by the Vitis results: LUT is the binding constraint and
# phi is replicated once per particle, so the levers are narrower phi and fewer
# particles.  Event features are on for most of these because the first sweep
# showed they are worth +0.02 AUC once the particle branch is shrunk -- and they
# cost nothing per particle, being computed once per event.
set -e
cd "$(dirname "$0")"
run () { local tag=$1 phi=$2 rho=$3 np=$4; shift 4
  echo "##### $tag phi=$phi rho=$rho P=$np $*"
  python train.py --model deepset_plus --phi "$phi" --rho "$rho" --dropout 0 --pool-norm \
    --epochs 25 --n-particles-use "$np" --tag "$tag" "$@" 2>&1 | grep -E "trainable params|eval slice\) ="; }
run D1_8p   24,12,8  32,16   8 --no-event-features
run D1e_8p  24,12,8  32,16   8 --event-scale 0.2
run D2e_16p 24,12,8  32,16  16 --event-scale 0.2
run C3e_8p  32,16,8  64,32   8 --event-scale 0.2
