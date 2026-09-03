#!/usr/bin/env bash
# Size sweep for Challenge 1.  Two families:
#   A = the originally requested ~40k/10k/3k sizes at phi 64-32-16
#   B/C = FPGA-driven: narrow phi and fewer particles, because Vitis showed phi
#         (replicated once per particle) is the entire DSP/LUT bill.
# No dropout anywhere -- these are the small final models.
set -e
cd "$(dirname "$0")"
run () { # tag phi rho npart extra...
  local tag=$1 phi=$2 rho=$3 np=$4; shift 4
  echo "##### $tag  phi=$phi rho=$rho particles=$np $*"
  python train.py --model deepset_plus --phi "$phi" --rho "$rho" --dropout 0 \
    --pool-norm --epochs 25 --n-particles-use "$np" --tag "$tag" "$@" 2>&1 \
    | grep -E "trainable params|BINARY AUC \(signal|vs QCD|vs tt|vs Wjets|params = "
}
# --- A: requested sizes, phi 64-32-16, no event features -------------------
run A1_40k  64,32,16  256,128  16 --no-event-features
run A2_9k   64,32,16  96,40    16 --no-event-features
run A3_3k   64,32,16  16,8     16 --no-event-features
# --- A with event features, to test whether they pay off once rho is tiny ---
run A2e_9k  64,32,16  96,40    16 --event-scale 0.2
run A3e_3k  64,32,16  16,8     16 --event-scale 0.2
# --- B: narrow phi, 16 particles -------------------------------------------
run B1_16p  32,16,8   32,16    16 --no-event-features
run B1e_16p 32,16,8   32,16    16 --event-scale 0.2
run B2_16p  32,16,8   64,32    16 --no-event-features
# --- C: narrow phi, 8 particles (halves the phi bill) ----------------------
run C1_8p   32,16,8   32,16     8 --no-event-features
run C1e_8p  32,16,8   32,16     8 --event-scale 0.2
run C2_8p   16,8      32,16     8 --no-event-features
run C2e_8p  16,8      32,16     8 --event-scale 0.2
