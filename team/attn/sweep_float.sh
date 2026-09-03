#!/usr/bin/env bash
# Float architecture sweep for the attention student (train1M, distilled from ds_big_s0).
set -u
cd "$(dirname "$0")"
P="$HOME/hlsenv/bin/python"
run () { KERAS_BACKEND=torch $P train_attn.py --train-tag train1M --epochs 30 "$@"; }
run --tag a_d16      --d 16 ;
run --tag a_d8       --d 8  ;
run --tag a_d32      --d 32 ;
run --tag a_d16_b2   --d 16 --blocks 2 ;
run --tag a_d16_h2   --d 16 --heads 2 ;
run --tag a_d16_nomlp --d 16 --mlp-ratio 0 ;
run --tag a_d16_base --d 16 --no-rich ;
