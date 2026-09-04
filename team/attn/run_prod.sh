#!/usr/bin/env bash
# The deliverable. Two candidates:
#   q_h2_*  : 3,073 weights, from a_d16_h2      -- the smallest student that clears 0.905
#   q_b2_*  : 5,233 weights, from a_d16_b2_t2   -- the best float student (official 0.88825)
# Best epoch is selected on the official mixture and only after beta has ramped; the best
# weights are written to disk every time they improve, so a run can be converted mid-flight.
cd "$(dirname "$0")"
P="$HOME/hlsenv/bin/python"
pids=()
launch () {  # tag seed beta
  KERAS_BACKEND=torch PYTHONUNBUFFERED=1 $P train_attn.py --tag "$1" --quantized \
      --init-from "$2" --beta0 "$3" --beta-ramp 7 --train-tag train1M --epochs 30 \
      --lr 1e-3 > "logs_$1.log" 2>&1 &
  pids+=($!)
}
launch q_h2_b3e-6 a_d16_h2    3e-6
launch q_h2_b1e-5 a_d16_h2    1e-5
launch q_b2_b3e-6 a_d16_b2_t2 3e-6
wait "${pids[@]}"
echo "PROD DONE"
grep -hE "EVAL AUC|OFFICIAL" logs_q_h2_b3e-6.log logs_q_h2_b1e-5.log logs_q_b2_b3e-6.log
