#!/usr/bin/env bash
# The deliverable: QAT on train1M from the best 3,073-weight float student
# (a_d16_h2, official 0.88223), two betas bracketing the ~350k-EBOPs target,
# best epoch selected on the official mixture and only after beta has ramped.
cd "$(dirname "$0")"
P="$HOME/hlsenv/bin/python"
pids=()
KERAS_BACKEND=torch PYTHONUNBUFFERED=1 $P train_attn.py --tag a_d16_h2_t2 --d 16 --heads 2 \
    --train-tag train1M --epochs 30 > logs_a_d16_h2_t2.log 2>&1 &
pids+=($!)
for b in 3e-6 1e-5; do
  KERAS_BACKEND=torch PYTHONUNBUFFERED=1 $P train_attn.py --tag "q_h2_b$b" --quantized \
      --init-from a_d16_h2 --beta0 "$b" --beta-ramp 8 --train-tag train1M --epochs 35 \
      --lr 1e-3 > "logs_q_h2_b$b.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "PROD DONE"
grep -hE "EVAL AUC|OFFICIAL" logs_a_d16_h2_t2.log logs_q_h2_b3e-6.log logs_q_h2_b1e-5.log
