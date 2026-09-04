#!/usr/bin/env bash
# Re-train the two candidate students against the new ParT-ensemble teacher
# (ens_part4, 0.92480 / tt 0.85181) instead of ds_big_s0 (0.91515 / tt 0.82612).
cd "$(dirname "$0")"
P="$HOME/hlsenv/bin/python"
pids=()
for a in "a_d16_t2 --d 16" "a_d16_b2_t2 --d 16 --blocks 2"; do
  set -- $a; tag=$1; shift
  KERAS_BACKEND=torch PYTHONUNBUFFERED=1 $P train_attn.py --tag $tag "$@" \
      --train-tag train1M --epochs 30 > "logs_$tag.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "T2 RUNS DONE"
grep -h "EVAL AUC" logs_a_d16_t2.log logs_a_d16_b2_t2.log
