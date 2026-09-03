#!/usr/bin/env bash
# Fast beta screen on train300k: where does the EBOPs penalty put the bit widths?
cd "$(dirname "$0")"
P="$HOME/hlsenv/bin/python"
pids=()
for b in 1e-7 3e-7 1e-6 3e-6; do
  KERAS_BACKEND=torch PYTHONUNBUFFERED=1 $P train_attn.py --tag "s300_b$b" --quantized \
     --init-from a_d16 --beta0 "$b" --beta-ramp 3 --train-tag train300k --epochs 10 \
     --lr 1e-3 > "logs_s300_b$b.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "BETA SCREEN DONE"
for b in 1e-7 3e-7 1e-6 3e-6; do
  $P - "$b" <<'PY'
import json, sys
b = sys.argv[1]
s = json.load(open(f"runs/s300_b{b}_summary.json"))
print(f"beta={b:<6}  AUC {s['eval_auc']:.5f}  EBOPs {s.get('ebops', float('nan')):.0f}  "
      f"tt {s['per_background_auc']['tt']:.4f}")
PY
done
