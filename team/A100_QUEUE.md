# A100 job queue (Purdue AF, agent hh4b runs it)

The A100 (40 GB) is fully ours until the hackathon ends (Friday). Any agent can append a job here;
hh4b pulls this file every ~10 minutes, runs jobs top to bottom, and writes the result under the job.

Rules: one job = one fenced command block that runs from the repo root on the AF (`/work/users/das214/fastml26/fastml26-c1`,
venv `../venv`, caches in `team/cache/{train1M,train300k,eval100k}` with raw X 16x5, F 11, y, group; rich
channels are computed on the fly by `team/data.py`). Put the expected runtime and the number to beat.
hh4b: mark a job `[running]`, then `[done: <AUC etc>]` and commit+push; never delete jobs.

## Jobs


---

### c3-0 — setup for every c3 job below (run once)

The attention lane trains a Keras 3 model on the torch backend with **HGQ2**, so the AF
venv needs two packages it does not have. Nothing else is new: no TensorFlow, no hls4ml
(conversion happens on the pod), and the data comes from the caches already there —
`team/attn/attn_data.py` derives c2's 6 rich per-candidate channels from the base
`X.npy` itself (verified equal to `cache/eval100k_rich` to 0.0) and memoises them under
`team/attn/cache/` (~1.4 GB for train1M, gitignored). Teacher logits are read from
`team/teacher/soft_targets_train1M.npy`.

```bash
cd /work/users/das214/fastml26/fastml26-c1 && git pull
../venv/bin/pip install "keras>=3.15" hgq2==0.2.0
# sanity: should print the shapes and a 0.0
KERAS_BACKEND=torch ../venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "team/attn"); sys.path.insert(0, "team")
import attn_data, numpy as np
X, F, y, g, m = attn_data.load("eval100k")
print(X.shape, F.shape, y.mean())
print("teacher logits:", attn_data.soft_targets("eval100k").shape)
PY
```

Expected: ~10 min including the pip install and the rich derivation.
**If `pip install hgq2` fails, skip every c3 job and say so under this block** — the pod
will run them on the A10 instead.

### c3-1 — QAT beta sweep for the attention student (the lane's long pole)

HGQ2 quantization-aware training, warm-started from the float run `a_d16_b2`
(5,233 weights, **float eval AUC 0.91138**, vs tt 0.8168 — already committed as
`team/attn/runs/a_d16_b2.weights.h5`). `beta0` is the EBOPs penalty: it sets the
per-parameter bit widths and through them the LUT bill. I need the AUC-vs-EBOPs curve
to pick the point that fits one VU9P SLR (~350k LUT; the reference design in
arXiv:2510.24784 is 279k LUT at an EBOPs target of ~350k).

**Number to beat: eval AUC 0.9062** (the DeepSet lane's synthesized `model_2777_rich`).
The quantized attention student converts bit-exactly, so its quantized AUC *is* its
FPGA AUC — no closure loss to budget for.

```bash
cd /work/users/das214/fastml26/fastml26-c1/team/attn
for b in 3e-6 1e-5 3e-5 1e-4 3e-4; do
  KERAS_BACKEND=torch ../../../venv/bin/python train_attn.py \
      --tag q_b2_b$b --quantized --init-from a_d16_b2 --beta0 $b --beta-ramp 5 \
      --train-tag train1M --epochs 40 --lr 1e-3 > logs_q_b2_b$b.log 2>&1
  ../../../venv/bin/python - "$b" <<'PY'
import json, sys
b = sys.argv[1]
s = json.load(open(f"runs/q_b2_b{b}_summary.json"))
print(b, "AUC", round(s["eval_auc"], 5), "EBOPs", round(s.get("ebops", float("nan"))),
      "vs tt", round(s["per_background_auc"]["tt"], 4))
PY
done
git add -A runs && git commit -m "c3-1: QAT beta sweep on A100" && git push
```

Expected: **~2.5 h per beta, ~12 h for the five** — corrected after measuring it on the
A10: HGQ2 QAT costs ~7 min/epoch on 2M events (vs ~40 s/epoch for the float model), because
every weight and every activation carries its own trainable bit width. If that is too much
of the queue, **run `3e-5` and `1e-4` first and push those two before starting the rest** —
they are the two most likely to land near the ~350k-EBOPs target, and two points plus the
unregularized 2.36M-EBOPs starting value already give the shape of the curve. The warm
start converges fast (epoch 1 is already at val AUC 0.9026), so 40 epochs is generous;
drop to `--epochs 20` if the queue is busy. Report the (beta, AUC, EBOPs, vs-tt) table.

### c3-2 — how far the float attention student goes with more capacity + time

The A10 sweep at 30 epochs gave 0.90818 (d=16, 3,073 w), 0.91138 (d=16 ×2 blocks,
5,233 w) and 0.91267 (d=32, 10,033 w) against a **teacher at 0.91515**. This asks
whether a wider/deeper student trained longer closes the last 0.002 — if it does, it
becomes the QAT seed; if it does not, `a_d16_b2` is the answer and we stop widening.

**Number to beat: 0.91267.**

```bash
cd /work/users/das214/fastml26/fastml26-c1/team/attn
V=../../../venv/bin/python
KERAS_BACKEND=torch $V train_attn.py --tag a_d32_b2 --d 32 --blocks 2 --train-tag train1M --epochs 60 > logs_a_d32_b2.log 2>&1
KERAS_BACKEND=torch $V train_attn.py --tag a_d24_b2 --d 24 --blocks 2 --train-tag train1M --epochs 60 > logs_a_d24_b2.log 2>&1
KERAS_BACKEND=torch $V train_attn.py --tag a_d16_b2_e60 --d 16 --blocks 2 --train-tag train1M --epochs 60 > logs_a_d16_b2_e60.log 2>&1
KERAS_BACKEND=torch $V train_attn.py --tag a_d16_b2_s1 --d 16 --blocks 2 --train-tag train1M --epochs 60 --seed 1 > logs_a_d16_b2_s1.log 2>&1
grep -h "EVAL AUC" logs_a_d32_b2.log logs_a_d24_b2.log logs_a_d16_b2_e60.log logs_a_d16_b2_s1.log
git add -A runs && git commit -m "c3-2: float capacity/length sweep on A100" && git push
```

Expected: ~40 min each, **~2.5 h total**. The seed-1 repeat of `a_d16_b2` is there to
size the run-to-run spread, so the table above can be read honestly.

---

### c1-1 — HGQ2 QAT, longer, best-epoch checkpointed

On the A10 (shared with screening runs) 12 epochs of HGQ2 QAT got EBOPs 853k -> 360k
(2.4x, which is the DSP lever) but only AUC 0.89044 — below the 0.895 bar and below the
PTQ path's 0.906 HLS. Val AUC peaked at epoch 6 (0.89477) and then decayed under EBOPs
pressure, so the run needs (a) best-epoch checkpointing, now in `qat_hgq.py`, and (b) many
more epochs at low beta0 for the quantizer bitwidths to settle. Warm-start pre-QAT AUC is
0.548, so a good fraction of training is just the bitwidths learning.

Expected runtime: ~40 min per beta0 at 40 epochs on an A100 (Keras 3 / torch backend).
**Number to beat: AUC 0.895** at EBOPs materially below 853k. Best so far: 0.89044 @ 360k EBOPs.

```bash
cd /work/users/das214/fastml26/fastml26-c1 && git pull
../venv/bin/pip install "keras>=3.15" hgq2==0.2.0 2>/dev/null || true
# rich caches: build once if absent (pure transform of train1M/eval100k, ~1 min)
[ -d team/cache/train1M_s ] || ../venv/bin/python team/make_student_cache.py --tag train1M
[ -d team/cache/eval100k_s ] || ../venv/bin/python team/make_student_cache.py --tag eval100k --norm-from train1M_s
for b in 3e-7 1e-6 3e-6; do
  KERAS_BACKEND=torch ../venv/bin/python team/hgq/qat_hgq.py \
    --beta0 $b --tag "a100_b$b" --epochs 40 --lr 5e-4 2>&1 | tail -25
done
```

---

### c1-2 — best student on train4M when c2's cache lands

The 2,041-param student gained +0.0031 going from 600k to 2M events, which is about what
the entire rho sweep was worth and costs nothing in hardware. The rich student has not been
retrained at larger scale at all.

Expected runtime: ~25 min (30 epochs, GPU-resident batching). **Number to beat: 0.90901.**

```bash
cd /work/users/das214/fastml26/fastml26-c1 && git pull
../venv/bin/python team/make_student_cache.py --tag train4M
../venv/bin/python team/distill.py --soft-targets team/teacher --tag c1_rich_4M \
  --temperature 2 --alpha 0.5 --phi 32,16,8 --rho 32,16 --pool meanmax \
  --gpu-batches --epochs 30 --train-tag train4M_s --eval-tag eval100k_s
```
