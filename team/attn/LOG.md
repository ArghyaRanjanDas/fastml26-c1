# Lane c3 — attention student. Working log.

## Step 1 — environment + HGQ2/hls4ml smoke test  ✅ (bit-exact)

`~/hlsenv/bin/pip install hgq da4ml` **fails**: PyPI `hgq` tops out at 0.2.6, which is
HGQ **v1** (Keras 2, `tensorflow<2.16`) and cannot coexist with the pod's TF 2.21 /
Keras 3.15. HGQ2 is published under a different name: **`hgq2`** (0.2.0), which installs
the importable module `hgq`. `~/hlsenv/bin/pip install hgq2 da4ml` → hgq2 0.2.0,
da4ml 0.6.0, no dependency conflicts.

Two things had to be fixed to convert a `QMultiHeadAttention`:

1. **Registry key mismatch (hls4ml 1.3.0 vs hgq2 0.2.0).** hls4ml dispatches keras-v3
   layers on `f'{layer.__module__}.{cls}'`. Its handler registers
   `hgq.layers.multi_head_attention.QMultiHeadAttention`, but in hgq2 0.2.0 the class
   lives at `hgq.layers.attn.mha.QMultiHeadAttention`. The lookup misses, hls4ml falls
   through to the da4ml fallback, and that dies with
   `ValueError: The name "keras_tensorCLONE" is used 2 times in the model`.
   Fix: `team/attn/hgq2_compat.py` re-keys the existing handler (also for
   `QLinformerAttention`). Import it before converting. Only these two layers moved.
2. **`io_stream` is not usable for HGQ2 models.** hls4ml raises
   `NotImplementedError: Heterogenous quantization for activations is only supported
   with IOType=io_parallel`. Per-parameter bit widths are the whole point of HGQ, so
   the lane uses **`io_parallel`** — which is also what the DeepSet lane synthesized
   and the lower-latency choice.

Also: `QMeanPow2` has no keras-v3 handler and falls back to da4ml, which then breaks on
a 2-D (16, 8) input (`operands could not be broadcast together with shapes (16,8) (128,)`).
Use `QGlobalAveragePooling1D` / `QGlobalMaxPooling1D`, both of which are registered.

Smoke test `team/attn/smoke_hgq2.py` (embed → 1-head MHA over 16 tokens → mean pool → dense,
Vitis backend, `xcu200-fsgd2104-2-e`, 5 ns):

```
keras out  [-0.66796875 -0.24536133  1.1367188 ]
hls out    [-0.66796875 -0.24536133  1.1367188 ]
max |diff| = 0.0     <- bit-exact, as HGQ2 promises
```

Elapsed: ~25 min, well inside the 2 h budget before falling back to 2b. **Step 2a it is.**

Training runs in `~/hlsenv` with `KERAS_BACKEND=torch` — that venv's torch 2.9.1+cu128
sees the A10, so the same Keras model is trained on the GPU and converted for HLS, with
no port between frameworks.

## Step 2a — the architecture, and two more hls4ml limits

`team/attn/model.py` builds one topology with two skins: plain Keras (float) and HGQ2
(`quantized=True`). Every HGQ layer subclasses its Keras counterpart, so `kernel`/`bias`
shapes match and `transfer_weights()` warm-starts QAT from the float run.

```
particles (16, 11)  ->  EinsumDense 'bnc,cd->bnd' d, relu          (= c1's Conv1D k=1 phi)
                    ->  [ MHA(1 head, key_dim=d) + residual
                          EinsumDense d->2d relu -> d + residual ] x blocks
                    ->  mean-pool  ||  max-pool                    (c2: max is free in firmware)
    + event (11)    ->  concat -> Dense 16 relu -> Dense 1
```

No positional encoding (Laatu et al.), no LayerNorm (nothing in HGQ2 maps to it, and the
residual blocks train fine without at this depth). The 11 particle channels are c2's
`rich` set; like the 5 base ones they are fixed per-candidate functions computed upstream
of the network, not layers we synthesize — `physics.derived.rich_particles` reproduces
c2's `cache/eval100k_rich` to 0.0 from the base cache, which is how `train1M_rich` (for
which c2 built no cache) is derived here.

Two more hls4ml facts found the hard way:

* **`Concatenate` is binary.** `hls4ml/model/layers.py:Concatenate.initialize` asserts
  exactly two inputs, so `[mean, max, event]` in one call dies on a bare `AssertionError`.
  Fold pairwise instead.
* **`QMeanPow2` / `QSum` have no keras-v3 handler** (see step 1).

End-to-end proof on a throwaway 2-epoch quantized run: convert → `compile()` → predict on
`team/export/eval_sample.npz`, **max |keras − hls| = 0.0**, AUC identical to 5 decimals.
So for this lane the closure question that cost the DeepSet lane 0.014 AUC does not exist:
HGQ2 + hls4ml is bit-exact by construction, and the number to manage is EBOPs, not overflow.

## Step 3 — distillation, and what each piece is worth  ✅

All rows: `train1M` (2M events), 30 epochs, AdamW + cosine, KD from `ds_big_s0`'s logits
at T=2, alpha 0.7, evaluated on the same `eval100k` slice as every other lane.
"params" is kernels + biases, i.e. what gets synthesized.

| run | change from `a_d16` | params | AUC | vs QCD | vs tt | vs W+jets |
|---|---|---|---|---|---|---|
| `a_d16_b0` | **no attention** (0 blocks) | 913 | 0.89067 | 0.9289 | 0.7726 | 0.9706 |
| `a_d8` | d = 8 | 1,129 | 0.89651 | 0.9320 | 0.7866 | 0.9709 |
| `a_d16_nomlp` | no per-token MLP | 2,001 | 0.90272 | 0.9376 | 0.7979 | 0.9726 |
| `a_d16_base` | 5 base channels, no `rich` | 2,977 | 0.90119 | 0.9310 | 0.8016 | 0.9709 |
| **`a_d16`** | — | **3,073** | **0.90818** | 0.9403 | 0.8100 | 0.9742 |
| `a_d16_nokd` | no distillation | 3,073 | 0.90648 | 0.9390 | 0.8072 | 0.9732 |
| `a_d16_h2` | 2 heads | 3,073 | 0.90864 | 0.9396 | 0.8136 | 0.9728 |
| `a_d16_b2` | 2 blocks | 5,233 | 0.91138 | 0.9422 | 0.8168 | 0.9752 |
| `a_d32` | d = 32 | 10,033 | 0.91267 | 0.9429 | 0.8198 | 0.9753 |

Reference points: DeepSet student `B1e_16p_1M` 0.88687 / tt 0.75869; the rich DeepSet
`model_2777_rich` 0.9077 float / 0.9062 synthesized; teacher `ds_big_s0` 0.91515 / tt 0.82612.

Reading the ablations:

* **Attention is worth +0.0175 overall and +0.037 vs tt** (`a_d16_b0` → `a_d16`), holding
  the embedding, the pooling, the head, the inputs and the training recipe fixed. That is
  the answer to the question the lane was set up to ask, and it is the same size as the
  jump c2 measured from adding the six rich channels — the two are additive, not
  alternative: `a_d16_base` (attention, no rich) is 0.90119 and `a_d16_b0` (rich, no
  attention) is 0.89067, while both together are 0.90818.
* **Distillation is worth +0.0017** (`a_d16_nokd` → `a_d16`) — real but small; the
  attention student gets most of the teacher's advantage from its own architecture rather
  than from the soft targets. For the DeepSet student the same teacher was worth more,
  because the DeepSet had no way to represent what the teacher knew.
* **Depth beats width.** `a_d16_b2` (2 blocks, 5,233 params) ≈ `a_d32` (1 block, 10,033)
  at half the weights and a quarter of the attention arithmetic.
* `a_d16` reaches **97 % of the teacher's margin over the DeepSet student**
  ((0.90818 − 0.88687) / (0.91515 − 0.88687)) with 3,073 weights against the teacher's 72,717.

## Step 4 — QAT: calibrating the EBOPs penalty

`beta0` multiplies EBOPs directly in the loss. Unregularized, the d=16 model sits at
**2.36M EBOPs**, so `beta0 = 1e-5` puts 24 loss units against a BCE of ~1.8 — the model is
crushed to 29k EBOPs and 0.858 val AUC within five epochs. **The useful range is 1e-7 to
3e-6**, not the 1e-5..1e-4 that a first guess suggests. (Recorded because the same wrong
guess had already gone into the A100 queue; corrected there in `A100_QUEUE.md`.)

Practical note: HGQ2 QAT costs ~7 min/epoch on 2M events against ~40 s for the float model
— every weight and activation carries its own trainable bit width — so the float sweep runs
on the A10 and the long QAT runs go to the A100 queue.
