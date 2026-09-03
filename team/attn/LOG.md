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
