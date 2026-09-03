"""The c3 attention student, in one topology with two skins.

`build(cfg, quantized=False)` gives the plain-Keras float model; `quantized=True`
gives the byte-identical HGQ2 model (every HGQ layer subclasses its Keras
counterpart, so `kernel`/`bias` shapes match and `transfer_weights` moves a trained
float model into the quantized one).

Shape, per arXiv:2510.24784 (Laatu et al.): 16 tokens, no positional encoding,
1 head, tiny d.  Pooling is mean **and** max (c2: free in firmware, +0.001), the 11
event features are concatenated after the pool (c1: they cost nothing in the
per-particle block), and the head is a single hidden Dense.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import keras


@dataclass
class Cfg:
    n_particles: int = 16
    n_channels: int = 11          # 5 base + 6 c2 "rich"
    n_event: int = 11
    d: int = 16                   # token width
    heads: int = 1
    blocks: int = 1
    mlp_ratio: int = 2            # 0 disables the per-token MLP after attention
    head_dim: int = 16            # width of the Dense after the pool
    pool: str = "meanmax"         # "mean" | "meanmax"
    residual: bool = True

    def to_dict(self):
        return asdict(self)


def _layers(quantized: bool):
    if quantized:
        from hgq.layers import (QDense, QEinsumDense, QMultiHeadAttention,
                                QGlobalAveragePooling1D, QGlobalMaxPooling1D, QAdd)
        return dict(Dense=QDense, EinsumDense=QEinsumDense, MHA=QMultiHeadAttention,
                    Avg=QGlobalAveragePooling1D, Max=QGlobalMaxPooling1D, Add=QAdd)
    from keras.layers import (Dense, EinsumDense, MultiHeadAttention,
                              GlobalAveragePooling1D, GlobalMaxPooling1D, Add)
    return dict(Dense=Dense, EinsumDense=EinsumDense, MHA=MultiHeadAttention,
                Avg=GlobalAveragePooling1D, Max=GlobalMaxPooling1D, Add=Add)


def build(cfg: Cfg, quantized: bool = False) -> keras.Model:
    L = _layers(quantized)
    P, C, d = cfg.n_particles, cfg.n_channels, cfg.d

    parts = keras.Input(shape=(P, C), name="particles")
    evt = keras.Input(shape=(cfg.n_event,), name="event")

    # shared per-candidate embedding; EinsumDense over the token axis is a Dense
    # replicated P times -- the same object as c1's Conv1D(kernel=1) phi.
    x = L['EinsumDense']("bnc,cd->bnd", output_shape=(P, d), bias_axes="d",
                         activation="relu", name="embed")(parts)

    for b in range(cfg.blocks):
        a = L['MHA'](num_heads=cfg.heads, key_dim=d // cfg.heads,
                     name=f"mha{b}")(x, x)
        x = L['Add'](name=f"res_attn{b}")([x, a]) if cfg.residual else a
        if cfg.mlp_ratio:
            h = L['EinsumDense']("bnc,cd->bnd", output_shape=(P, d * cfg.mlp_ratio),
                                 bias_axes="d", activation="relu", name=f"mlp{b}_0")(x)
            h = L['EinsumDense']("bnc,cd->bnd", output_shape=(P, d), bias_axes="d",
                                 name=f"mlp{b}_1")(h)
            x = L['Add'](name=f"res_mlp{b}")([x, h]) if cfg.residual else h

    pooled = [L['Avg'](name="mean_pool")(x)]
    if cfg.pool == "meanmax":
        pooled.append(L['Max'](name="max_pool")(x))
    # hls4ml's Concatenate is binary (`model/layers.py:Concatenate.initialize`
    # asserts two inputs), so fold the three branches pairwise instead of in one call.
    z = pooled[0]
    for i, nxt in enumerate([*pooled[1:], evt]):
        z = keras.layers.Concatenate(name=f"concat{i}")([z, nxt])
    z = L['Dense'](cfg.head_dim, activation="relu", name="head0")(z)
    out = L['Dense'](1, name="logit")(z)
    return keras.Model([parts, evt], out, name="attn_student")


def transfer_weights(src: keras.Model, dst: keras.Model) -> int:
    """Copy every variable whose (layer name, variable basename) matches. Returns
    the number of tensors moved; quantizer state in `dst` is left untouched."""
    moved = 0
    for ls in src.layers:
        try:
            ld = dst.get_layer(ls.name)
        except ValueError:
            continue
        by_name = {}
        for v in ld.weights:
            by_name.setdefault(v.path.split('/')[-1], []).append(v)
        for v in ls.weights:
            key = v.path.split('/')[-1]
            cands = [c for c in by_name.get(key, []) if tuple(c.shape) == tuple(v.shape)]
            if len(cands) == 1:
                cands[0].assign(v.value)
                moved += 1
    return moved


def n_synth_params(model: keras.Model) -> int:
    """Trainable kernel/bias count -- what actually gets synthesized, excluding
    HGQ's bit-width variables."""
    import numpy as np

    n = 0
    for v in model.weights:
        base = v.path.split('/')[-1]
        if base.lstrip('_').startswith(("kernel", "bias", "gamma", "beta")):
            n += int(np.prod(v.shape))
    return n
