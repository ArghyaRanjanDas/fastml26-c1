"""hls4ml 1.3.0 <-> HGQ2 0.2.0 compatibility.

hls4ml's keras_v3 handler registry is keyed by `<module>.<class>`. Between HGQ2 0.1.x
and 0.2.0 the attention layers moved from `hgq.layers.multi_head_attention` /
`hgq.layers.linformer_attention` to `hgq.layers.attn.{mha,linformer}`, so hls4ml no
longer finds them and silently falls back to the da4ml handler (which then dies on
duplicate tensor names). Re-key the existing handlers under the new module paths.

Import this *before* calling `hls4ml.converters.convert_from_keras_model`.
"""

from hls4ml.converters.keras_v3 import layer_handlers as _registry

_ALIASES = {
    'hgq.layers.multi_head_attention.QMultiHeadAttention': 'hgq.layers.attn.mha.QMultiHeadAttention',
    'hgq.layers.linformer_attention.QLinformerAttention': 'hgq.layers.attn.linformer.QLinformerAttention',
}

patched = []
for _old, _new in _ALIASES.items():
    if _old in _registry and _new not in _registry:
        _registry[_new] = _registry[_old]
        patched.append(_new)


def check():
    """Raise unless every hgq layer class reachable from `hgq.layers` is registered."""
    import hgq.layers as HL

    missing = []
    for name in dir(HL):
        obj = getattr(HL, name)
        if not isinstance(obj, type) or not name.startswith('Q'):
            continue
        key = f'{obj.__module__}.{obj.__qualname__}'
        if key not in _registry and name not in _registry:
            missing.append(key)
    return missing
