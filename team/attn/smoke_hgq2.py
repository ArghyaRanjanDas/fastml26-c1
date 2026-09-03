"""Step 1 smoke test: toy HGQ2 MultiHeadAttention over 16 tokens -> hls4ml (Vitis, io_stream)
-> compile() -> predict(). No Vitis on this pod; C-simulation via compile() is the check."""
import os, sys, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--backend', default=os.environ.get('KERAS_BACKEND', 'torch'))
ap.add_argument('--io-type', default='io_stream')
ap.add_argument('--outdir', default='/tmp/hls_smoke')
args = ap.parse_args()
os.environ['KERAS_BACKEND'] = args.backend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keras
from keras import ops
import hgq
from hgq.layers import QDense, QEinsumDense, QMultiHeadAttention, QSum, QMeanPow2, QGlobalAveragePooling1D, QGlobalMaxPooling1D
from hgq.config import LayerConfigScope, QuantizerConfigScope
import hls4ml
import hgq2_compat  # noqa: F401  (registry alias for hgq 0.2 module move)

print('keras backend:', keras.backend.backend())

N_TOK, D = 16, 8

def build():
    inp = keras.Input(shape=(N_TOK, 5), name='particles')
    # per-token embedding: EinsumDense acts as shared Dense over tokens
    x = QEinsumDense('bnc,cd->bnd', output_shape=(N_TOK, D), activation='relu', name='embed')(inp)
    a = QMultiHeadAttention(num_heads=1, key_dim=D, name='mha')(x, x)
    x = keras.layers.Add()([x, a]) if False else a
    x = QGlobalAveragePooling1D(name='pool')(x)
    out = QDense(1, name='head')(x)
    return keras.Model(inp, out)

with QuantizerConfigScope(default_q_type='kbi', place='datalane', overflow_mode='SAT_SYM'), \
     QuantizerConfigScope(default_q_type='kbi', place='weight', overflow_mode='SAT_SYM'), \
     LayerConfigScope(enable_ebops=True, beta0=1e-5):
    model = build()
model.summary()

X = np.random.randn(64, N_TOK, 5).astype('float32')
# a fwd pass to initialise quantizer stats
y_k = np.asarray(keras.ops.convert_to_numpy(model(X)))
print('keras out', y_k.shape, y_k[:3, 0])

hls_model = hls4ml.converters.convert_from_keras_model(
    model, backend='Vitis', io_type=args.io_type,
    output_dir=args.outdir, part='xcu200-fsgd2104-2-e', clock_period=5,
)
hls_model.compile()
y_h = hls_model.predict(np.ascontiguousarray(X)).reshape(y_k.shape)
print('hls out  ', y_h[:3, 0])
print('max |diff| =', float(np.max(np.abs(y_k - y_h))))
print('SMOKE OK')
