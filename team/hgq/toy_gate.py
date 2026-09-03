"""Toy-model gate: does a mean-pool + concat DeepSet survive hls4ml conversion?

Per the action brief, this runs BEFORE spending GPU hours on HGQ2 QAT.  It builds
the smallest thing with our topology -- per-candidate QDense over the candidate
axis, mean pool over 16 slots, concat of event scalars, QDense head -- and pushes
it through convert_from_keras_model.  If the pool or the concat breaks conversion,
we stop here rather than after a training run.

  KERAS_BACKEND=torch ~/venv-hgq/bin/python hgq/toy_gate.py
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
import hls4ml

N_PART, N_CH, N_EVT = 16, 11, 19
MEANMAX = os.environ.get("MEANMAX", "0") == "1"


def build(qkeras_style=True):
    from hgq.layers import QDense, QEinsumDenseBatchnorm  # noqa: F401
    from hgq.config import LayerConfigScope, QuantizerConfigScope

    x = keras.Input(shape=(N_PART, N_CH), name="particles")
    e = keras.Input(shape=(N_EVT,), name="event")
    # QDense applied over the candidate axis is the per-particle phi: Keras Dense
    # already acts on the last axis, so a (16, 11) input gives 16 shared copies.
    h = QDense(8, activation="relu", name="phi0")(x)
    h = QDense(4, activation="relu", name="phi1")(h)
    if MEANMAX:
        # mean+max concatenated: max is comparators only, no DSP -- but it has to
        # survive conversion, which is exactly what this gate is for.
        h = keras.layers.Concatenate(name="pool")(
            [keras.layers.GlobalAveragePooling1D(name="pool_mean")(h),
             keras.layers.GlobalMaxPooling1D(name="pool_max")(h)])
    else:
        h = keras.layers.GlobalAveragePooling1D(name="pool")(h)
    h = keras.layers.Concatenate(name="concat")([h, e])
    h = QDense(4, activation="relu", name="rho0")(h)
    out = QDense(1, activation="sigmoid", name="score")(h)
    return keras.Model([x, e], out)


def main():
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    with QuantizerConfigScope(place="all", default_q_type="kbi", overflow_mode="SAT_SYM"), \
         QuantizerConfigScope(place="datalane", default_q_type="kif", overflow_mode="WRAP"), \
         LayerConfigScope(enable_ebops=True, beta0=1e-5):
        model = build()
    model.summary(print_fn=lambda s: print("  " + s))

    xs = [np.random.rand(64, N_PART, N_CH).astype("float32"),
          np.random.rand(64, N_EVT).astype("float32")]
    yk = keras.ops.convert_to_numpy(model(xs)).ravel()
    print(f"keras forward ok, output range {yk.min():.4f}..{yk.max():.4f}")

    out = os.path.expanduser("~/hls_toy_gate_mm" if MEANMAX else "~/hls_toy_gate")
    hm = hls4ml.converters.convert_from_keras_model(
        model, output_dir=out, backend="Vitis", part="xcu200-fsgd2104-2-e",
        clock_period=5.0, io_type="io_parallel")
    print("CONVERSION OK -- pool and concat survived")
    hm.compile()
    yh = np.asarray(hm.predict(xs)).ravel()
    print(f"csim max|keras - hls| = {np.abs(yk - yh).max():.5f}")
    print("GATE PASSED")


if __name__ == "__main__":
    main()
