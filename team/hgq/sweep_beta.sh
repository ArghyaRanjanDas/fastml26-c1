#!/usr/bin/env bash
# HGQ2 QAT beta0 sweep. beta0 weights the EBOPs term against accuracy: higher
# beta0 -> fewer effective bit-operations -> fewer DSPs, at some AUC cost.
# DSP is the binding resource at 1692/1900, so this is the lever that matters.
set -e
cd "$(dirname "$0")/.."
for b in 1e-6 1e-5 1e-4; do
  echo "##### beta0=$b"
  KERAS_BACKEND=torch ~/venv-hgq/bin/python hgq/qat_hgq.py --beta0 "$b" \
    --tag "b$b" --epochs 12 2>&1 | grep -E "pre-QAT|epoch 12|eval AUC|vs QCD|vs tt|vs Wjets|EBOPs|saved"
done
