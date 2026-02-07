#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install CUDA toolkit first."
  exit 1
fi

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install numpyro arviz corner

# Install a CUDA-enabled JAX build suitable for your platform.
# Example (Linux CUDA 12):
# python -m pip install -U "jax[cuda12]"

HARMONICA_ENABLE_CUDA=1 python setup.py build_ext --inplace
python -m unittest tests.test_flux tests.test_flux_derivatives tests.test_flux_derivatives_batch
python scripts/verify_jax_batch_gpu.py --batch-size 10 --n-times 120 --n-rs 7 --require-gpu
