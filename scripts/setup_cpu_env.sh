#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install numpyro arviz corner

python setup.py build_ext --inplace
python -m unittest tests.test_flux tests.test_flux_derivatives tests.test_flux_derivatives_batch
python scripts/verify_jax_batch_gpu.py --batch-size 10 --n-times 120 --n-rs 7
