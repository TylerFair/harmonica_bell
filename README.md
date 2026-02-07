<p align="center">
  <img src="/docs/source/images/transmission_string_animation_logo_small.gif" align="middle" width="400px" height="400px" alt="Harmonica"><br>
</p>

<h1 align="center">Harmonica</h1>
<p align="center">
  Light curves for exoplanet transmission mapping in python.
</p>

<p align="center">
  <a href="https://github.com/DavoGrant/harmonica/actions/workflows/unittests.yml">
    <img alt="GitHub Workflow Status" src="https://github.com/DavoGrant/harmonica/actions/workflows/unittests.yml/badge.svg">
  </a>
  <a href="https://harmonica.readthedocs.io/en/latest/?badge=latest">
    <img alt="Read the Docs" src="https://readthedocs.org/projects/harmonica/badge/?version=latest">
  </a>
</p>

```
pip install planet-harmonica
```

Read the full documentation at: [harmonica.readthedocs.io](https://harmonica.readthedocs.io).<br>
Read the full methods paper at: [mnras](https://doi.org/10.1093/mnras/stac3632) or on the [arxiv](http://arxiv.org/abs/2212.07294).<br>

## Batched JAX API

The JAX subpackage now includes explicit batched wrappers for shared-time-grid
multi-light-curve inference:

- `harmonica.jax.harmonica_transit_quad_ld_batch`
- `harmonica.jax.harmonica_transit_nonlinear_ld_batch`

Input contract:

- `times`: shape `(T,)` (shared across all curves)
- orbital / limb-darkening parameters: scalar or shape `(B,)`
- ripple coefficients `r`: shape `(B, K)` with fixed odd `K`
- output flux: shape `(B, T)`

## Reproducible Test Environments

CPU verification:

```bash
./scripts/setup_cpu_env.sh
```

CUDA verification (requires CUDA toolkit + CUDA-enabled JAX):

```bash
./scripts/setup_cuda_env.sh
```

Quick local check only:

```bash
python scripts/verify_jax_batch_gpu.py --batch-size 10 --n-times 120 --n-rs 7
```
