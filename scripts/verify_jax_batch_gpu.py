#!/usr/bin/env python
"""Quick verification for batched Harmonica JAX APIs (CPU + optional GPU parity)."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Support direct execution from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harmonica.jax import (
    harmonica_transit_quad_ld,
    harmonica_transit_quad_ld_batch,
)
from harmonica.jax.custom_primitives import _quad_ld_flux_and_derivatives_batch


def build_inputs(batch_size: int, n_times: int, n_rs: int):
    rng = np.random.default_rng(42)
    times = jnp.linspace(-0.15, 0.15, n_times)
    t0 = jnp.linspace(-0.02, 0.02, batch_size)

    rs = np.zeros((batch_size, n_rs), dtype=np.float64)
    rs[:, 0] = np.linspace(0.12, 0.18, batch_size)
    if n_rs > 1:
        rs[:, 1:] = 0.01 * rs[:, [0]] * rng.uniform(-1.0, 1.0, size=(batch_size, n_rs - 1))

    return {
        "times": times,
        "t0": t0,
        "period": 4.0,
        "a": 11.0,
        "inc": 87.0 * np.pi / 180.0,
        "ecc": 0.12,
        "omega": 0.1,
        "u1": jnp.linspace(0.02, 0.05, batch_size),
        "u2": jnp.linspace(0.20, 0.26, batch_size),
        "r": jnp.asarray(rs),
    }


def verify_batch_vs_loop(params):
    batch_flux = harmonica_transit_quad_ld_batch(
        params["times"],
        params["t0"],
        params["period"],
        params["a"],
        params["inc"],
        params["ecc"],
        params["omega"],
        params["u1"],
        params["u2"],
        params["r"],
    )

    loop_flux = jnp.stack(
        [
            harmonica_transit_quad_ld(
                params["times"],
                t0=params["t0"][i],
                period=params["period"],
                a=params["a"],
                inc=params["inc"],
                ecc=params["ecc"],
                omega=params["omega"],
                u1=params["u1"][i],
                u2=params["u2"][i],
                r=params["r"][i],
            )
            for i in range(params["r"].shape[0])
        ],
        axis=0,
    )

    max_diff = float(np.max(np.abs(np.asarray(batch_flux - loop_flux))))
    return max_diff


def verify_grad(params):
    def scalar_sum(t0_vec):
        return jnp.sum(
            harmonica_transit_quad_ld_batch(
                params["times"],
                t0_vec,
                params["period"],
                params["a"],
                params["inc"],
                params["ecc"],
                params["omega"],
                params["u1"],
                params["u2"],
                params["r"],
            )
        )

    grad_t0 = jax.grad(scalar_sum)(params["t0"])
    grad_ok = bool(np.all(np.isfinite(np.asarray(grad_t0))))

    f, jac = _quad_ld_flux_and_derivatives_batch(
        params["times"],
        params["t0"],
        params["period"],
        params["a"],
        params["inc"],
        params["ecc"],
        params["omega"],
        params["u1"],
        params["u2"],
        params["r"],
    )

    jac_ok = bool(np.all(np.isfinite(np.asarray(f))) and np.all(np.isfinite(np.asarray(jac))))
    return grad_ok and jac_ok


def verify_gpu_parity(params):
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpu_devices:
        return None, "no-gpu"

    cpu_fn = jax.jit(
        lambda: harmonica_transit_quad_ld_batch(
            params["times"],
            params["t0"],
            params["period"],
            params["a"],
            params["inc"],
            params["ecc"],
            params["omega"],
            params["u1"],
            params["u2"],
            params["r"],
        ),
        backend="cpu",
    )
    gpu_fn = jax.jit(
        lambda: harmonica_transit_quad_ld_batch(
            params["times"],
            params["t0"],
            params["period"],
            params["a"],
            params["inc"],
            params["ecc"],
            params["omega"],
            params["u1"],
            params["u2"],
            params["r"],
        ),
        backend="gpu",
    )

    try:
        cpu_flux = np.asarray(cpu_fn())
        gpu_flux = np.asarray(gpu_fn())
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"gpu-error:{exc}"

    max_diff = float(np.max(np.abs(cpu_flux - gpu_flux)))
    return max_diff, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-times", type=int, default=120)
    parser.add_argument("--n-rs", type=int, default=7)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    devices = [f"{d.platform}:{d.id}" for d in jax.devices()]
    print("devices:", devices)

    params = build_inputs(args.batch_size, args.n_times, args.n_rs)

    max_diff = verify_batch_vs_loop(params)
    print("batch_vs_loop_max_abs_diff:", max_diff)

    if max_diff > args.rtol:
        raise SystemExit(f"batch vs loop check failed: {max_diff} > {args.rtol}")

    if not verify_grad(params):
        raise SystemExit("gradient/jacobian finite check failed")
    print("gradient_check: ok")

    gpu_diff, gpu_status = verify_gpu_parity(params)
    print("gpu_status:", gpu_status)

    if gpu_status == "ok":
        print("cpu_gpu_max_abs_diff:", gpu_diff)
        if gpu_diff > args.rtol:
            raise SystemExit(f"CPU/GPU parity failed: {gpu_diff} > {args.rtol}")
    elif args.require_gpu:
        raise SystemExit("GPU parity requested but GPU path is unavailable")


if __name__ == "__main__":
    main()
