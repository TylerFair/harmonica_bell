import unittest

import jax
import jax.numpy as jnp
import numpy as np

from harmonica.jax import (
    harmonica_transit_nonlinear_ld,
    harmonica_transit_nonlinear_ld_batch,
    harmonica_transit_quad_ld,
    harmonica_transit_quad_ld_batch,
)
from harmonica.jax.custom_primitives import (
    _nonlinear_ld_flux_and_derivatives_batch,
    _quad_ld_flux_and_derivatives_batch,
)


class TestFluxBatch(unittest.TestCase):
    """Test batched JAX primitives and wrappers."""

    def setUp(self):
        np.random.seed(7)

    @staticmethod
    def _build_batch_inputs(batch_size=10, n_times=120, n_rs=7):
        times = jnp.linspace(-0.15, 0.15, n_times)
        t0 = jnp.linspace(-0.02, 0.02, batch_size)
        period = 4.0
        a = 11.0
        inc = 87.0 * np.pi / 180.0
        ecc = 0.12
        omega = 0.1
        u1 = jnp.linspace(0.02, 0.05, batch_size)
        u2 = jnp.linspace(0.20, 0.26, batch_size)

        # Keep fixed odd K across the batch.
        rs = np.zeros((batch_size, n_rs), dtype=np.float64)
        rs[:, 0] = np.linspace(0.12, 0.18, batch_size)
        if n_rs > 1:
            rs[:, 1:] = 0.01 * rs[:, [0]] * np.random.uniform(
                -1.0, 1.0, size=(batch_size, n_rs - 1)
            )

        return (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            jnp.asarray(rs),
        )

    def test_api_jax_quad_ld_batch_shape(self):
        batch_size = 10
        n_times = 120
        (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r,
        ) = self._build_batch_inputs(batch_size=batch_size, n_times=n_times)

        flux = harmonica_transit_quad_ld_batch(
            times, t0, period, a, inc, ecc, omega, u1, u2, r
        )

        self.assertEqual(flux.shape, (batch_size, n_times))
        self.assertTrue(np.all(np.isfinite(np.asarray(flux))))

    def test_api_jax_nonlinear_ld_batch_shape(self):
        batch_size = 8
        n_times = 80
        (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r,
        ) = self._build_batch_inputs(batch_size=batch_size, n_times=n_times)

        u3 = jnp.linspace(-0.10, 0.10, batch_size)
        u4 = jnp.linspace(-0.05, 0.05, batch_size)

        flux = harmonica_transit_nonlinear_ld_batch(
            times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4, r
        )

        self.assertEqual(flux.shape, (batch_size, n_times))
        self.assertTrue(np.all(np.isfinite(np.asarray(flux))))

    def test_batch_vs_loop_parity_quad_ld(self):
        (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r,
        ) = self._build_batch_inputs(batch_size=7, n_times=90)

        batch_flux = harmonica_transit_quad_ld_batch(
            times, t0, period, a, inc, ecc, omega, u1, u2, r
        )

        loop_flux = jnp.stack(
            [
                harmonica_transit_quad_ld(
                    times,
                    t0=t0[i],
                    period=period,
                    a=a,
                    inc=inc,
                    ecc=ecc,
                    omega=omega,
                    u1=u1[i],
                    u2=u2[i],
                    r=r[i],
                )
                for i in range(r.shape[0])
            ],
            axis=0,
        )

        np.testing.assert_allclose(
            np.asarray(batch_flux), np.asarray(loop_flux), rtol=1e-12, atol=1e-12
        )

    def test_batch_vs_loop_parity_nonlinear_ld(self):
        (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r,
        ) = self._build_batch_inputs(batch_size=6, n_times=75)

        u3 = jnp.linspace(-0.10, 0.10, r.shape[0])
        u4 = jnp.linspace(-0.05, 0.05, r.shape[0])

        batch_flux = harmonica_transit_nonlinear_ld_batch(
            times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4, r
        )

        loop_flux = jnp.stack(
            [
                harmonica_transit_nonlinear_ld(
                    times,
                    t0=t0[i],
                    period=period,
                    a=a,
                    inc=inc,
                    ecc=ecc,
                    omega=omega,
                    u1=u1[i],
                    u2=u2[i],
                    u3=u3[i],
                    u4=u4[i],
                    r=r[i],
                )
                for i in range(r.shape[0])
            ],
            axis=0,
        )

        np.testing.assert_allclose(
            np.asarray(batch_flux), np.asarray(loop_flux), rtol=1e-12, atol=1e-12
        )

    def test_batch_derivative_shapes_quad_ld(self):
        (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r,
        ) = self._build_batch_inputs(batch_size=5, n_times=50, n_rs=7)

        f, df_dz = _quad_ld_flux_and_derivatives_batch(
            times, t0, period, a, inc, ecc, omega, u1, u2, r
        )

        self.assertEqual(f.shape, (5, 50))
        self.assertEqual(df_dz.shape, (5, 50, 6 + 2 + 7))
        self.assertTrue(np.all(np.isfinite(np.asarray(f))))
        self.assertTrue(np.all(np.isfinite(np.asarray(df_dz))))

    def test_batch_derivative_shapes_nonlinear_ld(self):
        (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r,
        ) = self._build_batch_inputs(batch_size=4, n_times=45, n_rs=5)

        u3 = jnp.linspace(-0.10, 0.10, r.shape[0])
        u4 = jnp.linspace(-0.05, 0.05, r.shape[0])

        f, df_dz = _nonlinear_ld_flux_and_derivatives_batch(
            times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4, r
        )

        self.assertEqual(f.shape, (4, 45))
        self.assertEqual(df_dz.shape, (4, 45, 6 + 4 + 5))
        self.assertTrue(np.all(np.isfinite(np.asarray(f))))
        self.assertTrue(np.all(np.isfinite(np.asarray(df_dz))))

    def test_batch_grad_and_jvp_quad_ld(self):
        (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r,
        ) = self._build_batch_inputs(batch_size=5, n_times=30, n_rs=3)

        def scalar_sum(t0_vec):
            return jnp.sum(
                harmonica_transit_quad_ld_batch(
                    times, t0_vec, period, a, inc, ecc, omega, u1, u2, r
                )
            )

        grad_t0 = jax.grad(scalar_sum)(t0)
        self.assertEqual(grad_t0.shape, (5,))
        self.assertTrue(np.all(np.isfinite(np.asarray(grad_t0))))

        values, tangents = jax.jvp(
            lambda x: harmonica_transit_quad_ld_batch(
                times, x, period, a, inc, ecc, omega, u1, u2, r
            ),
            (t0,),
            (jnp.ones_like(t0),),
        )
        self.assertEqual(values.shape, (5, 30))
        self.assertEqual(tangents.shape, (5, 30))
        self.assertTrue(np.all(np.isfinite(np.asarray(tangents))))

    def test_vmap_single_api_matches_batch(self):
        (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r,
        ) = self._build_batch_inputs(batch_size=6, n_times=60, n_rs=5)

        vmapped = jax.vmap(
            harmonica_transit_quad_ld,
            in_axes=(None, 0, None, None, None, None, None, 0, 0, 0),
        )(times, t0, period, a, inc, ecc, omega, u1, u2, r)

        batched = harmonica_transit_quad_ld_batch(
            times, t0, period, a, inc, ecc, omega, u1, u2, r
        )

        np.testing.assert_allclose(
            np.asarray(vmapped), np.asarray(batched), rtol=1e-12, atol=1e-12
        )

    def test_cuda_cpu_parity_quad_ld_if_available(self):
        cuda_devices = [d for d in jax.devices() if d.platform == "gpu"]
        if not cuda_devices:
            self.skipTest("CUDA device not available")
        cpu_devices = [d for d in jax.devices() if d.platform == "cpu"]
        if not cpu_devices:
            self.skipTest("CPU backend not visible; cannot run CPU/GPU parity")

        (
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r,
        ) = self._build_batch_inputs(batch_size=3, n_times=25, n_rs=3)

        cpu_fn = jax.jit(
            lambda: harmonica_transit_quad_ld_batch(
                times, t0, period, a, inc, ecc, omega, u1, u2, r
            ),
            backend="cpu",
        )
        gpu_fn = jax.jit(
            lambda: harmonica_transit_quad_ld_batch(
                times, t0, period, a, inc, ecc, omega, u1, u2, r
            ),
            backend="gpu",
        )

        cpu_flux = np.asarray(cpu_fn())
        gpu_flux = np.asarray(gpu_fn())

        np.testing.assert_allclose(cpu_flux, gpu_flux, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
