import jax.numpy as jnp
import numpy as np
from harmonica.jax import harmonica_transit_quad_ld as transit
from jax import jit


def test_api_jax_quad_ld_simple():
    t0 = 5.0
    period = 10.0
    a = 10.0
    inc = jnp.array(89. * jnp.pi / 180.)
    ecc = 0.0
    omega = jnp.array(0.1 * jnp.pi / 180.)
    times = jnp.linspace(2.5, 7.5, 1000)
    ld_coeffs = dict(u1=0.1, u2=0.5)

    # Wrap the function inside the test to ensure fresh JIT trace
    f_transit = jit(transit)

    f = f_transit(times, t0, period, a, inc)
    assert f.shape == times.shape
    assert np.sum(np.isfinite(f)) == 1000

    f = f_transit(times, t0, period, a, inc, ecc, omega,
                  r=jnp.array([0.1, -0.003, 0.0]), **ld_coeffs)
    assert f.shape == times.shape
    assert np.sum(np.isfinite(f)) == times.size

    f = f_transit(times, t0, period, a, inc, ecc, omega,
                  r=jnp.array([0.1, -0.003, 0., 0., 0., 0., 0.001]), **ld_coeffs)

    isnan = ~jnp.isfinite(f)
    nan_indices = jnp.where(isnan)[0]
    time_indices = times[jnp.where(isnan)[0]]
    if isnan.any():
        print("NaNs at indices:", nan_indices)
        print("Times at NaNs:", time_indices)
        print("Values:", f[nan_indices])

    import matplotlib.pyplot as plt
    plt.plot(times, f, label='flux')
    plt.scatter(times[nan_indices], f[nan_indices], color='red', label='NaNs')
    plt.axvline(5.0, color='gray', linestyle='--', label='t0')
    plt.legend()
    plt.title("Flux curve with NaNs highlighted")
    plt.show()

    assert f.shape == times.shape
    assert np.sum(np.isfinite(f)) == times.size
