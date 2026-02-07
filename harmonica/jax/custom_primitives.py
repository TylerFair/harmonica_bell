import jax
import numpy as np
import jax.numpy as jnp
from jaxlib import xla_client
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jax._src.interpreters.mlir import ir, custom_call
from jaxlib.mlir.dialects import mhlo

from harmonica.core import bindings

# Enable double floating precision.
jax.config.update("jax_enable_x64", True)

MIN_ABS = 1e-9
SIGMOID_WIDTH = 0.1 * MIN_ABS
SAFE_SIGN_EPS = 1e-12


def ir_dtype(np_dtype):
    """Convert a NumPy dtype or Python type to an MLIR IR type."""
    np_dtype = np.dtype(np_dtype)
    if np_dtype == np.float32:
        return ir.F32Type.get()
    elif np_dtype == np.float64:
        return ir.F64Type.get()
    elif np_dtype == np.int32:
        return ir.IntegerType.get_signless(32)
    elif np_dtype == np.int64:
        return ir.IntegerType.get_signless(64)
    else:
        raise TypeError(f"Unsupported dtype: {np_dtype}")


def ir_constant(val):
    if not isinstance(val, np.generic):
        dtype = np.dtype(type(val))
    else:
        dtype = val.dtype
    arr = np.array(val, dtype=dtype).reshape(())
    attr = ir.DenseElementsAttr.get(arr)
    return mhlo.ConstantOp(attr).result


def _smooth_min_abs(x, min_abs=MIN_ABS, softness=SIGMOID_WIDTH):
    abs_x = jnp.abs(x)
    scale = (abs_x - min_abs) / softness
    blend = jax.nn.sigmoid(scale)
    safe_sign = x / jnp.sqrt(x**2 + SAFE_SIGN_EPS)
    return blend * x + (1 - blend) * min_abs * safe_sign


def _ensure_last_two_nonzero(r, min_tail=MIN_ABS, softness=SIGMOID_WIDTH):
    tail = r[-2:]
    abs_tail = jnp.abs(tail)
    blend = jax.nn.sigmoid((jnp.sum(abs_tail) - min_tail) / softness)
    safe_tail = jnp.sign(tail + SAFE_SIGN_EPS) * min_tail
    new_tail = blend * tail + (1 - blend) * safe_tail
    return r.at[-2:].set(new_tail)


def _prepare_single_args(times, params, r):
    times = jnp.atleast_1d(jnp.asarray(times, dtype=jnp.float64))
    if times.ndim > 1:
        raise ValueError("`times` must be a scalar or 1D array")
    n = times.shape[0]

    broadcasted_params = []
    for p in params:
        p = jnp.asarray(p, dtype=jnp.float64)
        if p.ndim == 0 or (p.ndim == 1 and p.shape[0] != n):
            p = jnp.broadcast_to(p, (n,))
        elif p.ndim != 1:
            raise ValueError(f"Unexpected parameter shape: {p.shape}")
        broadcasted_params.append(p)

    r = jnp.asarray(r, dtype=jnp.float64)
    if r.ndim == 1:
        if r.shape[0] % 2 == 0:
            r = jnp.append(r, 0.0)
        flip = r[0] < 0
        r = r.at[0].set(_smooth_min_abs(jnp.abs(r[0])))
        r = _ensure_last_two_nonzero(r)
    elif r.ndim == 2:
        if r.shape[1] % 2 == 0:
            r = jnp.pad(r, ((0, 0), (0, 1)), constant_values=0.0)
        flip = r[:, 0] < 0
        r = r.at[:, 0].set(_smooth_min_abs(jnp.abs(r[:, 0])))
        r = jax.vmap(_ensure_last_two_nonzero)(r)
    else:
        raise ValueError(f"`r` must be shape (k,) or (n, k); got {r.shape}")

    if r.ndim == 1:
        r_list = [jnp.broadcast_to(r[i], (n,)) for i in range(r.shape[0])]
    else:
        r_list = [r[:, i] for i in range(r.shape[1])]

    args = [jnp.asarray(arg, dtype=jnp.float64)
            for arg in (times, *broadcasted_params, *r_list)]
    return args, flip


def _prepare_batch_args(times, params, r):
    times = jnp.atleast_1d(jnp.asarray(times, dtype=jnp.float64))
    if times.ndim != 1:
        raise ValueError("`times` must have shape (T,) for batch APIs")

    r = jnp.asarray(r, dtype=jnp.float64)
    if r.ndim != 2:
        raise ValueError("`r` must have shape (B, K) for batch APIs")

    if r.shape[1] % 2 == 0:
        r = jnp.pad(r, ((0, 0), (0, 1)), constant_values=0.0)

    b = r.shape[0]
    flip = r[:, 0] < 0
    r = r.at[:, 0].set(_smooth_min_abs(jnp.abs(r[:, 0])))
    r = jax.vmap(_ensure_last_two_nonzero)(r)

    broadcasted_params = []
    for p in params:
        p = jnp.asarray(p, dtype=jnp.float64)
        if p.ndim == 0:
            p = jnp.broadcast_to(p, (b,))
        elif p.ndim == 1:
            if p.shape[0] != b:
                raise ValueError(
                    f"Batch parameter length mismatch: got {p.shape[0]} expected {b}"
                )
        else:
            raise ValueError(f"Unexpected batched parameter shape: {p.shape}")
        broadcasted_params.append(p)

    r_list = [r[:, i] for i in range(r.shape[1])]
    args = [jnp.asarray(arg, dtype=jnp.float64)
            for arg in (times, *broadcasted_params, *r_list)]
    return args, flip


def _sanitize_outputs(flux, rest):
    flux = jnp.nan_to_num(flux, nan=1.0, posinf=1.0, neginf=1.0)
    if rest:
        rest = [jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                for x in rest]
    return flux, rest


def _harmonica_transit_common(primitive_fn, times, params, r):
    args, flip = _prepare_single_args(times, params, r)
    flux, *rest = primitive_fn(*args)
    flux, rest = _sanitize_outputs(flux, rest)
    flux = jnp.where(flip, 2.0 - flux, flux)
    return (flux, *rest)


def _harmonica_transit_batch_common(primitive_fn, times, params, r):
    args, flip = _prepare_batch_args(times, params, r)
    flux, *rest = primitive_fn(*args)
    flux, rest = _sanitize_outputs(flux, rest)
    flux = jnp.where(flip[:, None], 2.0 - flux, flux)
    return (flux, *rest)


@jax.jit
def harmonica_transit_quad_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                              u1=0., u2=0., r=jnp.array([0.1])):
    return _harmonica_transit_common(
        jax_light_curve_quad_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2],
        r,
    )[0]


@jax.jit
def harmonica_transit_quad_ld_batch(times, t0, period, a, inc, ecc=0., omega=0.,
                                    u1=0., u2=0., r=jnp.array([[0.1]])):
    """Batched quadratic LD transit model.

    Parameters
    ----------
    times : ndarray
        Shared 1D time grid with shape (T,).
    t0, period, a, inc, ecc, omega, u1, u2 : float or ndarray
        Scalar or per-light-curve arrays with shape (B,).
    r : ndarray
        Batched transmission-string coefficients with shape (B, K).

    Returns
    -------
    flux : ndarray
        Batched flux array with shape (B, T).
    """
    return _harmonica_transit_batch_common(
        jax_light_curve_quad_ld_batch_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2],
        r,
    )[0]


@jax.jit
def _quad_ld_flux_and_derivatives(times, t0, period, a, inc, ecc, omega,
                                  u1, u2, r):
    return _prepare_args_and_call_primitive(
        jax_light_curve_quad_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2],
        r,
    )


@jax.jit
def _quad_ld_flux_and_derivatives_batch(times, t0, period, a, inc, ecc, omega,
                                        u1, u2, r):
    return _prepare_args_and_call_primitive_batch(
        jax_light_curve_quad_ld_batch_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2],
        r,
    )


def jax_light_curve_quad_ld_prim(times, *params):
    return jax_light_curve_quad_ld_p.bind(times, *params)


def jax_light_curve_quad_ld_batch_prim(times, *params):
    return jax_light_curve_quad_ld_batch_p.bind(times, *params)


def jax_light_curve_quad_ld_abstract_eval(abstract_times, *abstract_params):
    abstract_model_eval = jax.core.ShapedArray(
        abstract_times.shape, abstract_times.dtype
    )
    n_params = len(abstract_params)
    abstract_model_derivatives = jax.core.ShapedArray(
        tuple(abstract_times.shape) + (n_params,), abstract_times.dtype
    )
    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_quad_ld_batch_abstract_eval(abstract_times, *abstract_params):
    if len(abstract_params) == 0:
        raise ValueError("Batched primitive expects at least one parameter input")
    b = abstract_params[0].shape[0]
    t = abstract_times.shape[0]
    n_params = len(abstract_params)
    abstract_model_eval = jax.core.ShapedArray((b, t), abstract_times.dtype)
    abstract_model_derivatives = jax.core.ShapedArray(
        (b, t, n_params), abstract_times.dtype
    )
    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_quad_ld_xla_translation(ctx, timesc, *paramssc):
    timesc_aval = ctx.avals_in[0]
    data_type = timesc_aval.dtype
    shape = timesc_aval.shape

    n_times = int(np.prod(shape))
    n_times_const = ir_constant(n_times)

    n_rs = len(paramssc) - 6 - 2
    n_rs_const = ir_constant(n_rs)

    output_shape_model_eval = ir.RankedTensorType.get((n_times,), ir_dtype(data_type))
    shape_derivatives = (n_times, 6 + 2 + n_rs)
    output_shape_model_derivatives = ir.RankedTensorType.get(
        shape_derivatives, ir_dtype(data_type)
    )

    return custom_call(
        b"jax_light_curve_quad_ld",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[n_times_const, n_rs_const, timesc, *paramssc],
        api_version=1,
        operand_layouts=[(), (), list(reversed(range(len(shape))))]
        + [list(reversed(range(len(shape))))] * len(paramssc),
        result_layouts=[[0], [1, 0]],
    ).results


def jax_light_curve_quad_ld_batch_xla_translation(ctx, timesc, *paramssc):
    times_aval = ctx.avals_in[0]
    data_type = times_aval.dtype
    n_times = int(np.prod(times_aval.shape))
    batch_size = int(ctx.avals_in[1].shape[0])

    n_rs = len(paramssc) - 6 - 2

    batch_size_const = ir_constant(batch_size)
    n_times_const = ir_constant(n_times)
    n_rs_const = ir_constant(n_rs)

    output_shape_model_eval = ir.RankedTensorType.get(
        (batch_size, n_times), ir_dtype(data_type)
    )
    shape_derivatives = (batch_size, n_times, 6 + 2 + n_rs)
    output_shape_model_derivatives = ir.RankedTensorType.get(
        shape_derivatives, ir_dtype(data_type)
    )

    return custom_call(
        b"jax_light_curve_quad_ld_batch",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[batch_size_const, n_times_const, n_rs_const, timesc, *paramssc],
        api_version=1,
        operand_layouts=[(), (), (), [0]] + [[0]] * len(paramssc),
        result_layouts=[[1, 0], [2, 1, 0]],
    ).results


def jax_light_curve_quad_ld_value_and_jvp(arg_values, arg_tangents):
    times, *args = arg_values
    _, *dargs = arg_tangents

    f, df_dz = jax_light_curve_quad_ld_prim(times, *args)

    df = 0.0
    for idx_pd, pd in enumerate(dargs):
        if isinstance(pd, ad.Zero):
            continue
        df += pd * df_dz[..., idx_pd]

    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


def jax_light_curve_quad_ld_batch_value_and_jvp(arg_values, arg_tangents):
    times, *args = arg_values
    _, *dargs = arg_tangents

    f, df_dz = jax_light_curve_quad_ld_batch_prim(times, *args)

    df = jnp.zeros_like(f)
    for idx_pd, pd in enumerate(dargs):
        if isinstance(pd, ad.Zero):
            continue
        df += pd[:, None] * df_dz[..., idx_pd]

    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


@jax.jit
def harmonica_transit_nonlinear_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                                   u1=0., u2=0., u3=0., u4=0.,
                                   r=jnp.array([0.1])):
    return _harmonica_transit_common(
        jax_light_curve_nonlinear_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2, u3, u4],
        r,
    )[0]


@jax.jit
def harmonica_transit_nonlinear_ld_batch(times, t0, period, a, inc, ecc=0., omega=0.,
                                         u1=0., u2=0., u3=0., u4=0.,
                                         r=jnp.array([[0.1]])):
    return _harmonica_transit_batch_common(
        jax_light_curve_nonlinear_ld_batch_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2, u3, u4],
        r,
    )[0]


@jax.jit
def _nonlinear_ld_flux_and_derivatives(times, t0, period, a, inc, ecc, omega,
                                       u1, u2, u3, u4, r):
    return _prepare_args_and_call_primitive(
        jax_light_curve_nonlinear_ld_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2, u3, u4],
        r,
    )


@jax.jit
def _nonlinear_ld_flux_and_derivatives_batch(times, t0, period, a, inc, ecc,
                                             omega, u1, u2, u3, u4, r):
    return _prepare_args_and_call_primitive_batch(
        jax_light_curve_nonlinear_ld_batch_prim,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2, u3, u4],
        r,
    )


def jax_light_curve_nonlinear_ld_prim(times, *params):
    return jax_light_curve_nonlinear_ld_p.bind(times, *params)


def jax_light_curve_nonlinear_ld_batch_prim(times, *params):
    return jax_light_curve_nonlinear_ld_batch_p.bind(times, *params)


def jax_light_curve_nonlinear_ld_abstract_eval(abstract_times, *abstract_params):
    abstract_model_eval = jax.core.ShapedArray(
        abstract_times.shape, abstract_times.dtype
    )
    n_params = len(abstract_params)
    abstract_model_derivatives = jax.core.ShapedArray(
        tuple(abstract_times.shape) + (n_params,), abstract_times.dtype
    )
    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_nonlinear_ld_batch_abstract_eval(abstract_times, *abstract_params):
    if len(abstract_params) == 0:
        raise ValueError("Batched primitive expects at least one parameter input")
    b = abstract_params[0].shape[0]
    t = abstract_times.shape[0]
    n_params = len(abstract_params)
    abstract_model_eval = jax.core.ShapedArray((b, t), abstract_times.dtype)
    abstract_model_derivatives = jax.core.ShapedArray(
        (b, t, n_params), abstract_times.dtype
    )
    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_nonlinear_xla_translation(ctx, timesc, *paramssc):
    timesc_aval = ctx.avals_in[0]
    data_type = timesc_aval.dtype
    shape = timesc_aval.shape

    n_times = int(np.prod(shape))
    n_times_const = ir_constant(n_times)

    n_rs = len(paramssc) - 6 - 4
    n_rs_const = ir_constant(n_rs)

    output_shape_model_eval = ir.RankedTensorType.get((n_times,), ir_dtype(data_type))
    shape_derivatives = (n_times, 6 + 4 + n_rs)
    output_shape_model_derivatives = ir.RankedTensorType.get(
        shape_derivatives, ir_dtype(data_type)
    )

    return custom_call(
        b"jax_light_curve_nonlinear_ld",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[n_times_const, n_rs_const, timesc, *paramssc],
        api_version=1,
        operand_layouts=[(), (), list(reversed(range(len(shape))))]
        + [list(reversed(range(len(shape))))] * len(paramssc),
        result_layouts=[[0], [1, 0]],
    ).results


def jax_light_curve_nonlinear_ld_batch_xla_translation(ctx, timesc, *paramssc):
    times_aval = ctx.avals_in[0]
    data_type = times_aval.dtype
    n_times = int(np.prod(times_aval.shape))
    batch_size = int(ctx.avals_in[1].shape[0])

    n_rs = len(paramssc) - 6 - 4

    batch_size_const = ir_constant(batch_size)
    n_times_const = ir_constant(n_times)
    n_rs_const = ir_constant(n_rs)

    output_shape_model_eval = ir.RankedTensorType.get(
        (batch_size, n_times), ir_dtype(data_type)
    )
    shape_derivatives = (batch_size, n_times, 6 + 4 + n_rs)
    output_shape_model_derivatives = ir.RankedTensorType.get(
        shape_derivatives, ir_dtype(data_type)
    )

    return custom_call(
        b"jax_light_curve_nonlinear_ld_batch",
        result_types=[output_shape_model_eval, output_shape_model_derivatives],
        operands=[batch_size_const, n_times_const, n_rs_const, timesc, *paramssc],
        api_version=1,
        operand_layouts=[(), (), (), [0]] + [[0]] * len(paramssc),
        result_layouts=[[1, 0], [2, 1, 0]],
    ).results


def jax_light_curve_nonlinear_ld_value_and_jvp(arg_values, arg_tangents):
    times, *args = arg_values
    _, *dargs = arg_tangents

    f, df_dz = jax_light_curve_nonlinear_ld_prim(times, *args)
    f = jnp.nan_to_num(f, nan=1.0, posinf=1.0, neginf=1.0)
    df_dz = jnp.nan_to_num(df_dz, nan=0.0, posinf=0.0, neginf=0.0)

    df = 0.0
    for idx_pd, pd in enumerate(dargs):
        if isinstance(pd, ad.Zero):
            continue
        df += pd * df_dz[..., idx_pd]

    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


def jax_light_curve_nonlinear_ld_batch_value_and_jvp(arg_values, arg_tangents):
    times, *args = arg_values
    _, *dargs = arg_tangents

    f, df_dz = jax_light_curve_nonlinear_ld_batch_prim(times, *args)
    f = jnp.nan_to_num(f, nan=1.0, posinf=1.0, neginf=1.0)
    df_dz = jnp.nan_to_num(df_dz, nan=0.0, posinf=0.0, neginf=0.0)

    df = jnp.zeros_like(f)
    for idx_pd, pd in enumerate(dargs):
        if isinstance(pd, ad.Zero):
            continue
        df += pd[:, None] * df_dz[..., idx_pd]

    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


def _register_custom_call_targets():
    names = [
        "jax_light_curve_quad_ld",
        "jax_light_curve_nonlinear_ld",
        "jax_light_curve_quad_ld_batch",
        "jax_light_curve_nonlinear_ld_batch",
    ]

    regs = bindings.jax_registrations()
    for name in names:
        xla_client.register_custom_call_target(name, regs[name], api_version=0)

    has_cuda_targets = False
    if hasattr(bindings, "jax_registrations_cuda"):
        cuda_regs = bindings.jax_registrations_cuda()
        if len(cuda_regs) > 0:
            # Different JAX/jaxlib builds may identify the GPU runtime with
            # different backend names. Register all relevant aliases.
            gpu_platform_aliases = ("cuda", "gpu", "CUDA")
            for name in names:
                if name in cuda_regs:
                    for platform in gpu_platform_aliases:
                        xla_client.register_custom_call_target(
                            name, cuda_regs[name], platform=platform, api_version=0
                        )
            has_cuda_targets = True

    return has_cuda_targets


def _prepare_args_and_call_primitive(primitive_fn, times, param_list, r):
    return _harmonica_transit_common(primitive_fn, times, param_list, r)


def _prepare_args_and_call_primitive_batch(primitive_fn, times, param_list, r):
    return _harmonica_transit_batch_common(primitive_fn, times, param_list, r)


def _generic_multiple_results_batcher(primitive_fn):
    def _batcher(batched_args, batch_dims):
        if all(dim is None for dim in batch_dims):
            out = primitive_fn(*batched_args)
            return out, tuple(None for _ in out)

        batch_size = None
        for arg, dim in zip(batched_args, batch_dims):
            if dim is not None:
                moved = jnp.moveaxis(arg, dim, 0)
                batch_size = moved.shape[0]
                break

        aligned = []
        for arg, dim in zip(batched_args, batch_dims):
            if dim is None:
                aligned.append(jnp.broadcast_to(arg, (batch_size,) + arg.shape))
            else:
                aligned.append(jnp.moveaxis(arg, dim, 0))

        out = jax.lax.map(lambda xs: primitive_fn(*xs), tuple(aligned))
        return out, tuple(0 for _ in out)

    return _batcher


HAS_CUDA_TARGETS = _register_custom_call_targets()

# Create primitives.
jax_light_curve_quad_ld_p = Primitive("jax_light_curve_quad_ld")
jax_light_curve_quad_ld_p.multiple_results = True
jax_light_curve_quad_ld_p.def_abstract_eval(jax_light_curve_quad_ld_abstract_eval)

jax_light_curve_nonlinear_ld_p = Primitive("jax_light_curve_nonlinear_ld")
jax_light_curve_nonlinear_ld_p.multiple_results = True
jax_light_curve_nonlinear_ld_p.def_abstract_eval(jax_light_curve_nonlinear_ld_abstract_eval)

jax_light_curve_quad_ld_batch_p = Primitive("jax_light_curve_quad_ld_batch")
jax_light_curve_quad_ld_batch_p.multiple_results = True
jax_light_curve_quad_ld_batch_p.def_abstract_eval(
    jax_light_curve_quad_ld_batch_abstract_eval
)

jax_light_curve_nonlinear_ld_batch_p = Primitive(
    "jax_light_curve_nonlinear_ld_batch"
)
jax_light_curve_nonlinear_ld_batch_p.multiple_results = True
jax_light_curve_nonlinear_ld_batch_p.def_abstract_eval(
    jax_light_curve_nonlinear_ld_batch_abstract_eval
)

# CPU lowerings.
mlir.register_lowering(
    jax_light_curve_quad_ld_p,
    jax_light_curve_quad_ld_xla_translation,
    platform="cpu",
)
mlir.register_lowering(
    jax_light_curve_nonlinear_ld_p,
    jax_light_curve_nonlinear_xla_translation,
    platform="cpu",
)
mlir.register_lowering(
    jax_light_curve_quad_ld_batch_p,
    jax_light_curve_quad_ld_batch_xla_translation,
    platform="cpu",
)
mlir.register_lowering(
    jax_light_curve_nonlinear_ld_batch_p,
    jax_light_curve_nonlinear_ld_batch_xla_translation,
    platform="cpu",
)

# CUDA lowerings are optional and enabled only when targets are available.
if HAS_CUDA_TARGETS:
    for platform in ("cuda", "gpu"):
        mlir.register_lowering(
            jax_light_curve_quad_ld_p,
            jax_light_curve_quad_ld_xla_translation,
            platform=platform,
        )
        mlir.register_lowering(
            jax_light_curve_nonlinear_ld_p,
            jax_light_curve_nonlinear_xla_translation,
            platform=platform,
        )
        mlir.register_lowering(
            jax_light_curve_quad_ld_batch_p,
            jax_light_curve_quad_ld_batch_xla_translation,
            platform=platform,
        )
        mlir.register_lowering(
            jax_light_curve_nonlinear_ld_batch_p,
            jax_light_curve_nonlinear_ld_batch_xla_translation,
            platform=platform,
        )

# JVP rules.
ad.primitive_jvps[jax_light_curve_quad_ld_p] = jax_light_curve_quad_ld_value_and_jvp
ad.primitive_jvps[jax_light_curve_nonlinear_ld_p] = (
    jax_light_curve_nonlinear_ld_value_and_jvp
)
ad.primitive_jvps[jax_light_curve_quad_ld_batch_p] = (
    jax_light_curve_quad_ld_batch_value_and_jvp
)
ad.primitive_jvps[jax_light_curve_nonlinear_ld_batch_p] = (
    jax_light_curve_nonlinear_ld_batch_value_and_jvp
)

# Batching rules.
batching.primitive_batchers[jax_light_curve_quad_ld_p] = _generic_multiple_results_batcher(
    jax_light_curve_quad_ld_prim
)
batching.primitive_batchers[
    jax_light_curve_nonlinear_ld_p
] = _generic_multiple_results_batcher(jax_light_curve_nonlinear_ld_prim)
batching.primitive_batchers[
    jax_light_curve_quad_ld_batch_p
] = _generic_multiple_results_batcher(jax_light_curve_quad_ld_batch_prim)
batching.primitive_batchers[
    jax_light_curve_nonlinear_ld_batch_p
] = _generic_multiple_results_batcher(jax_light_curve_nonlinear_ld_batch_prim)
