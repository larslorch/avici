import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import jax.numpy as jnp
import jax.random as random
import jax.lax as lax

"""Standardization"""

def _jax_cpm_standardizer(x):
    # compute library sizes (sum of row)
    x_libsize = x.sum(-1, keepdims=True)

    # divide each cell by library size and multiply by 10^6
    # will yield nan for rows with zero expression and for zero expression entries
    log2cpm = jnp.where(jnp.isclose(x, 0.0), jnp.nan, jnp.log2(x / (x_libsize * 1e-6)))
    return log2cpm


def _jax_cpm_shift_scale(x, shift, scale):
    # shift and scale
    x = (x - jnp.where(jnp.isnan(shift), 0.0, shift)) / jnp.where(jnp.isnan(scale), 1.0, scale)

    # set nans (set for all-zero rows, i.e. zero libsize) to minimum (i.e. to zero)
    # catch empty arrays (when x_int is empty and has axis n=0)
    if not x.size == 0:
        x = jnp.where(jnp.isnan(x), 0.0, x)

    return x


def _jax_standardize_count(x_obs, x_int):
    # cpm normalization
    x_obs = x_obs.at[..., 0].set(_jax_cpm_standardizer(x_obs[..., 0]))
    x_int = x_int.at[..., 0].set(_jax_cpm_standardizer(x_int[..., 0]))

    # subtract min (~robust global median) and divide by global std dev
    global_ref_x = jnp.concatenate([x_obs[..., 0], x_int[..., 0]], axis=-2)
    global_min = jnp.nanmin(global_ref_x, axis=(-1, -2), keepdims=True)
    global_std = jnp.nanstd(global_ref_x, axis=(-1, -2), keepdims=True)

    x_obs = x_obs.at[..., 0].set(_jax_cpm_shift_scale(x_obs[..., 0], global_min, global_std))
    x_int = x_int.at[..., 0].set(_jax_cpm_shift_scale(x_int[..., 0], global_min, global_std))

    return x_obs, x_int


def jax_standardize_count_simple(x):
    """_jax_standardize_count but with only one argument"""
    # cpm normalization
    x = x.at[..., 0].set(_jax_cpm_standardizer(x[..., 0]))

    # subtract min (~robust global median) and divide by global std dev
    global_ref_x = x[..., 0]
    global_min = jnp.nanmin(global_ref_x, axis=(-1, -2), keepdims=True)
    global_std = jnp.nanstd(global_ref_x, axis=(-1, -2), keepdims=True)

    x = x.at[..., 0].set(_jax_cpm_shift_scale(x[..., 0], global_min, global_std))
    return x


def _jax_standardize_default(x_obs, x_int):
    # default z-standardization
    ref_x = jnp.concatenate([x_obs[..., 0], x_int[..., 0]], axis=-2)
    mean = ref_x.mean(-2, keepdims=True)
    std = ref_x.std(-2, keepdims=True)
    x_obs_default = x_obs.at[..., 0].set((x_obs[..., 0] - mean) / jnp.where(std == 0.0, 1.0, std))
    x_int_default = x_int.at[..., 0].set((x_int[..., 0] - mean) / jnp.where(std == 0.0, 1.0, std))
    return x_obs_default, x_int_default


def jax_standardize_default_simple(x):
    """_jax_standardize_default but with only one argument"""
    # default z-standardization
    ref_x = x[..., 0]
    mean = ref_x.mean(-2, keepdims=True)
    std = ref_x.std(-2, keepdims=True)
    x = x.at[..., 0].set((x[..., 0] - mean) / jnp.where(std == 0.0, 1.0, std))
    return x


def jax_standardize_data(data):
    """Standardize observations `x_obs` and `x_int`"""
    x_obs = data["x_obs"]
    x_int = data["x_int"]

    assert (x_obs.shape[-1] == 1 or x_obs.shape[-1] == 2) \
       and (x_int.shape[-1] == 1 or x_int.shape[-1] == 2), \
        f"Assume concat 3D shape but got: x_obs {x_obs.shape} and x_int {x_int.shape}"

    x_obs_count, x_int_count = _jax_standardize_count(x_obs, x_int)
    x_obs_default, x_int_default = _jax_standardize_default(x_obs, x_int)
    return {
        **data,
        "x_obs": jnp.where(data["is_count_data"][..., None, None, None], x_obs_count, x_obs_default),
        "x_int": jnp.where(data["is_count_data"][..., None, None, None], x_int_count, x_int_default),
    }

def jax_standardize_x(x, is_count_data):
    """Standardize observations"""
    assert (x.shape[-1] == 1 or x.shape[-1] == 2),\
        f"Assume concat 3D shape but got: x {x.shape}"

    # TODO this is not a nice implementation; refactor jax_standardize_data to accept variable number of x fields in data
    dummy = jnp.zeros((*x.shape[:-3], 0, *x.shape[-2:]))
    data = jax_standardize_data({"x_obs": x, "x_int": dummy, "is_count_data": is_count_data})
    return data["x_obs"]


"""Data batching"""

def jax_get_train_x(key, batch, p_obs_only):
    key, subk = random.split(key)
    n_obs_and_int, n_int = batch["x_obs"].shape[-3], batch["x_int"].shape[-3]

    only_observational = random.bernoulli(key, p_obs_only)
    x = lax.cond(
        only_observational,
        # only observational data
        lambda _: batch["x_obs"], # already has N=n_obs + n_int
        # mix observational and interventional
        # select `n_obs` elements of `n_obs + n_int` available observational data
        lambda _: jnp.concatenate([
            batch["x_obs"][..., random.permutation(subk, n_obs_and_int)[:(n_obs_and_int - n_int)], :, :],
            batch["x_int"]
        ], axis=-3),
        operand=None,
    )

    key, subk = random.split(key)
    x = x[..., random.permutation(key, n_obs_and_int), :, :]
    x = jax_standardize_x(x, batch["is_count_data"])  # place 1/2 where jax data should be standardized; do not do anywhere else to avoid double-standardizing count data
    return x


def jax_get_x(batch):
    x = jnp.concatenate([batch["x_obs"], batch["x_int"]], axis=-3)
    x = jax_standardize_x(x, batch["is_count_data"]) # place 2/2 where jax data should be standardized; do not do anywhere else to avoid double-standardizing count data
    return x