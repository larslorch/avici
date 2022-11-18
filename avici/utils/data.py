import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in log2")

import numpy as onp

"""Standardization"""

def _onp_cpm_standardizer(x):
    # compute library sizes (sum of row)
    x_libsize = x.sum(-1, keepdims=True)

    # divide each cell by library size and multiply by 10^6
    # will yield nan for rows with zero expression and for zero expression entries
    log2cpm = onp.where(onp.isclose(x, 0.0), onp.nan, onp.log2(x / (x_libsize * 1e-6)))
    return log2cpm


def _onp_cpm_shift_scale(x, shift, scale):
    # shift and scale
    x = (x - onp.where(onp.isnan(shift), 0.0, shift)) / onp.where(onp.isnan(scale), 1.0, scale)

    # set nans (set for all-zero rows, i.e. zero libsize) to minimum (i.e. to zero)
    # catch empty arrays (when x_int is empty and has axis n=0)
    if not x.size == 0:
        x = onp.where(onp.isnan(x), 0.0, x)

    return x


def _onp_standardize_count(x_obs, x_int, x_obs_ho, x_int_ho):
    """
    log2 CPM normalization for gene expression count data
    https://bioconductor.org/packages/release/bioc/vignettes/edgeR/inst/doc/edgeRUsersGuide.pdf
    https://rdrr.io/bioc/edgeR/src/R/cpm.R
    http://luisvalesilva.com/datasimple/rna-seq_units.html

    log2 CPM(mat) = log2( 10^6 * mat / libsize ) where libsize = mat.sum(1) (i.e. sum over genes)

    Why log2(1+cpm) is a bad idea https://support.bioconductor.org/p/107719/
    Specific scaling https://support.bioconductor.org/p/59846/#59917

    """
    # cpm normalization
    x_obs[..., 0] = _onp_cpm_standardizer(x_obs[..., 0])
    x_int[..., 0] = _onp_cpm_standardizer(x_int[..., 0])

    # subtract min (~robust global median) and divide by global std dev
    global_ref_x = onp.concatenate([x_obs[..., 0], x_int[..., 0]], axis=-2)
    global_min = onp.nanmin(global_ref_x, axis=(-1, -2), keepdims=True)
    global_std = onp.nanstd(global_ref_x, axis=(-1, -2), keepdims=True)

    x_obs[..., 0] = _onp_cpm_shift_scale(x_obs[..., 0], global_min, global_std)
    x_int[..., 0] = _onp_cpm_shift_scale(x_int[..., 0], global_min, global_std)

    if x_obs_ho is not None and x_int_ho is not None:
        x_obs_ho[..., 0] = _onp_cpm_standardizer(x_obs_ho[..., 0])
        x_int_ho[..., 0] = _onp_cpm_standardizer(x_int_ho[..., 0])
        x_obs_ho[..., 0] = _onp_cpm_shift_scale(x_obs_ho[..., 0], global_min, global_std)
        x_int_ho[..., 0] = _onp_cpm_shift_scale(x_int_ho[..., 0], global_min, global_std)

    return x_obs, x_int, x_obs_ho, x_int_ho


def _onp_standardize_default(x_obs, x_int, x_obs_ho, x_int_ho):
    # default z-standardization
    ref_x = onp.concatenate([x_obs[..., 0], x_int[..., 0]], axis=-2)
    mean = ref_x.mean(-2, keepdims=True)
    std = ref_x.std(-2, keepdims=True)
    x_obs[..., 0] = (x_obs[..., 0] - mean) / onp.where(std == 0.0, 1.0, std)
    x_int[..., 0] = (x_int[..., 0] - mean) / onp.where(std == 0.0, 1.0, std)

    if x_obs_ho is not None and x_int_ho is not None:
        x_obs_ho[..., 0] = (x_obs_ho[..., 0] - mean) / onp.where(std == 0.0, 1.0, std)
        x_int_ho[..., 0] = (x_int_ho[..., 0] - mean) / onp.where(std == 0.0, 1.0, std)

    return x_obs, x_int, x_obs_ho, x_int_ho


def onp_standardize_data(data):
    """
    Standardize data pytree
    Heldout data is standardized using the inference data statistics
    """
    x_obs = data["x_obs"]
    x_int = data["x_int"]

    x_obs_ho = data.get("x_heldout_obs", None)
    x_int_ho = data.get("x_heldout_int", None)

    assert (x_obs.shape[-1] == 1 or x_obs.shape[-1] == 2) \
       and (x_int.shape[-1] == 1 or x_int.shape[-1] == 2), \
        f"Assume concat 3D shape but got: x_obs {x_obs.shape} and x_int {x_int.shape}"

    if data["is_count_data"]:
        x_obs, x_int, x_obs_ho, x_int_ho = _onp_standardize_count(x_obs, x_int, x_obs_ho, x_int_ho)
    else:
        x_obs, x_int, x_obs_ho, x_int_ho = _onp_standardize_default(x_obs, x_int, x_obs_ho, x_int_ho)

    if x_obs_ho is not None and x_int_ho is not None:
        return {**data, "x_obs": x_obs, "x_int": x_int, "x_heldout_obs": x_obs_ho, "x_heldout_int": x_int_ho}
    else:
        return {**data, "x_obs": x_obs, "x_int": x_int}


"""Data batching"""

def onp_subbatch(rng, data, n_obs, n_int):
    """Sample random subbatch from observations `x_obs` and `x_int` in data"""
    x_obs = data["x_obs"]
    x_int = data["x_int"]
    assert "x_heldout_obs" not in data and "x_heldout_int" not in data

    if n_obs > x_obs.shape[-3]:
        warnings.warn(f"Warning: Larger batch requested than available in subbatch. "
                      f"Got `n_obs`={n_obs} and `x_obs` shape {x_obs.shape}. Will just use `x_obs`")

    if n_int > x_int.shape[-3]:
        warnings.warn(f"Warning: Larger batch requested than available in subbatch. "
                      f"Got `n_int`={n_int} and `x_int` shape {x_int.shape}. Will just use `x_int`")

    if rng is None:
        x_obs = x_obs[..., :n_obs, :, :]
        x_int = x_int[..., :n_int, :, :]
    else:
        x_obs = x_obs[..., rng.permutation(x_obs.shape[-3])[:n_obs], :, :]
        x_int = x_int[..., rng.permutation(x_int.shape[-3])[:n_int], :, :]

    data = {**data, "x_obs": x_obs, "x_int": x_int}
    return data


def onp_bootstrap(rng, data):
    """Sample bootstrap data set from `data`"""
    x_obs = data["x_obs"]
    x_int = data["x_int"]
    n_total = x_obs.shape[0] + x_int.shape[0]
    assert "x_heldout_obs" not in data and "x_heldout_int" not in data

    bootstrap = rng.choice(n_total, size=n_total)
    bootstrap_obs = bootstrap[bootstrap < x_obs.shape[0]]
    bootstrap_int = bootstrap[bootstrap >= x_obs.shape[0]] - x_obs.shape[0]
    x_obs = x_obs[..., bootstrap_obs, :, :]
    x_int = x_int[..., bootstrap_int, :, :]
    assert x_obs.shape[0] + x_int.shape[0] == data["x_obs"].shape[0] + data["x_int"].shape[0], \
        f"The total number of observations in bootstrap data set changed. " \
        f"{x_obs.shape} {x_int.shape}  {data['x_obs'].shape} {data['x_int'].shape} "

    data = {**data, "x_obs": x_obs, "x_int": x_int}
    return data


