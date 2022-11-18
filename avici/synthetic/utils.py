import math
from collections import defaultdict

import numpy as onp

from avici.synthetic import Data
from avici.utils.graph import mat_to_toporder


def draw_rff_params(*, rng, d, length_scale, output_scale, n_rff):
    """Draws random instantiation of rffs"""
    # draw parameters
    ls = length_scale(rng, shape=(1,)).item() if callable(length_scale) else length_scale
    c = output_scale(rng, shape=(1,)).item() if callable(output_scale) else output_scale

    # draw rffs
    # [d, n_rff]
    omega_j = rng.normal(loc=0, scale=1.0 / ls, size=(d, n_rff))

    # [n_rff, ]
    b_j = rng.uniform(0, 2 * onp.pi, size=(n_rff,))

    # [n_rff, ]
    w_j = rng.normal(loc=0, scale=1.0, size=(n_rff,))

    return dict(
        c=c,
        omega=omega_j,
        b=b_j,
        w=w_j,
        n_rff=n_rff,
    )


def sample_recursive_scm(*,
                         rng,
                         n_observations_obs,
                         n_observations_int,
                         g,
                         f,
                         nse,
                         interv_dist,
                         n_interv_vars=0):
    """Ancestral sampling over a DAG

    Args:
        rng:
        n_observations_obs: number of observational data rows to be sampled
        n_observations_int: number of interventional data rows to be sampled
        g: adjacency matrix of the DAG of shape [n_vars, n_vars]
        f: list of functions (mechanisms), one for each node.
            Each f[j] maps: observation matrix `x` [n_obs, n_vars], noise vector `z` [n_obs,], and
            parent indicator `is_parent` [n_vars,] to the observations observed for
            node j [n_obs,]
        nse: list of class instances representing the noise distributions for each node, subclassing `NoiseModel` ABC
        interv_dist: Subclass of `DistributionModel` ABC for sampling intervention values
        n_interv_vars (optional): number of variables intervened upon (default is 0). If -1 is passed, all variables are intervened
            upon. For other integers, a set of intervened variables is randomly selected and interventional data
            is generated in equal proportion for each node based on the total number of `n_observations_int` data points

    Returns:
        dict containing `g`, `x_obs`, `x_int`, `n_vars`, `n_observations_obs`, `n_observations_int`, `is_count_data`
    """
    n_vars = g.shape[-1]
    toporder = mat_to_toporder(g)

    # sample target nodes for the interventions
    interv_targets = []

    simulate_observ_data = n_observations_obs > 0
    if simulate_observ_data:
        interv_targets += [None]

    simulate_interv_data = n_observations_int > 0
    if simulate_interv_data:
        assert n_interv_vars != 0, f"Need n_interv_vars != 0 to sample interventional data"
        if n_interv_vars == -1:
            n_interv_vars = n_vars
        interv_targets += sorted(rng.choice(n_vars, size=min(n_vars, n_interv_vars), replace=False).tolist())

    assert (n_interv_vars == -1) or (0 <= n_interv_vars <= n_vars),\
        f"Got `n_interv_vars` = {n_interv_vars} for `n_vars` = {n_vars}, which is invalid."

    # simulate data for observational data and for each interventional target
    data = defaultdict(lambda: defaultdict(list))
    for interv_target in interv_targets:

        if interv_target is None:
            # observational
            data_type = "obs"
            is_intervened = onp.zeros(n_vars).astype(bool)
            n_obs = n_observations_obs

        else:
            # interventional
            data_type = "int"
            is_intervened = onp.eye(n_vars)[interv_target].astype(bool)
            n_obs = math.ceil(n_observations_int / n_interv_vars)

        # ancestral sampling in topological order
        x = onp.zeros((n_obs, n_vars))
        for j in toporder:
            # sample noise
            z_j = nse[j](rng=rng, x=x, is_parent=g[:, j])

            # compute node given parents and noise or perform intervention state
            if is_intervened[j]:
                x[:, j] = interv_dist(rng, shape=z_j.shape)
            else:
                x[:, j] = f[j](x=x, z=z_j, is_parent=g[:, j])

        # generate intervention mask
        # [n_obs, n_vars] with True/False depending on whether node was intervened upon
        interv_mask = onp.tile(is_intervened, (x.shape[0], 1)).astype(onp.float32)

        data[data_type]["x"].append(x)
        data[data_type]["interv_mask"].append(interv_mask)


    # concatenate interventional data, interweaving rows to have balanced observation counts when clipping the end
    if simulate_observ_data:
        x_obs = onp.stack(data["obs"]["x"]).reshape(-1, n_vars, order="F")
        x_obs_msk = onp.stack(data["obs"]["interv_mask"]).reshape(-1, n_vars, order="F")
    else:
        x_obs = onp.zeros((0, n_vars))  # dummy
        x_obs_msk = onp.zeros((0, n_vars))  # dummy

    if simulate_interv_data:
        x_int = onp.stack(data["int"]["x"]).reshape(-1, n_vars, order="F")
        x_int_msk = onp.stack(data["int"]["interv_mask"]).reshape(-1, n_vars, order="F")
    else:
        x_int = onp.zeros((0, n_vars))  # dummy
        x_int_msk = onp.zeros((0, n_vars))  # dummy

    # clip number of observations to have invariant shape (in case n_obs doesn't evenly devide no. interv targets)
    # [n_observations, n_vars, 2]
    x_obs = onp.stack([x_obs, x_obs_msk], axis=-1)[:n_observations_obs, :, :]
    x_int = onp.stack([x_int, x_int_msk], axis=-1)[:n_observations_int, :, :]

    assert x_obs.size != 0 or x_int.size != 0, f"Need to sample at least some observations; " \
                                               f"got shapes x_obs {x_obs.shape} x_int {x_int.shape}"

    # collect data
    data = Data(
        x_obs=x_obs,
        x_int=x_int,
        is_count_data=False,
    )
    return data