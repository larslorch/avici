import math
import numpy as onp
from collections import defaultdict
from avici.synthetic.utils import onp_standardize_data


def sample_recursive(*,
                     spec,
                     rng,
                     g,
                     toporder,
                     n_vars,
                     f,
                     nse,
                     is_count_data,
                     n_interv_vars=0,
                     interv_dist=None):

    """Ancestral sampling from DAG

    `f` is list of functions (mechanisms), one for each node.
    Each f[j] maps
        rng:
        x:              [n_obs, n_vars,]
        z_j:            [n_obs,]
        is_parent:      [n_vars,]
        is_intervened:  bool
    to observations [n_obs,]

    `f` is list of sampling functions (noise distributions), one for each node.
    Each nse[j] maps
        rng:
        x:              [n_obs, n_vars,]
        is_parent:      [n_vars,]
    to noise samples [n_obs,]

    """

    # sample target nodes for the interventions
    interv_targets = []

    simulate_observ_data = spec.n_observations_obs > 0
    if simulate_observ_data:
        interv_targets += [None]

    simulate_interv_data = spec.n_observations_int > 0
    if simulate_interv_data:
        assert n_interv_vars != 0, f"Need n_interv_vars != 0 to sample interventional data"
        assert interv_dist is not None, f"Need an interventional distribution to perform interventions"
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
            n_obs = spec.n_observations_obs

        else:
            # interventional
            data_type = "int"
            is_intervened = onp.eye(n_vars)[interv_target].astype(bool)
            n_obs = math.ceil(spec.n_observations_int / n_interv_vars)

        # ancestral sampling in topological order
        x = onp.zeros((n_obs, n_vars))
        for j in toporder:
            # sample noise
            z_j = nse[j](rng=rng, x=x, is_parent=g[:, j])

            # compute node given data, parents, noise, and intervention state
            x[:, j] = f[j](rng=rng, x=x, z_j=z_j, is_parent=g[:, j], is_intervened=is_intervened[j])

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
    x_obs = onp.stack([x_obs, x_obs_msk], axis=-1)[:spec.n_observations_obs, :, :]
    x_int = onp.stack([x_int, x_int_msk], axis=-1)[:spec.n_observations_int, :, :]

    assert x_obs.size != 0 or x_int.size != 0, f"Need to sample at least some observations; " \
                                               f"got shapes x_obs {x_obs.shape} x_int {x_int.shape}"

    # collect data
    data = dict(
        g=g,
        n_vars=n_vars,
        x_obs=x_obs,
        x_int=x_int,
        n_observations_obs=spec.n_observations_obs,
        n_observations_int=spec.n_observations_int,
        is_count_data=is_count_data,
    )
    data = onp_standardize_data(data) # only done here because z-standardization is a projection; do not do for count data
    return data

