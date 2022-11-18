from pathlib import Path
import numpy as onp

from avici.definitions import PROJECT_DIR
from avici.utils.parse import load_data_config
from avici.buffer import Sampler
from avici.utils.data import onp_standardize_data


def simulate_data( d, n, *, n_interv=0, seed=0, domain=None, path=None, module_paths=None):
    """
    Helper function for simulating data from a pre-specified `domain` or a YAML domain configuration file.

    Args:
        d (int): number of variables in the system to be simulated
        n (int): number of observational data points to be sampled
        n_interv (int): number of interventional data points to be sampled
        seed (int): random seed
        domain (str): specifier of domain to be simulated. Currently implemented options:
            `linear-gaussian-scm`, `rff-cauchy-scm`, `gene-ecoli` (all `.yaml` files inside `avici.config.examples`).
            Only one of `domain`
        path (str): path to YAML domain configuration, like the examples in `avici.config`
        module_paths (str): path (or list of paths) to additional modules used in the domain configuration file

    Returns:
        tuple: the function returns a 3-tuple of
            - g (ndarray): `[d, d]` causal graph of `d` variables
            - x (ndarray): `[n + n_interv, d]` data matrix containing `n + n_interv` observations of the `d` variables
            - interv (ndarray): `[n + n_interv, d]` binary matrix indicating which nodes were intervened upon

    """
    rng = onp.random.default_rng(onp.random.SeedSequence(entropy=seed))

    # load example domain specification
    if domain is not None:
        assert path is None, "Only specify one of `domain` and `path`"
        abspath = PROJECT_DIR / f"avici/config/examples/{domain}.yaml"
    elif path is not None:
        assert domain is None, "Only specify one of `domain` and `path`"
        abspath = Path(path)
    else:
        raise KeyError("Specify either an an `avici.config.examples` domain (YAML name) or a path to a YAML config.")

    if abspath.is_file():
        kwargs = dict(n_observations_obs=n, n_observations_int=n_interv)
        spec_tree = load_data_config(abspath, force_kwargs=kwargs, abspath=True,
                                     module_paths=module_paths, load_modules=True)["data"]
        spec = spec_tree[next(iter(spec_tree))]
    else:
        raise KeyError(f"`{abspath}` does not exist.")

    # sample and concatenate all data
    data = Sampler.generate_data(
        rng,
        n_vars=d,
        spec_list=spec,
    )
    x = onp.concatenate([data["x_obs"], data["x_int"]], axis=-3)

    # standardize only if not real-valued data
    data = onp_standardize_data(data) if not data["is_count_data"] else data

    if n_interv:
        return data["g"].astype(int), x[..., 0], x[..., 1]
    else:
        return data["g"].astype(int), x[..., 0]