from pathlib import Path
import numpy as onp

from avici.definitions import PROJECT_DIR
from avici.utils.parse import load_data_config
from avici.buffer import Sampler
from avici.utils.data import onp_standardize_data

def simulate_data(n, d, *, seed=0, n_interv=0, domain=None, path=None, module_paths=None):
    rng = onp.random.default_rng(onp.random.SeedSequence(entropy=seed))

    # load example domain specification
    if domain is not None:
        abspath = PROJECT_DIR / f"avici/config/examples/{domain}.yaml"
    elif path is not None:
        abspath = Path(path)
    else:
        raise KeyError("Specify either an an `avici.config.examples` domain or a path.")

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