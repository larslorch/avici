import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import functools
import itertools
from collections import defaultdict

from typing import Any, NamedTuple

from avici.definitions import YAML_FUNC, REAL_DATASET_KEY


import avici.synthetic.graph as module_graph
import avici.synthetic.distribution as module_distribution
import avici.synthetic.mechanism as module_mechanims
from avici.utils.yaml import load_config


class Synthetic(NamedTuple):

    g: Any
    mechanism: Any

    n_observations_obs: int
    n_observations_int: int = 0

    name: str = None


def cartesian_dict(d):
    """
    Cartesian product of nested dict/defaultdicts of lists
    Example:

    d = {'s1': {'a': 0,
                'b': [0, 1, 2]},
         's2': {'c': [0, 1],
                'd': [0, 1]}}

    yields
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 0, 'd': 0}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 0, 'd': 1}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 1, 'd': 0}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 1, 'd': 1}}
        {'s1': {'a': 0, 'b': 1}, 's2': {'c': 0, 'd': 0}}
        ...
    """
    if type(d) in [dict, defaultdict]:
        keys, values = d.keys(), d.values()
        for c in itertools.product(*(cartesian_dict(v) for v in values)):
            yield dict(zip(keys, c))
    elif type(d) == list:
        for c in d:
            yield from cartesian_dict(c)
    else:
        yield d



def _parse_functions_in_config(config):

    if type(config) in [dict, defaultdict]:

        # check if `config` specifies a function
        is_function = YAML_FUNC in config.keys()

        if is_function:
            f = None
            kwargs = {}

            for key, subconfig in config.items():
                # check for function named `config[YAML_FUNC]` in the three modules
                if key == YAML_FUNC:
                    for mod in [module_graph, module_distribution, module_mechanims]:
                        try:
                            f = getattr(mod, config[YAML_FUNC])
                            break
                        except AttributeError:
                            pass
                    assert f is not None, f"__func__ `{subconfig}` not defined in modules. Spelled correctly?"

                # update arguments of function
                else:
                    kwargs.update({key: _parse_functions_in_config(subconfig)})

            # apply args
            new = functools.partial(f, **kwargs)

        # if not a function, recurse
        else:
            new = {key: _parse_functions_in_config(subconfig) for (key, subconfig) in config.items()}

    elif type(config) == list:
        # recurse
        new = [_parse_functions_in_config(subconfig) for subconfig in config]

    else:
        assert (type(config) in [int, float, bool, str] or config is None), f"Unknown yaml entry `{config}`"
        new = config

    return new


def load_data_config(path, abspath=False, verbose=True):
    """Load yaml config for data specification"""

    config = load_config(path, abspath=abspath)
    if config is None:
        return None

    spec = {"data": {}}

    # add meta info (all but "data")
    for key, val in config.items():
        if key not in spec:
            spec[key] = val

    # real data case
    if REAL_DATASET_KEY in spec:
        return spec

    # process data field
    for descr, config_testcase in config["data"].items():
        spec["data"][descr] = list(cartesian_dict(config_testcase))
        if verbose:
            print(f"config `{descr}`: {len(spec['data'][descr])} generative models")

    # check that n_observations_obs >= 2x n_observations_int
    for descr, config_testcases in spec["data"].items():
        sufficient_observ_data = [arg["n_observations_obs"] >= 2 * arg["n_observations_int"]
                                  for arg in config_testcases]
        if not all(sufficient_observ_data):
            warnings.warn("Not enough observational data sampled when evaluating. Sample 2x more observational "
                          "than interventional data so that we can compare both classes of methods"
                          "with the same number of observations.\n")

    # init functions in spec
    spec = _parse_functions_in_config(spec)

    # instantiate Synthetic data objects
    for descr, config_testcases in spec["data"].items():
        spec["data"][descr] = [Synthetic(**arg) for arg in config_testcases]

    return spec


def get_list_leaves(d):
    if type(d) in [dict, defaultdict]:
        opt = {}
        for k, v in d.items():
            if type(v) == list:
                opt[k] = v
        if opt:
            return cartesian_dict(opt)
        else:
            return []

def load_methods_config(path, abspath=False, verbose=True):
    """Load yaml config for method specification"""

    config = load_config(path, abspath=abspath)
    if config is None:
        return None

    # if grid validation, expand options and update name of resulting methods
    if "__grid__" in config:
        del config["__grid__"]
        expanded_config = {}
        for method, hparams in config.items():
            all_full_settings = list(cartesian_dict(hparams))
            all_options = list(get_list_leaves(hparams))

            # match expanded set of full settings with differences in leaves for unique naming
            if not all_options:
                # only one setting
                assert len(all_full_settings) == 1
                expanded_config[method] = next(iter(all_full_settings))

            else:

                for option in all_options:
                    match = list(filter(lambda setting: all([setting[k] == v for k, v in option.items()]),
                                        all_full_settings))
                    assert len(match) == 1
                    option_descr = '-'.join([f"{k}={v}" for k, v in option.items()])
                    expanded_config[f"{method}__{option_descr}"] = next(iter(match))

        return expanded_config
    else:
        return config
