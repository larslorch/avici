import sys
# import importlib.util
import importlib
import itertools
import warnings
from collections import defaultdict
from pathlib import Path, PosixPath

import avici.synthetic
from avici.definitions import YAML_CLASS, YAML_MODULES, REAL_DATASET_KEY
from avici.synthetic import SyntheticSpec, CustomClassWrapper
from avici.utils.load import load_yaml


def cartesian_dict(d):
    """
    Cartesian product of nested dict of lists
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


def _parse_config_tree(config, force_kwargs=None, module_paths=None):

    if type(config) in [dict, defaultdict]:

        # check if `config` specifies a class
        is_class = YAML_CLASS in config.keys()

        # class
        if is_class:
            # recurse for kwargs
            name = config[YAML_CLASS]
            kwargs = {k: _parse_config_tree(subconfig, force_kwargs=force_kwargs, module_paths=module_paths)
                      for k, subconfig in config.items() if k != YAML_CLASS}

            # check for class in existing module
            try:
                cls = getattr(avici.synthetic, name)
                new = cls(**kwargs)

            # if class does not exist:
            except AttributeError:
                if module_paths is None:
                    raise SyntaxError(f"{YAML_CLASS} `{name}` of data config is not defined in existing "
                                      f"or registered modules. \nSpelled correctly and specified additional module paths?")
                # parse as custom class
                # if the class actually does not exist, loading will fail inside `load_custom_modules`
                new = CustomClassWrapper(name=name, kwargs=kwargs, paths=module_paths)

        # if not a class, recurse
        else:
            new = {key: _parse_config_tree(subconfig, force_kwargs=force_kwargs, module_paths=module_paths)
                   for (key, subconfig) in config.items()}

            # forcing key-value assignment if passed (e.g. n_observations)
            if force_kwargs is not None:
                for key in filter(lambda k: k in force_kwargs.keys(), new.keys()):
                    new[key] = force_kwargs[key]

    elif type(config) == list:
        # recurse
        new = [_parse_config_tree(subconfig, force_kwargs=force_kwargs, module_paths=module_paths)
               for subconfig in config]

    else:
        assert (type(config) in [int, float, bool, str, PosixPath] or config is None), f"Unknown yaml entry `{config}`"
        new = config

    return new


def init_custom_classes(config):

    # check if `config` specifies a `CustomModuleWrapper`
    if type(config) == CustomClassWrapper:

        # recurse on kwargs
        kwargs = init_custom_classes(config.kwargs)

        # load additional modules
        loaded_modules = []
        for p in config.paths:
            module_name = f"avici.synthetic._custom_.{p.stem}"
            module_spec = importlib.util.spec_from_file_location(module_name, p)
            sys.modules[module_name] = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(sys.modules[module_name])
            loaded_modules.append(sys.modules[module_name])

        # instantiate custom class
        new = None
        for mod in loaded_modules:
            try:
                cls = getattr(mod, config.name)
                new = cls(**kwargs)
                break
            except AttributeError:
                pass

        if new is None:
            raise SyntaxError(f"{YAML_CLASS} `{config.name}` of data config is not defined in existing "
                              f"or registered modules. \nSpelled correctly and specified additional module paths?")

    # else recurse
    elif type(config) == SyntheticSpec:
        new = SyntheticSpec(**init_custom_classes(config._asdict()))

    elif type(config) in [dict, defaultdict]:
        new = {key: init_custom_classes(subconfig) for (key, subconfig) in config.items()}

    elif type(config) == list:
        new = [init_custom_classes(subconfig) for subconfig in config]

    elif callable(config):
        new = config

    else:
        assert (type(config) in [int, float, bool, str, PosixPath] or config is None), f"Unknown entry `{config}`"
        new = config

    return new


def load_data_config(path, force_kwargs=None, abspath=False, module_paths=None, load_modules=False):
    """Load yaml config for data specification"""

    config = load_yaml(Path(path).resolve(strict=True), abspath=abspath)
    if config is None:
        raise SyntaxError("`config` is None; make sure there are no false tabs and indents in the .yaml file")

    spec = {"data": {}}

    # add meta info (all but "data" key)
    for key, val in config.items():
        if key not in spec:
            spec[key] = val

    # additional module paths
    additional_paths = []
    if YAML_MODULES in spec:
        ps = spec[YAML_MODULES] if isinstance(spec[YAML_MODULES], list) else [spec[YAML_MODULES]]
        additional_paths += list(map(lambda p: Path(p).resolve(strict=True), ps))

    if module_paths is not None:
        ps = module_paths if isinstance(module_paths, list) else [module_paths]
        additional_paths += list(map(lambda p: Path(p).resolve(strict=True), ps))

    additional_paths = list(set(additional_paths)) if additional_paths else None

    # real data case
    if REAL_DATASET_KEY in spec:
        return spec

    # process data field by expanding options lists with cartesian product
    spec["data"]["train"] = list(cartesian_dict(config["data"]))
    spec["data"]["val"] = list(cartesian_dict(config["data"]))

    if len(spec['data']["train"]) >= 1000:
        warnings.warn(f"config defines {len(spec['data'])} generative models")

    # init functions in spec
    spec = _parse_config_tree(spec, force_kwargs=force_kwargs, module_paths=additional_paths)

    # load modules; need to do this inside multiprocessing workers, otherwise pickle breaks
    if load_modules:
        spec = init_custom_classes(spec)

    # instantiate Synthetic data objects
    for descr, config_testcases in spec["data"].items():
        spec["data"][descr] = [SyntheticSpec(**arg) for arg in config_testcases]

    return spec
