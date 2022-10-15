import copy
import argparse
import re
import platform
from pathlib import Path
import json
import subprocess
import numpy as onp
import math

from avici.experiment.utils import load_data, timer, NumpyJSONEncoder, get_git_hash_long
from avici.synthetic.utils import onp_bootstrap, onp_standardize_data, onp_subbatch
from avici.utils.parse import load_methods_config

from avici.definitions import BASELINE_GIES, BASELINE_LINGAM,\
    BASELINE_GES, BASELINE_PC, BASELINE_DAGGNN, BASELINE_DCDI, BASELINE_GRANDAG, BASELINE_IGSP,\
    BASELINES_OBSERV, BASELINES_INTERV, RNG_ENTROPY_TEST, BASELINE_BOOTSTRAP,\
    BASELINE_DIBS
from avici.experiment.methods import run_ours, run_pc, run_ges, run_gies, run_lingam, run_daggnn, run_grandag,\
    run_dcdi, run_igsp, run_dibs

if __name__ == "__main__":
    """
    Runs methods on a data instance and creates predictions 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--path_results", type=Path, required=True)
    parser.add_argument("--path_data_root", type=Path)
    parser.add_argument("--data_id", type=int)
    parser.add_argument("--bootstrap_id", type=int)
    parser.add_argument("--path_methods_config", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--heldout_split", default=0.0, type=float)
    kwargs = parser.parse_args()

    # generate directory if it doesn't exist
    kwargs.path_results.mkdir(exist_ok=True, parents=True)
    (kwargs.path_results / "logs").mkdir(exist_ok=True, parents=True)

    # get cpu info
    cpu_model = "not detected"
    if platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                cpu_model = re.sub(".*model name.*:", "", line, 1)

    # load data
    assert kwargs.data_id is not None
    methods_config = load_methods_config(kwargs.path_methods_config, abspath=True)
    assert kwargs.method in methods_config,  f"{kwargs.method} not in config with keys {list(methods_config.keys())}"
    config = methods_config[kwargs.method]

    data_observ = load_data(kwargs.path_data_root / f"{kwargs.data_id}", observ_only=True)
    data_interv = load_data(kwargs.path_data_root / f"{kwargs.data_id}", observ_only=False)

    assert "x_heldout_obs" not in data_observ and "x_heldout_int" not in data_observ, "heldout data shouldn't be seen here"
    assert "x_heldout_obs" not in data_interv and "x_heldout_int" not in data_interv, "heldout data shouldn't be seen here"

    # bootstrap?
    is_bootstrap = kwargs.bootstrap_id is not None
    bootstrap_file_id = f"_b={kwargs.bootstrap_id}" if is_bootstrap else ""
    if is_bootstrap:
        bootstrap_rng = onp.random.default_rng(onp.random.SeedSequence(
            entropy=(RNG_ENTROPY_TEST, kwargs.seed, kwargs.bootstrap_id)))
        data_observ = onp_bootstrap(copy.deepcopy(bootstrap_rng), data_observ)
        data_interv = onp_bootstrap(copy.deepcopy(bootstrap_rng), data_interv)

    # run method and measure walltime
    ps = []
    base_run_name = f"{kwargs.method}_{kwargs.data_id}{bootstrap_file_id}.json"

    base_method_full = kwargs.method.split("__")[0] # catch hyperparameter calibration case where name differs
    base_method = base_method_full.split(BASELINE_BOOTSTRAP)[-1]  # catch bootstrap case
    assert (base_method_full.split(BASELINE_BOOTSTRAP)[0] == "") == is_bootstrap, \
        "Need to provide `bootstrap_id` for bootstrap methods"

    if base_method in BASELINES_OBSERV:
        standardized_data = onp_standardize_data(data_observ)
    elif base_method in BASELINES_INTERV:
        standardized_data = onp_standardize_data(data_interv)
    elif base_method == "ours":
        # ours standardizes the data just before the forward pass because of possible subselection
        pass
    else:
        raise KeyError(f"Unknown class of base method `{base_method}`")

    with timer() as walltime:

        # observational
        if base_method == BASELINE_GES:
            pred = run_ges(kwargs.seed, standardized_data, config)
            ps.append((base_run_name, standardized_data, pred))

        elif base_method == BASELINE_PC:
            pred = run_pc(kwargs.seed, standardized_data, config)
            ps.append((base_run_name, standardized_data, pred))

        elif base_method == BASELINE_LINGAM:
            pred = run_lingam(kwargs.seed, standardized_data, config)
            ps.append((base_run_name, standardized_data, pred))

        elif base_method == BASELINE_DAGGNN:
            pred = run_daggnn(kwargs.seed, standardized_data, config, heldout_split=kwargs.heldout_split)
            ps.append((base_run_name, standardized_data, pred))

        elif base_method == BASELINE_GRANDAG:
            pred = run_grandag(kwargs.seed, standardized_data, config, heldout_split=kwargs.heldout_split)
            ps.append((base_run_name, standardized_data, pred))

        # observational and interventional
        elif base_method == BASELINE_GIES:
            pred = run_gies(kwargs.seed, standardized_data, config)
            ps.append((base_run_name, standardized_data, pred))

        elif base_method == BASELINE_IGSP:
            pred = run_igsp(kwargs.seed, standardized_data, config)
            ps.append((base_run_name, standardized_data, pred))

        elif base_method == BASELINE_DCDI:
            pred = run_dcdi(kwargs.seed, standardized_data, config, heldout_split=kwargs.heldout_split)
            ps.append((base_run_name, standardized_data, pred))

        elif base_method == BASELINE_DIBS:
            pred = run_dibs(kwargs.seed, standardized_data, config, heldout_split=kwargs.heldout_split)
            ps.append((base_run_name, standardized_data, pred))

        elif base_method == "ours":

            # artificially limit available observations
            if "lim_n_obs" in config:
                ratio_obs_int = data_interv["x_obs"].shape[0] / (data_interv["x_obs"].shape[0] + data_interv["x_int"].shape[0])
                n_obs, n_int = math.floor(config["lim_n_obs"] * ratio_obs_int), math.ceil(config["lim_n_obs"] * (1 - ratio_obs_int))

                data_observ = onp_subbatch(None, data_interv, config["lim_n_obs"], 0)
                data_interv = onp_subbatch(None, data_interv, n_obs, n_int)

            pred_observ = run_ours(kwargs.seed, data_observ, kwargs.checkpoint_dir)
            pred_interv = run_ours(kwargs.seed, data_interv, kwargs.checkpoint_dir)

            name_observ = base_method + "-observ"
            name_interv = base_method + "-interv"
            if len(kwargs.method.split("__")) > 1:
                name_observ += "__" + "__".join(kwargs.method.split("__")[1:])
                name_interv += "__" + "__".join(kwargs.method.split("__")[1:])

            ps.append((f"{name_observ}_{kwargs.data_id}.json", data_observ, pred_observ))
            ps.append((f"{name_interv}_{kwargs.data_id}.json", data_interv, pred_interv))


        else:
            raise KeyError(f"Unknown method `{kwargs.method}`")

    t_finish = walltime() / 60.0 # mins

    """Save predictions"""
    for run_name, seen_data, p in ps:
        p["config"] = config
        p["cpu_model"] = cpu_model
        p["walltime"] = t_finish
        p["data_shape_observational"] =  seen_data["x_obs"].shape[0:2]
        p["data_shape_interventional"] = seen_data["x_int"].shape[0:2]
        p["checkpoint_dir"] = str(kwargs.checkpoint_dir)
        p["git_version"] = get_git_hash_long()

        with open(kwargs.path_results / run_name, "w") as file:
            json.dump(p, file, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

    print(f"{kwargs.descr}:  {kwargs.method} seed {kwargs.seed} data_id {kwargs.data_id} "
          f"{'' if kwargs.bootstrap_id is None else ' b=' + str(kwargs.bootstrap_id)} finished successfully.")