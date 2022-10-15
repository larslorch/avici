import numpy as onp
import argparse
import json
import math
from pathlib import Path

from avici.experiment.utils import save_csv, to_dict
from avici.utils.parse import load_data_config
from avici.synthetic.buffer import Sampler

from avici.definitions import RNG_ENTROPY_TEST, RNG_ENTROPY_HPARAMS, \
    FILE_DATA_META, FILE_DATA_G, \
    FILE_DATA_X_OBS, FILE_DATA_X_INT, FILE_DATA_X_INT_INFO, \
    FILE_DATA_X_OBS_HELDOUT, FILE_DATA_X_INT_HELDOUT, FILE_DATA_X_INT_INFO_HELDOUT

if __name__ == "__main__":
    """
    Generates data for experiment
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--data_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--j", type=int, required=True)
    kwargs = parser.parse_args()

    data_config = load_data_config(kwargs.data_config_path, verbose=False)

    # determine whether test set or validation data
    assert len(data_config["data"]) == 1
    data_mode = next(iter(data_config["data"].keys()))

    if data_mode == "evaluation":
        rng_entropy = RNG_ENTROPY_TEST
    elif data_mode == "hyperparameter-tuning":
        rng_entropy = RNG_ENTROPY_HPARAMS
    else:
        raise ValueError(f"Unknown data mode `{data_mode}` for evaluation")

    rng = onp.random.default_rng(onp.random.SeedSequence(entropy=(rng_entropy, kwargs.j)))

    # in eval, make sure graph model classes are sampled in equal proportions
    spec_list = data_config["data"][data_mode]
    unique_graph_models = sorted(list(set(sp.g.func.__name__ for sp in spec_list)))
    graph_model_j = unique_graph_models[kwargs.j % len(unique_graph_models)]

    # filter spec_list for selected graph model for this seed
    spec_list_j = list(filter(lambda sp: sp.g.func.__name__ == graph_model_j, spec_list))

    # sample
    spec = spec_list_j[rng.choice(len(spec_list_j))]
    data = Sampler.generate_data(
        rng,
        n_vars=data_config["n_vars"],
        spec_list=None,
        spec=spec,
    )

    # write to file
    data_folder = kwargs.path_data / f"{kwargs.j}"
    data_folder.mkdir(exist_ok=True, parents=True)
    data_dtype = onp.int32 if data["is_count_data"] else onp.float32
    ho_obs = math.ceil(0.5 * data["x_obs"].shape[0])
    ho_int = math.ceil(0.5 * data["x_int"].shape[0])

    save_csv(data["g"].astype(onp.int32), data_folder / FILE_DATA_G)

    save_csv(data["x_obs"][:ho_obs, :, 0].astype(data_dtype), data_folder / FILE_DATA_X_OBS)
    save_csv(data["x_int"][:ho_int, :, 0].astype(data_dtype), data_folder / FILE_DATA_X_INT)
    save_csv(data["x_int"][:ho_int, :, 1].astype(onp.int32),  data_folder / FILE_DATA_X_INT_INFO)

    save_csv(data["x_obs"][ho_obs:, :, 0].astype(data_dtype), data_folder / FILE_DATA_X_OBS_HELDOUT)
    save_csv(data["x_int"][ho_int:, :, 0].astype(data_dtype), data_folder / FILE_DATA_X_INT_HELDOUT)
    save_csv(data["x_int"][ho_int:, :, 1].astype(onp.int32),  data_folder / FILE_DATA_X_INT_INFO_HELDOUT)

    meta_info_path = data_folder / FILE_DATA_META

    # generate directory if it doesn't exist
    meta_info_path.parent.mkdir(exist_ok=True, parents=True)

    with open(meta_info_path, "w") as file:
        meta_info = {
            "data_mode": data_mode,
            "is_count_data": data["is_count_data"],
            "n_vars": data_config["n_vars"],
            "n_data_observational": ho_obs,
            "n_data_interventional": ho_int,
            "n_heldout_data_observational": data["x_obs"].shape[0] - ho_obs,
            "n_heldout_data_interventional": data["x_int"].shape[0] - ho_int,
            "model": to_dict(spec),
        }
        json.dump(meta_info, file, indent=4, sort_keys=True)

    print(f"{kwargs.descr}: {kwargs.j} finished successfully.")