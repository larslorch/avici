import sys
import avici
sys.modules['amortibs'] = avici # legacy compatibility with old package name with pickle

import inspect
import subprocess
import json
import pickle
import time
from contextlib import contextmanager

import numpy as onp
import pandas as pd

from avici.definitions import FILE_DATA_META, FILE_DATA_G, \
    FILE_DATA_X_OBS, FILE_DATA_X_INT, FILE_DATA_X_INT_INFO, \
    FILE_DATA_X_OBS_HELDOUT, FILE_DATA_X_INT_HELDOUT, FILE_DATA_X_INT_INFO_HELDOUT, \
    CHECKPOINT_KWARGS

from avici.utils.parse import Synthetic

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, onp.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class NumpyJSONDecoder(json.JSONDecoder):
    def _postprocess(self, obj):
        if isinstance(obj, dict):
            return {k: self._postprocess(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return onp.array([self._postprocess(v) for v in obj])
        else:
            return obj

    def decode(self, obj, recurse=False):
        decoded = json.JSONDecoder.decode(self, obj)
        return self._postprocess(decoded)

@contextmanager
def timer() -> float:
    start = time.time()
    yield lambda: time.time() - start

def get_git_hash_long():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode(sys.stdout.encoding)

def get_id(path):
    return int(path.name.split("_")[-1].split('.')[0])

def get_bootstrap_id(path):
    return int(path.name.split("_")[-2]), int(path.name.split("_")[-1].split("=")[1].split('.')[0])

def to_dict(d):
    if hasattr(d, "func"):
        # partial func
        args = {
            k: v.default
            for k, v in inspect.signature(d).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        # get defaults if not specified
        args.update({
            k: v.default
            for k, v in inspect.signature(d.func).parameters.items()
            if (v.default is not inspect.Parameter.empty
                and k not in args)
        })
        for k in ["n_interv_vars"]:
            if k in args:
                del args[k]
        return {
            "__type__": d.func.__name__,
            **{k: to_dict(v) for k, v in args.items() if v is not None},
        }
    elif isinstance(d, Synthetic):
        # namedtuple
        return {k: to_dict(v) for k, v in d._asdict().items() if v is not None}
    else:
        return d


def save_csv(arr, path):
    if arr.size > 0:
        pd.DataFrame(arr).to_csv(
            path,
            index=False,
            header=False,
            float_format="%.8f",
        )


def load_data(path, *, observ_only, load_heldout=False):
    loaded_data = {}

    # load meta data
    with open(path / FILE_DATA_META, "r") as file:
        meta_info = json.load(file)
        loaded_data["is_count_data"] = onp.array(meta_info["is_count_data"], dtype=bool)

    # load g
    loaded_data["g"] = onp.array(pd.read_csv(path / FILE_DATA_G, index_col=False, header=None), dtype=onp.int32)

    # try to load x_obs
    if (p := path / FILE_DATA_X_OBS).is_file():
        x_obs_vals = onp.array(pd.read_csv(p, index_col=False, header=None), dtype=onp.float32)
        loaded_data["x_obs"] = onp.stack([x_obs_vals, onp.zeros_like(x_obs_vals)], axis=-1)
    else:
        loaded_data["x_obs"] = onp.zeros((0, meta_info["n_vars"], 2))  # dummy

    # try to load x_int
    if (p_vals := path / FILE_DATA_X_INT).is_file() and (p_info := path / FILE_DATA_X_INT_INFO).is_file():
        x_int_vals = onp.array(pd.read_csv(p_vals, index_col=False, header=None), dtype=onp.float32)
        x_int_info = onp.array(pd.read_csv(p_info, index_col=False, header=None), dtype=onp.float32)
        loaded_data["x_int"] = onp.stack([x_int_vals, x_int_info], axis=-1)
    else:
        loaded_data["x_int"] = onp.zeros((0, meta_info["n_vars"], 2))  # dummy

    # try to load heldout observations
    if load_heldout:
        if (p_ho := path / FILE_DATA_X_OBS_HELDOUT).is_file():
            x_obs_ho_vals = onp.array(pd.read_csv(p_ho, index_col=False, header=None), dtype=onp.float32)
            loaded_data["x_heldout_obs"] = onp.stack([x_obs_ho_vals, onp.zeros_like(x_obs_ho_vals)], axis=-1)
        if (p_ho_vals := path / FILE_DATA_X_INT_HELDOUT).is_file() and (p_ho_info := path / FILE_DATA_X_INT_INFO_HELDOUT).is_file():
            x_int_ho_vals = onp.array(pd.read_csv(p_ho_vals, index_col=False, header=None), dtype=onp.float32)
            x_int_ho_info = onp.array(pd.read_csv(p_ho_info, index_col=False, header=None), dtype=onp.float32)
            loaded_data["x_heldout_int"] = onp.stack([x_int_ho_vals, x_int_ho_info], axis=-1)

    # for real data: simply use all of the data
    if meta_info["model"] == "real":
        if observ_only:
            loaded_data["x_obs"] = onp.concatenate([loaded_data["x_obs"], loaded_data["x_int"]], axis=0)
            loaded_data["x_obs"][..., 1] = 0 # ignore intervention indicators for obs only
            loaded_data["x_int"] = onp.zeros((0, meta_info["n_vars"], 2))  # dummy
        else:
            pass

    # for synthetic data: equal total number of observations
    # i.e. if we use interventional data, use less observational data instead
    else:
        if observ_only:
            loaded_data["x_int"] = onp.zeros((0, meta_info["n_vars"], 2))  # dummy
            if load_heldout and "x_heldout_int" in loaded_data:
                loaded_data["x_heldout_int"] = onp.zeros((0, meta_info["n_vars"], 2))  # dummy
        else:
            n_observ_remaining = max(0, loaded_data["x_obs"].shape[0] - loaded_data["x_int"].shape[0])
            loaded_data["x_obs"] = loaded_data["x_obs"][:n_observ_remaining]
            if load_heldout and "x_heldout_obs" in loaded_data and "x_heldout_int" in loaded_data:
                n_observ_ho_remaining = max(0, loaded_data["x_heldout_obs"].shape[0] - loaded_data["x_heldout_int"].shape[0])
                loaded_data["x_heldout_obs"] = loaded_data["x_heldout_obs"][:n_observ_ho_remaining]

    return loaded_data


def load_pred(path):
    with open(path, "r") as file:
        pred = json.load(file, cls=NumpyJSONDecoder)
        return pred


def load_checkpoint(folder):
    # load state
    last_checkpoint = sorted([p for p in folder.iterdir() if ".pkl" in p.name])[-1]
    with open(last_checkpoint, 'rb') as file:
        state = pickle.load(file)

    # load kwargs
    kwargs_checkpoint = folder / CHECKPOINT_KWARGS
    with open(kwargs_checkpoint, "r") as file:
        kwargs = json.load(file)
        kwargs["inference_model_kwargs"]["acyclicity"] = None

    return state, kwargs

