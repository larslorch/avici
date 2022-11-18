import sys
import avici
sys.modules['amortibs'] = avici # legacy compatibility with old package name with pickled checkpoints
sys.modules['amortibs.modules'] = avici # legacy compatibility with old package name with pickled checkpoints

import warnings
from pathlib import Path
import yaml
import json
import pickle

from avici.definitions import CHECKPOINT_KWARGS, PROJECT_DIR


def load_yaml(path, abspath=False):
    """Load plain yaml config"""

    load_path = path if abspath else (PROJECT_DIR / path)

    try:
        with open(load_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                return config

            except yaml.YAMLError as exc:
                warnings.warn(f"YAML parsing error. Returning `None` for config.\n")
                print(exc, flush=True)
                return None
    except FileNotFoundError:
        warnings.warn(f"{Path(path).name} doesn't exist. Returning `None` for config. "
                      f"This is fine if it is the data.yaml in a train experiment folder.\n")
        return None


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
