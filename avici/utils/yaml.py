import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import yaml
from pathlib import Path
from avici.definitions import PROJECT_DIR


def load_config(path, abspath=False):
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