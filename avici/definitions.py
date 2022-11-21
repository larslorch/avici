from pathlib import Path

# rng entropies; these integers must be different to guarantee different randomness in data during train, val, test
RNG_ENTROPY_TRAIN = 0
RNG_ENTROPY_TEST = 1
RNG_ENTROPY_HPARAMS = 2

# directories
ROOT_DIR = Path(__file__).parents[0]
PROJECT_DIR = Path(__file__).parents[1]

CHECKPOINT_SUBDIR = "checkpoints"
ASSETS_SUBDIR = "assets"
CACHE_SUBDIR = "cache"
SERGIO_NOISE_CONFIG = ROOT_DIR / "synthetic/sergio/noise_config.yaml"

# real data
REAL_DATASET_KEY = "real_dataset"
GRAPH_YEAST = "assets/yeast_transcriptional_network_Balaji2006.tsv"
GRAPH_ECOLI = "assets/ecoli_transcriptional_network_regulonDB_6_7.tsv"

# yaml
CHECKPOINT_KWARGS = "kwargs.json"
YAML_CLASS = "__class__"
YAML_MODULES = "additional_modules"

# model figshare ids
MODEL_LINEAR_FIGSHARE_ID = 21341034
MODEL_RFF_FIGSHARE_ID = 21341043
MODEL_GENE_FIGSHARE_ID = 21341049
MODEL_RFF_ABLATIONS_FIGSHARE_ID = 21341316
