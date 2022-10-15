from pathlib import Path

# rng entropies; these integers must be different to guarantee different randomness in data during train, val, test
RNG_ENTROPY_TRAIN = 0
RNG_ENTROPY_TEST = 1
RNG_ENTROPY_HPARAMS = 2

# directories
ROOT_DIR = Path(__file__).parents[0]
PROJECT_DIR = Path(__file__).parents[1]

CLUSTER_GROUP_DIR = PROJECT_DIR # replace this with cluster directories
CLUSTER_SCRATCH_DIR = PROJECT_DIR # replace this with cluster scratch directories

# subdirectory names
CHECKPOINT_SUBDIR = "checkpoints"
DATA_SUBDIR = "data"
PROFILE_SUBDIR = "profile"
CONFIG_SUBDIR = "config"
EXPERIMENTS_SUBDIR = "experiments"
EXPERIMENT_CHECKPOINT_SUBDIR = "checkpoints/experiments"
RESULTS_SUBDIR = "results"
PLOTS_SUBDIR = "plots"
REAL_DATA_SUBDIR = "real_data"

# experiments
EXPERIMENT_DATA = "data"
EXPERIMENT_PREDS = "predictions"
EXPERIMENT_SUMMARY = "summary"

EXPERIMENT_CONFIG_TRAIN = "train.yaml"
EXPERIMENT_CONFIG_DATA = "data.yaml"
EXPERIMENT_CONFIG_METHODS = "methods.yaml"

CHECKPOINT_KWARGS = "kwargs.json"

FILE_DATA_G = "g.csv"
FILE_DATA_X_OBS = "x_observational.csv"
FILE_DATA_X_INT = "x_interventional.csv"
FILE_DATA_X_INT_INFO = "x_interventional_info.csv"
FILE_DATA_X_OBS_HELDOUT = "x_heldout_observational.csv"
FILE_DATA_X_INT_HELDOUT =  "x_heldout_interventional.csv"
FILE_DATA_X_INT_INFO_HELDOUT = "x_heldout_interventional_info.csv"

FILE_DATA_META = f"info.json"

BASELINE_ALL_TRIVIAL = "trivial"
BASELINE_ZERO = "zero"
BASELINE_RAND = "rand"
BASELINE_RAND_EDGES = "rand-edges"
BASELINE_PC = "pc"
BASELINE_GES = "ges"
BASELINE_GIES = "gies"
BASELINE_LINGAM = "lingam"
BASELINE_DAGGNN = "daggnn"
BASELINE_GRANDAG = "grandag"
BASELINE_DCDI = "dcdi"
BASELINE_IGSP = "igsp"
BASELINE_CAM = "cam"
BASELINE_DIBS = "dibs"

BASELINE_BOOTSTRAP = "bootstrap-"

BASELINE_ALL_TRIVIAL_ARR = [
    BASELINE_ZERO,
    BASELINE_RAND,
    BASELINE_RAND_EDGES,
]

BASELINES_OBSERV = [
    BASELINE_PC,
    BASELINE_GES,
    BASELINE_LINGAM,
    BASELINE_DAGGNN,
    BASELINE_GRANDAG,
    "ours-observ",
]

BASELINES_INTERV = [
    BASELINE_GIES,
    BASELINE_IGSP,
    BASELINE_CAM,
    BASELINE_DCDI,
    BASELINE_DIBS,
    "ours-interv",
]


SERGIO_NOISE_CONFIG = "avici/sergio/noise_config.yaml"

# data
# from https://github.com/tschaffter/genenetweaver/tree/master/src/ch/epfl/lis/networks
GRAPH_YEAST = "real_data/yeast_transcriptional_network_Balaji2006.tsv"
GRAPH_ECOLI = "real_data/ecoli_transcriptional_network_regulonDB_6_7.tsv"

REAL_DATASET_KEY = "real_dataset"
REAL_SACHS_RID = "sachs"
REAL_SACHS_SUBDIR = REAL_DATA_SUBDIR + "/" + "sachs"

# wandb
WANDB_ENTITY = "larslorch"

# yaml
YAML_FUNC = "__func__"
YAML_RUN = "__run__"
YAML_TRAIN = "__train__"
YAML_CHECKPOINT = "__checkpoint__"

DEFAULT_RUN_KWARGS = {"n_cpus": 1, "n_gpus": 0, "length": "short"}
