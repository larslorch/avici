import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import shutil
import re
from pathlib import Path
from pprint import pprint
import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

from launch import generate_run_commands

from avici.utils.parse import load_data_config, load_methods_config

from avici.definitions import PROJECT_DIR, CLUSTER_GROUP_DIR, RESULTS_SUBDIR, EXPERIMENTS_SUBDIR, \
    EXPERIMENT_DATA, EXPERIMENT_PREDS, EXPERIMENT_SUMMARY, YAML_RUN, YAML_TRAIN, YAML_CHECKPOINT, \
    EXPERIMENT_CONFIG_TRAIN, EXPERIMENT_CONFIG_DATA, EXPERIMENT_CONFIG_METHODS, EXPERIMENT_CHECKPOINT_SUBDIR, \
    DEFAULT_RUN_KWARGS, BASELINE_ALL_TRIVIAL, BASELINE_ZERO, BASELINE_RAND, BASELINE_RAND_EDGES, \
    BASELINES_OBSERV, BASELINES_INTERV, BASELINE_BOOTSTRAP,\
    REAL_DATASET_KEY, REAL_SACHS_RID, REAL_SACHS_SUBDIR


class ExperimentManager:
    """Tool for clean and reproducible experiment handling via folders"""

    def __init__(self, experiment, seed=0, verbose=True, compute="local", n_datasets=None, dry=True,
                 only_methods=None, train_comment=None, heldout_split=0.2, n_bootstraps=10, train_addon=None):
        self.experiment = experiment
        self.config_path = PROJECT_DIR / EXPERIMENTS_SUBDIR / self.experiment
        self.store_path_root = PROJECT_DIR
        self.store_path = self.store_path_root / RESULTS_SUBDIR / self.experiment
        self.seed = seed
        self.compute = compute
        self.verbose = verbose
        self.dry = dry
        self.train_comment = ("-" + train_comment) if train_comment is not None else ""
        self.train_addon = train_addon or ""
        self.heldout_split = heldout_split
        self.n_bootstraps = n_bootstraps if compute != "local" else 2

        self.lsf_logs_dir = f"{PROJECT_DIR}/logs_lsf/"
        Path(self.lsf_logs_dir).mkdir(exist_ok=True)

        self.train_config_path = self.config_path / EXPERIMENT_CONFIG_TRAIN
        self.data_config_path = self.config_path / EXPERIMENT_CONFIG_DATA
        self.methods_config_path = self.config_path / EXPERIMENT_CONFIG_METHODS

        if self.verbose:
            if self.config_path.exists() \
                and self.config_path.is_dir() \
                and ((self.train_config_path.is_file() and self.methods_config_path.is_file())
                 or (self.methods_config_path.is_file() and self.data_config_path.is_file())):
                print("experiment:       ", self.experiment, flush=True)
                print("results directory:", self.store_path, flush=True, end="\n\n")
            else:
                print(f"experiment `{self.experiment}` not specified in `{self.config_path}`."
                      f"check spelling and files")
                exit(1)

        # parse configs
        self.data_config = load_data_config(self.data_config_path, verbose=False, abspath=True)
        self.methods_config = load_methods_config(self.methods_config_path, abspath=True)

        # adjust configs
        if only_methods is not None:
            all_methods = list(self.methods_config.keys())
            for k in all_methods:
                if k not in only_methods:
                    del self.methods_config[k]

        self.n_datasets = n_datasets
        if self.n_datasets is None and self.data_config is not None:
            self.n_datasets = self.data_config["n_datasets"]

    def _inherit_specification(self, subdir, inherit_from):
        if inherit_from is not None:
            v = str(inherit_from.name).split("_")[1:]
            return subdir + "_" + "_".join(v)
        else:
            return subdir

    def _get_name_without_version(self, p):
        return "_".join(p.name.split("_")[:-1])

    def _list_main_folders(self, subdir, root_path=None, inherit_from=None):
        if root_path is None:
            root_path = self.store_path
        subdir = self._inherit_specification(subdir, inherit_from)
        if root_path.is_dir():
            return sorted([
                p for p in root_path.iterdir()
                if (p.is_dir() and subdir == self._get_name_without_version(p))
            ])
        else:
            return []

    def _init_folder(self, subdir, root_path=None, inherit_from=None, dry=False, add_logs_folder=False):
        if root_path is None:
            root_path = self.store_path
        subdir = self._inherit_specification(subdir, inherit_from)
        existing = self._list_main_folders(subdir, root_path=root_path)
        folder = root_path / (subdir + f"_{len(existing):02d}")
        if not dry:
            folder.mkdir(exist_ok=False, parents=True)
            if add_logs_folder:
                (folder / "logs").mkdir(exist_ok=False, parents=True)
        return folder

    def _copy_file(self, from_path, to_path):
        shutil.copy(from_path, to_path)

    def launch_train(self, check=False):
        # launch train script for our model
        checkpoint_folders = {}
        for k, (method, hparams) in enumerate({k: v for k, v in self.methods_config.items() if "ours" in k}.items()):
            checkpoint_dir = self.store_path_root / EXPERIMENT_CHECKPOINT_SUBDIR / self.experiment
            if check:
                # return the existing checkpoint folder
                # requesting custom checkpoint
                if YAML_CHECKPOINT in self.methods_config[method]:
                    requested_dir = Path(self.methods_config[method][YAML_CHECKPOINT])
                    if not requested_dir.exists():
                        warnings.warn(f"requested checkpoint folder `{requested_dir}` doesn't exist ")
                    checkpoint_folders[method] = requested_dir

                # get last checkpoint in the default checkpoint directory for the experiment
                elif not checkpoint_dir.exists():
                    warnings.warn(f"checkpoint folder for experiment `{self.experiment}` "
                                  f"doesn't exist at {checkpoint_dir}; run `--train` first")
                else:
                    paths_checkpoint = self._list_main_folders(method, root_path=checkpoint_dir)
                    if len(paths_checkpoint) == 0:
                        warnings.warn(f"checkpoints for `{method}` not created yet; run `--train` first")
                    else:
                        checkpoint_folders[method] = paths_checkpoint[-1]
            else:
                # create a checkpoint folder and launch train script
                checkpoint_folders[method] = self._init_folder(method, root_path=checkpoint_dir, dry=self.dry)
                args = {k: v for k, v in hparams.items() if k not in [YAML_TRAIN, YAML_RUN]}
                cmd = f"python scripts/train.py --config '{self.train_config_path}' " \
                      f"--project 'experiment-{self.experiment}' " \
                      f"--descr '{checkpoint_folders[method].name}{self.train_comment}' " \
                      f"--online {'False' if self.compute == 'local' else 'True'}  " \
                      f"--checkpoint True --checkpoint_dir {checkpoint_folders[method]} " \
                      f"--log_every 1000 --eval_every 15000 " \
                      f"--group_scratch True " \
                      f"{self.train_addon} "
                cmd += " " + " ".join([(f"--{k} '{v}'" if type(v) == str else f"--{k} {v}") for k, v in args.items()])

                all_kwargs = list(re.findall(r'(?<=--)\w+', cmd))  # get all args from kwargs
                for k in all_kwargs:
                    if all_kwargs.count(k) > 1:
                        print(f"error: kwarg `{k}` double in command.")
                        exit()

                run_kwargs = hparams[YAML_TRAIN] if hparams is not None else {}
                if "gpu_model" in run_kwargs:
                    if set(run_kwargs["gpu_model"]) == {'NVIDIATITANRTX', 'QuadroRTX6000'} and self.compute != "euler_lsf":
                        warnings.warn("Requested LSF-type GPUs with 24GB and 128 cores even though using slurm. \n"
                                      "Using NVIDIAGeForceRTX3090 instead on slurm.\n\ngpu_model=rtx_3090\n")
                        run_kwargs["gpu_model"] = "rtx_3090"

                generate_run_commands(
                    command_list=[cmd],
                    mode=self.compute,
                    dry=self.dry,
                    prompt=False,
                    relaunch=True,
                    relaunch_after=args.get("relaunch_after", None),
                    output_filename=f"{self.experiment}-",
                    output_path_prefix=self.lsf_logs_dir,
                    **run_kwargs,
                )

        return checkpoint_folders

    def make_data(self, check=False):
        if check:
            assert self.store_path.exists(), "folder doesn't exist; run `--data` first"
            paths_data = self._list_main_folders(EXPERIMENT_DATA)
            assert len(paths_data) > 0, "data not created yet; run `--data` first"
            final_data = list(filter(lambda p: p.name.rsplit("_", 1)[-1] == "final", paths_data))
            if final_data:
                assert len(final_data) == 1
                return final_data[0]
            else:
                return paths_data[-1]

        # init results folder
        if not self.store_path.exists():
            self.store_path.mkdir(exist_ok=False, parents=True)

        # init data folder
        path_data = self._init_folder(EXPERIMENT_DATA)
        self._copy_file(self.data_config_path, path_data / EXPERIMENT_CONFIG_DATA)
        if self.dry:
            shutil.rmtree(path_data)

        # handle real data case
        if REAL_DATASET_KEY in self.data_config:
            rid = self.data_config[REAL_DATASET_KEY]
            if rid == REAL_SACHS_RID:
                from_path = PROJECT_DIR / REAL_SACHS_SUBDIR
            else:
                raise KeyError(f"Unknown real dataset `{rid}`")

            try:
                path_data.mkdir(exist_ok=True)
                shutil.copytree(from_path, path_data / "1")
            except FileNotFoundError:
                print(f"Data of `{rid}` not available at `{from_path}.\nMaybe you need to unzip the folder?")

            if self.dry:
                shutil.rmtree(path_data)
            else:
                print(f"Copied `{REAL_SACHS_RID}` data to experiment folder.")
            return

        # launch runs that generate data
        experiment_name = kwargs.experiment.replace("/", "--")
        cmd = f"python '{PROJECT_DIR}/avici/experiment/data.py' " \
              f"--j \$LSB_JOBINDEX  " \
              f"--data_config_path '{self.data_config_path}' " \
              f"--path_data '{path_data}' " \
              f"--descr '{experiment_name}-data' "

        generate_run_commands(
            array_command=cmd,
            array_indices=range(1, self.n_datasets + 1),
            mode="euler_lsf" if "euler" in self.compute else self.compute,
            length="short",
            n_cpus=1,
            n_gpus=0,
            prompt=False,
            dry=self.dry,
            output_path_prefix=self.lsf_logs_dir,
        )
        print(f"\nLaunched {self.n_datasets} runs total.")
        return path_data

    def launch_methods(self, heldout=False, check=False):
        # check data has been generated
        path_data = self.make_data(check=True)

        if check:
            paths_results = self._list_main_folders(EXPERIMENT_PREDS, inherit_from=path_data)
            assert len(paths_results) > 0, "results not created yet; run `--methods` first"
            final_results = list(filter(lambda p: p.name.rsplit("_", 1)[-1] == "final", paths_results))
            if final_results:
                assert len(final_results) == 1
                return final_results[0]
            else:
                return paths_results[-1]

        # get checkpoints for our methods
        checkpoint_folders = self.launch_train(check=True)
        checkpoints = {}
        skipped = {}
        for our_method in [k for k in self.methods_config.keys() if "ours" in k]:
            try:
                folder = checkpoint_folders[our_method]
                if folder.is_dir() and [p for p in folder.iterdir() if ".pkl" in p.name]:
                    checkpoints[our_method] = folder
                else:
                    skipped[our_method] = folder
            except KeyError:
                skipped[our_method] = None

        # init results folder
        path_results = self._init_folder(EXPERIMENT_PREDS, inherit_from=path_data)
        self._copy_file(self.methods_config_path, path_results / EXPERIMENT_CONFIG_METHODS)
        if self.dry:
            shutil.rmtree(path_results)

        # print data sets expected and found
        data_found = sorted([p for p in path_data.iterdir() if p.is_dir()])
        if len(data_found) != self.n_datasets:
            warnings.warn(f"Number of data sets does not match data config "
                f"(got: `{len(data_found)}`, expected `{self.n_datasets}`).\n"
                f"data path: {path_data}\n")
            if len(data_found) < self.n_datasets:
                print("Aborting.")
                return
            else:
                print(f"Taking first {self.n_datasets} data folders")
                data_found = data_found[:self.n_datasets]

        elif self.verbose:
            print(f"\nLaunching experiments for {len(data_found)} data sets.")

        n_launched, n_methods = 0, 0
        path_data_root = data_found[0].parent

        # launch runs that execute methods
        print("baseline methods:\n")
        experiment_name = kwargs.experiment.replace("/", "--")
        for k, (method, hparams) in enumerate(self.methods_config.items()):
            method_core_full = method.split("__")[0] # catch hparam validation case where names differ

            method_core = method_core_full.split(BASELINE_BOOTSTRAP)[-1] # catch bootstrap case
            is_bootstrap = method_core_full.split(BASELINE_BOOTSTRAP)[0] == ""

            assert method_core in BASELINES_OBSERV or method_core in BASELINES_INTERV or "ours" in method_core, \
                f"Baseline `{method_core}` is neither in BASELINES_OBSERV nor BASELINES_INTERV. " \
                f"Specify s.t. correct data is used for inference in jobs."

            if method in skipped:
                continue
            n_methods += 1
            seed_indices = sorted([int(p.name) for p in data_found])

            # if possible convert to range for shorter bsub command
            if seed_indices == list(range(seed_indices[0], seed_indices[-1] + 1)):
                seed_indices = range(seed_indices[0], seed_indices[-1] + 1)

            cmd = f"python '{PROJECT_DIR}/avici/experiment/run.py' " \
                  f"--method {method} " \
                  f"--seed \$LSB_JOBINDEX " \
                  f"--data_id \$LSB_JOBINDEX " \
                  f"--path_results '{path_results}' " \
                  f"--path_data_root '{path_data_root}' " \
                  f"--path_methods_config '{self.methods_config_path}' " \
                  f"--descr '{experiment_name}-{method}-run' "
            if heldout:
                cmd += f"--heldout_split {self.heldout_split} "
            if method in checkpoints:
                cmd += f"--checkpoint_dir '{checkpoints.get(method)}' "

            print()
            assert YAML_RUN in hparams or  hparams is None, f"Add `__run__` specification of `{method}` method in yaml"
            run_kwargs = hparams[YAML_RUN] if hparams is not None else DEFAULT_RUN_KWARGS
            cmd_args = dict(
                array_indices=seed_indices,
                mode="euler_lsf" if "euler" in self.compute else self.compute,
                dry=self.dry,
                prompt=False,
                output_path_prefix=f"{path_results}/logs/",
                **run_kwargs,
            )
            # create log directory here already in case there is a failure before folder creation in script
            if not self.dry:
                (path_results / "logs").mkdir(exist_ok=True, parents=True)

            # normal method (1 run per dataset)
            if not is_bootstrap:
                n_launched += len(seed_indices)
                generate_run_commands(array_command=cmd, **cmd_args)

            # bootstrap method (`n_bootstraps` runs per dataset)
            else:
                for b in range(self.n_bootstraps):
                    cmd_b = copy.deepcopy(cmd) + f" --bootstrap_id {b} "
                    n_launched += len(seed_indices)
                    generate_run_commands(array_command=cmd_b, **cmd_args)


        # warn if skipped
        for our_method, folder in checkpoints.items():
            print(f"checkpoint: {our_method}:\t{folder}")
        for our_method, folder in skipped.items():
            warnings.warn(f"Skipping `{our_method}`: no checkpoint found at {folder or '<no path created yet>'}")

        print(f"\nLaunched {n_launched} runs total ({n_methods} methods)")
        return path_results

    def make_summary(self, compute_sid=False):
        # check results have been generated
        path_data = self.make_data(check=True)
        path_results = self.launch_methods(check=True)

        # init results folder
        path_plots = self._init_folder(EXPERIMENT_SUMMARY, inherit_from=path_results)
        if self.dry:
            shutil.rmtree(path_plots)

        # print results expected and found
        methods_config_raw = self.methods_config
        methods_config = {}
        for method in methods_config_raw.keys():
            if "ours" in method:
                methods_config["ours-observ" + method.split("ours")[1]] = methods_config_raw[method]
                methods_config["ours-interv" + method.split("ours")[1]] = methods_config_raw[method]
            else:
                methods_config[method] = methods_config_raw[method]

        results = sorted([p for p in path_results.iterdir()])
        results_found = {}
        n_results_found = 0
        for method, _ in methods_config.items():
            is_bootstrap = len(method.split(BASELINE_BOOTSTRAP)) != 1 # catch bootstrap case
            n_expected = (self.n_datasets * self.n_bootstraps) if is_bootstrap else self.n_datasets
            method_results = list(filter(lambda p: p.name.rsplit("_", 1)[0] == method, results))
            results_found[method] = method_results
            n_results_found += len(method_results)
            print(f"{method + ':':30s}{len(method_results):4d}/{n_expected}\t"
                  f"{'(!)' if len(method_results) != n_expected else ''}")

        print()
        if not n_results_found:
            return

        # create summary
        experiment_name = kwargs.experiment.replace("/", "--")
        cmd = f"python '{PROJECT_DIR}/avici/experiment/summary.py' " \
              f"--methods_config_path {self.methods_config_path} " \
              f"--path_data {path_data} " \
              f"--path_plots '{path_plots}' " \
              f"--path_results '{path_results}' " \
              f"--descr '{experiment_name}-{path_plots.parts[-1]}' "
        if compute_sid:
            cmd += f" --compute_sid "

        generate_run_commands(
            command_list=[cmd],
            mode="euler_lsf" if "euler" in self.compute else self.compute,
            length="short",
            n_gpus=0,
            n_cpus=4,
            mem=3000,
            # n_cpus=20,
            # mem=100000, # sometimes SID goes OOM, then can use this
            prompt=False,
            dry=self.dry,
            output_path_prefix=self.lsf_logs_dir,
        )
        return path_plots



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, nargs="?", default="test", help="experiment config folder")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--compute", type=str, default="local")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--methods", action="store_true")
    parser.add_argument("--methods_heldout", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--summary_sid", action="store_true")
    parser.add_argument("--only_methods", nargs="+", type=str)
    parser.add_argument("--n_datasets", type=int, help="overwrites default specified in config")
    parser.add_argument("--train_comment", type=str, help="prefix in wandb description; not used otherwise")
    parser.add_argument("--train_addon", type=str, help="string added to train.py command")
    kwargs = parser.parse_args()

    kwargs_sum = sum([
        kwargs.train,
        kwargs.data,
        kwargs.methods,
        kwargs.methods_heldout,
        kwargs.summary,
        kwargs.summary_sid,
    ])
    assert kwargs_sum == 1, f"pass 1 option, got `{kwargs_sum}`"

    exp = ExperimentManager(experiment=kwargs.experiment, compute=kwargs.compute, n_datasets=kwargs.n_datasets,
                            dry=not kwargs.submit, only_methods=kwargs.only_methods, train_comment=kwargs.train_comment,
                            train_addon=kwargs.train_addon)
    if kwargs.train:
        _ = exp.launch_train()
    elif kwargs.data:
        _ = exp.make_data()
    elif kwargs.methods or kwargs.methods_heldout:
        _ = exp.launch_methods(heldout=kwargs.methods_heldout)
    elif kwargs.summary or kwargs.summary_sid:
        _ = exp.make_summary(compute_sid=kwargs.summary_sid)
    else:
        raise ValueError()

