import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")

import argparse
from avici.utils.version_control import str2bool
from pathlib import Path
from avici.experiment.utils import load_data, get_id, get_bootstrap_id, load_pred
import numpy as onp
import copy
from scipy.special import logsumexp

from pprint import pprint

import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

from avici.eval.metrics import shd, threshold_metrics, is_cyclic, classification_metrics, n_edges, orient_cpdag_oracle
from collections import defaultdict
from avici.eval.plot import benchmark_summary, visualize_preds, visualize_calibration
from avici.utils.parse import load_methods_config
from avici.utils.sugar import to_unique

from cdt.metrics import SID
from avici.definitions import BASELINE_ZERO, BASELINE_RAND, BASELINE_RAND_EDGES, BASELINES_OBSERV, BASELINES_INTERV, \
    BASELINE_ALL_TRIVIAL_ARR, BASELINE_BOOTSTRAP


def make_summary(summary_path, result_paths, data_paths, seed, decision_threshold=0.5,
                 mask_definite_loops=True, show_trivial=False, compute_sid=False):
    """
    Args:
        summary_path
        result_paths (dict): {method: [paths to result csv's]}
        data_paths (list): [paths to data csv's]
        seed
        decision_threshold
        mask_definite_loops
        show_trivial

    """
    # id: data dict
    data_observ = {get_id(p): load_data(p, observ_only=True, load_heldout=True) for p in data_paths}
    data_interv = {get_id(p): load_data(p, observ_only=False, load_heldout=True) for p in data_paths}

    # method: dict of metrics
    results = defaultdict(lambda: defaultdict(list))
    g_preds = defaultdict(dict)
    g_pred_probs = defaultdict(dict)
    checkpoint_check = {}

    for method, mpaths in result_paths.items():

        print(f"summarizing: {method}")
        rng = onp.random.default_rng(seed) # for random oracle cpdag extensions

        # determine heldout data
        base_method_full = method.split("__")[0] # catch hyperparameter calibration case where name differs
        base_method = base_method_full.split(BASELINE_BOOTSTRAP)[-1] # catch bootstrap case
        is_bootstrap = len(base_method_full.split(BASELINE_BOOTSTRAP)) != 1

        if base_method in BASELINES_OBSERV or base_method in BASELINE_ALL_TRIVIAL_ARR:
            data = data_observ
        elif base_method in BASELINES_INTERV:
            data = data_interv
        else:
            raise KeyError(f"Unknown class of base method `{base_method}`")

        # load predictions as id: prediction dict
        if is_bootstrap:
            all_predictions = defaultdict(list)
            for p in mpaths:
                idx, _ = get_bootstrap_id(p)
                all_predictions[idx].append(load_pred(p))
        else:
            all_predictions = {get_id(p): load_pred(p) for p in mpaths}

        # compute metrics individually for every test case
        for pred_id, pred in all_predictions.items():
            assert pred_id in data.keys(), \
                f"pred `{pred_id}` doesn't have matching data instance.\ndata_paths: {data_paths}\nmpaths: {mpaths}"

            # load ground truth graph
            g_true = data[pred_id]["g"]
            if pred_id not in g_preds["__true__"]:
                g_preds["__true__"][pred_id] = g_true

            # bootstrap methods: convert set of estimates to probabilistic estimate
            if type(pred) == list:
                pred_collection = defaultdict(list)
                for pr in pred:
                    # cpdag methods: create an (semi-oracle) prediction where undirected edges are oriented correctly
                    if "cpdag_edges" in pr.keys():
                        assert "cpdag_oracle_expansion" not in pr.keys()
                        pr["cpdag_oracle_expansion"] = orient_cpdag_oracle(rng, g_true, pr["cpdag_edges"])

                    # append all predictions for which we want to compute the bootstrap estimate
                    if "g_edges" in pr:
                        pred_collection["g_edges"].append(pr["g_edges"])
                    if "cpdag_oracle_expansion" in pr:
                        pred_collection["cpdag_oracle_expansion"].append(pr["cpdag_oracle_expansion"])

                # convert to nonparametric bootstrap probability estimate and also convert to hard pred
                pred = {f"{k}_prob": onp.mean(onp.stack(vs), axis=0) for k, vs in pred_collection.items()}
                for k, v in copy.deepcopy(pred).items():
                    pred[k.rsplit("_", 1)[0]] = (v > 0.5).astype(onp.int32)

            # generate predictions for our method given the predicted variational parameters
            if "logp_ij_mixt" in pred and "logp_mixt" in pred:
                # [..., mixture_k, d, d]
                logp_ij_mixt = pred["logp_ij_mixt"]
                d = logp_ij_mixt.shape[-1]
                assert logp_ij_mixt.shape[-2] == logp_ij_mixt.shape[-1]

                # [..., mixture_k]
                logp_mixt = pred["logp_mixt"]
                assert logp_mixt.shape[0] == logp_ij_mixt.shape[-3]

                # edges
                # [..., d, d]
                g_edges_logprob = logsumexp(onp.expand_dims(logp_mixt, axis=(-2, -1)) + logp_ij_mixt, axis=-3)
                if mask_definite_loops:
                    g_edges_logprob[..., onp.arange(d), onp.arange(d)] = -onp.inf

                pred["g_edges_prob"] = onp.exp(g_edges_logprob)
                pred["g_edges"] = (onp.exp(g_edges_logprob) > decision_threshold).astype(onp.int32)


            # cpdag methods: create a semi-oracle prediction where undirected edges are oriented correctly
            if "cpdag_edges" in pred:
                pred["cpdag_oracle_expansion"] = orient_cpdag_oracle(rng, g_true, pred["cpdag_edges"])

            """Save graphs for visualization"""
            if "cpdag_oracle_expansion_prob" in pred:
                g_preds[method][pred_id] = pred["cpdag_oracle_expansion_prob"]
                g_pred_probs[method][pred_id] = pred["cpdag_oracle_expansion_prob"]

            elif "g_edges_prob" in pred:
                g_preds[method][pred_id] = pred["g_edges_prob"]
                g_pred_probs[method][pred_id] = pred["g_edges_prob"]

            else:
                g_preds[method][pred_id] = pred["g_edges"]

            """Compute metrics"""
            # shd
            if compute_sid:
                if "cpdag_edges" in pred:
                    results[method]["sid"].append(float(SID(g_true, pred["cpdag_oracle_expansion"])))
                elif "g_edges" in pred:
                    results[method]["sid"].append(float(SID(g_true, pred["g_edges"])))

            if "cpdag_edges" in pred:
                results[method]["shd"].append(shd(g_true, pred["cpdag_oracle_expansion"]))
            elif "g_edges" in pred:
                results[method]["shd"].append(shd(g_true, pred["g_edges"]))

            # cyclic (use consistent extension over oracle here for cpdag methods)
            if "g_edges" in pred:
                results[method]["cyclic"].append(is_cyclic(pred["g_edges"]))
            elif "cpdag_oracle_expansion" in pred:
                results[method]["cyclic"].append(is_cyclic(pred["cpdag_oracle_expansion"]))

            # edges
            if "cpdag_edges" in pred:
                results[method]["edges"].append(n_edges(pred["cpdag_oracle_expansion"]))
            elif "g_edges" in pred:
                results[method]["edges"].append(n_edges(pred["g_edges"]))


            # classification_metrics
            if "cpdag_edges" in pred:
                cl = classification_metrics(g_true, pred["cpdag_oracle_expansion"])
                for k, v in cl.items():
                    results[method][k].append(v)
            elif "g_edges" in pred:
                cl = classification_metrics(g_true, pred["g_edges"])
                for k, v in cl.items():
                    results[method][k].append(v)

            # probabilistic metrics
            # AUROC, AUPRC, AP
            if "cpdag_oracle_expansion_prob" in pred:
                thres = threshold_metrics(g_true, pred["cpdag_oracle_expansion_prob"])
                for k, v in thres.items():
                    results[method][k].append(v)

            elif "g_edges_prob" in pred:
                thres = threshold_metrics(g_true, pred["g_edges_prob"])
                for k, v in thres.items():
                    results[method][k].append(v)

            # record heldout score if available
            if "heldout_score" in pred:
                results[method]["heldout_score"].append(pred["heldout_score"])

            # walltime
            if "walltime" in pred:
                results[method]["walltime"].append(pred["walltime"])

            # num_params
            if "num_params" in pred:
                results[method]["num_params"].append(pred["num_params"])

            # byte_size_f32
            if "byte_size_f32" in pred:
                results[method]["byte_size_f32"].append(pred["byte_size_f32"])

            # mbyte_size_f32
            if "mbyte_size_f32" in pred:
                results[method]["mbyte_size_f32"].append(pred["mbyte_size_f32"])

            # check checkpoint the same across results if there is one
            if "checkpoint_dir" in pred:
                if method not in checkpoint_check:
                    checkpoint_check[method] = pred["checkpoint_dir"]
                elif pred["checkpoint_dir"] != checkpoint_check[method]:
                    warnings.warn(f"Checkpoints of method `{method}` do not match. "
                                  f"`{pred['checkpoint_dir']}` vs {checkpoint_check[method]}")

    print()

    # dump all metrics for later plotting
    benchmark_summary(summary_path / "dump", copy.deepcopy(results), only_metrics=None,
                      show_trivial=show_trivial, dump_main=True)

    # dump method-wise summary for hparam tuning
    if compute_sid:
        benchmark_summary(summary_path / "hparam-tuning", copy.deepcopy(results),
                          only_metrics=["sid", "f1", "heldout_score"], show_trivial=show_trivial, method_summaries=True)
    else:
        benchmark_summary(summary_path / "hparam-tuning", copy.deepcopy(results),
                          only_metrics=["shd", "f1", "heldout_score", ], show_trivial=show_trivial, method_summaries=True)

    # benchmark results
    benchmark_summary(summary_path / "benchmark", copy.deepcopy(results),
                      only_metrics=["shd", "f1"], show_trivial=show_trivial)

    if compute_sid:
        benchmark_summary(summary_path / "benchmark-sid", copy.deepcopy(results),
                          only_metrics=["sid", "f1"], show_trivial=show_trivial)

    benchmark_summary(summary_path / "benchmark-prob", copy.deepcopy(results),
                      only_metrics=["auroc", "auprc", "ap"], show_trivial=show_trivial)

    # calibration metrics
    experiment_name = summary_path.parent.name
    visualize_calibration(summary_path, experiment_name, copy.deepcopy(g_preds["__true__"]), copy.deepcopy(g_pred_probs))

    # visualize preds
    visualize_preds(summary_path / "predictions", copy.deepcopy(g_preds), only=10)

    print("Finished successfully.")




if __name__ == "__main__":
    """
    Runs plot call
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--path_plots", type=Path, required=True)
    parser.add_argument("--path_results", type=Path, required=True)
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--mask_definite_loops", default=True, type=str2bool)
    parser.add_argument("--show_trivial", action="store_true")
    parser.add_argument("--compute_sid", action="store_true")
    parser.add_argument("--descr")
    kwargs = parser.parse_args()

    methods_config_raw = load_methods_config(kwargs.methods_config_path, abspath=True)

    # account for "ours" being run on both observational and interventional data
    methods_config = {}
    for method in methods_config_raw.keys():
        if "ours" in method:
            methods_config["ours-observ" + method.split("ours")[1]] = methods_config_raw[method]
            methods_config["ours-interv" + method.split("ours")[1]] = methods_config_raw[method]
        else:
            methods_config[method] = methods_config_raw[method]


    data_found = sorted([p for p in kwargs.path_data.iterdir() if p.is_dir()])
    results_p = sorted([p for p in kwargs.path_results.iterdir()])
    results_found = {}
    methods_config.update({BASELINE_ZERO: None, BASELINE_RAND: None, BASELINE_RAND_EDGES: None})
    for meth, _ in methods_config.items():
        method_results = list(filter(lambda p: p.name.rsplit("_", 1)[0] == meth, results_p))
        results_found[meth] = method_results

    make_summary(kwargs.path_plots, results_found, data_found,
                 decision_threshold=kwargs.decision_threshold,
                 mask_definite_loops=kwargs.mask_definite_loops,
                 show_trivial=kwargs.show_trivial,
                 compute_sid=kwargs.compute_sid,
                 seed=kwargs.seed)
    print("Done.")
