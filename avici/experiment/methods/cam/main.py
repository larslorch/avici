import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import os
import time
import cdt
import random
import math
import argparse
import pandas as pd
import numpy as np
import networkx as nx

try:
    from .cam import CAM_with_score
except ImportError:
    from cam import CAM_with_score


def run_cam(seed, data, config):

    np.random.seed(seed)
    random.seed(seed)

    # concatenate all observations and discard target mask
    x_full = np.concatenate([data["x_obs"], data["x_int"]], axis=-3)
    x, interv_mask = x_full[..., 0], x_full[..., 1]

    # add noise to avoid inversion problems for sparse data
    if "rank_fail_add" in config and "rank_check_tol" in config:

        # check full rank of covariance matrix
        full_rank = np.linalg.matrix_rank((x - x.mean(-2, keepdims=True)).T @
                                           (x - x.mean(-2, keepdims=True)),
                                          tol=config["rank_check_tol"]) == x.shape[-1]

        if not full_rank:
            zero_cols = np.where((x == 0.0).all(-2))[0]
            zero_rows = np.where((x == 0.0).all(-1))[0]
            warnings.warn(f"covariance matrix not full rank; "
                          f"we have {len(zero_rows)} zero rows and {len(zero_cols)} zero cols "
                          f"(can occur in gene expression data). "
                          f"Adding infinitesimal noise to observations.")

            x += np.random.normal(loc=0, scale=config["rank_fail_add"], size=x.shape)


    # regimes: np.ndarray [n_observations,] indicating which regime/environment row i originated from
    # masks: list of length [n_obs], where masks[i] is list of intervened nodes; empty list for observational
    unique, regimes = np.unique(interv_mask, axis=0, return_inverse=True)
    interv_targets = [(np.where(v)[0].tolist() if v.sum() else []) for v in unique]
    masks = [interv_targets[jj] for jj in regimes]

    # split data into train and test
    perm = np.random.permutation(x.shape[0])
    test_cut = math.ceil(x.shape[0] * config["score_test_data"])
    train_idx = perm[test_cut:]
    test_idx = perm[:test_cut]

    train_data_pd, test_data_pd = pd.DataFrame(x[train_idx]), pd.DataFrame(x[test_idx])
    # regime_train, regime_test = regimes[train_idx], regimes[test_idx]
    mask_train = np.ones((train_idx.shape[0], x.shape[1]))
    for i, m in enumerate([masks[i] for i in train_idx]):
        for j in m:
            mask_train[i, j] = 0
    mask_test = np.ones((test_idx.shape[0], x.shape[1]))
    for i, m in enumerate([masks[i] for i in test_idx]):
        for j in m:
            mask_test[i, j] = 0

    # apply CAM
    obj = CAM_with_score(config["score"], config["cutoff"], config["variable_sel"], config["sel_method"],
                         config["pruning"], config["prune_method"])

    mask_train_pd = pd.DataFrame(mask_train)
    mask_test_pd = pd.DataFrame(mask_test)
    dag, train_score, val_score = obj.get_score(train_data_pd, test_data_pd, mask_train_pd, mask_test_pd)

    dag = nx.to_numpy_matrix(dag)
    return dict(g_edges=dag)


if __name__ == "__main__":


    from avici.utils.parse import load_data_config
    from avici.synthetic.utils import onp_standardize_data

    test_spec = load_data_config("config/linear_additive-0.yaml")["data"]["train"][0]
    # test_spec = load_data_config("config/rff_additive-0.yaml")["data"]["train"][0]
    # test_spec = load_data_config("config/sergio-0.yaml")["data"]["train"][0]
    # test_spec = load_data_config("experiments/test/data.yaml")["data"]["evaluation"][0]
    testnvars = 10

    testrng = np.random.default_rng(0)
    testg, testeffect_sgn, testtoporder = test_spec.g(testrng, testnvars)
    testdata = test_spec.mechanism(spec=test_spec, rng=testrng, g=testg, effect_sgn=testeffect_sgn,
                                   toporder=testtoporder, n_vars=testnvars)

    testdata = onp_standardize_data(testdata)

    print("true graph")
    print(testg)

    testpred = run_cam(
        42,
        testdata,
        dict(
            score="nonlinear",
            sel_method="gamboost",
            prune_method="gam",
            cutoff=0.001,
            score_test_data=0.2,
            variable_sel=False,
            pruning=False,
            # rank_fail_add=1e-3,
            # rank_check_tol=1e-6,
        )
    )

    print(testpred)
