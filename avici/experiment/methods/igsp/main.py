import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"


import random
import numpy as np
import pandas as pd

# try:
#     from .igsp import prepare_igsp, format_to_igsp, igsp
# except ImportError:
#     from igsp import prepare_igsp, format_to_igsp, igsp

from avici.experiment.methods.igsp.igsp import prepare_igsp, format_to_igsp, igsp



def run_igsp(seed, data, config):

    np.random.seed(seed)
    random.seed(seed)

    if "alpha_inv" not in config:
        config["alpha_inv"] = config["alpha"]

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

    # existing code
    nodes, obs_samples, iv_samples_list, targets_list = format_to_igsp(pd.DataFrame(x), pd.DataFrame(masks), regimes)

    ci_tester, invariance_tester = prepare_igsp(obs_samples,
                                                iv_samples_list, targets_list,
                                                config["alpha"], config["alpha_inv"], config["ci_test"])

    # Run IGSP
    setting_list = [dict(interventions=targets) for targets in targets_list]
    est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester, nruns=5)
    dag = est_dag.to_amat()[0]

    return dict(g_edges=dag)


if __name__ == "__main__":

    from avici.utils.parse import load_data_config
    from avici.synthetic.utils import onp_standardize_data

    # test_spec = load_data_config("config/linear_additive-0.yaml")["data"]["train"][0]
    # test_spec = load_data_config("config/sergio-0.yaml")["data"]["train"][0]
    test_spec = load_data_config("experiments/test/data.yaml")["data"]["evaluation"][0]
    testnvars = 10

    testrng = np.random.default_rng(0)
    testg, testeffect_sgn, testtoporder = test_spec.g(testrng, testnvars)
    testdata = test_spec.mechanism(spec=test_spec, rng=testrng, g=testg, effect_sgn=testeffect_sgn,
                                   toporder=testtoporder, n_vars=testnvars)

    testdata = onp_standardize_data(testdata)

    print("true graph")
    print(testg)

    testpred = run_igsp(
        42,
        testdata,
        dict(
            alpha=1e-3,
            alpha_inv=1e-3,
            ci_test="gaussian",
            # ci_test="hsic",
            # ci_test="kci",
            rank_fail_add=1e-6,
            rank_check_tol=1e-6,
        )
    )

    print(testpred)
