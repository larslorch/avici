import math
import functools
import time
from collections import defaultdict
import wandb

import jax
import jax.random as random
import jax.numpy as jnp
import pandas as pd

import numpy as onp

from avici.utils.plot import visualize_mats
from avici.utils.experiment import update_ave, retrieve_ave
from avici.synthetic.utils import onp_subbatch
from avici.synthetic.utils_jax import jax_get_x

from avici.eval.metrics import shd, threshold_metrics, is_cyclic, classification_metrics, n_edges

THRESHOLDS = [0.5]

def get_data_shape(batch):
    return batch["x"].shape[-3], batch["x"].shape[-2]


@functools.partial(jax.jit, static_argnums=(0,))
def eval_batch_helper(model, params, dual, batch, key, step):
    """jit-forward passes of net on a batch"""

    # loss
    key, subk = random.split(key)
    loss = model.loss(params, dual, subk, batch, step, False)[0]

    # graph preds
    x = jax_get_x(batch)

    g_mixture_prob = model.infer_mixture_probs(params, x)
    g_edges_prob = model.infer_edge_probs(params, x)
    g_edge_pair_prob = model.infer_edge_pair_probs(params, x)

    g_edges = {f"{thres}".replace(".", "_"): (g_edges_prob > thres).astype(jnp.int32) for thres in THRESHOLDS}
    g_edge_pairs = {f"{thres}".replace(".", "_"): (g_edge_pair_prob > thres).astype(jnp.int32) for thres in THRESHOLDS}

    return {
        f"loss": loss,
        f"dual": dual,
        f"g_mixture_prob": g_mixture_prob,
        f"g_edges_prob": g_edges_prob,
        f"g_edge_pair_prob": g_edge_pair_prob,
        f"g_edges": g_edges,
        f"g_edge_pairs": g_edge_pairs,
    }


def visualize_batch(model, state, ref_batches, step, max_cols=None, max_rows=10, size=0.5, diff=False):

    # extract mats
    mats = colnames = []
    for j, batch in enumerate(ref_batches):
        if max_rows is not None and j >= max_rows:
            break
        row = [batch["g"][0]]
        d = batch["g"].shape[-1]
        preds = eval_batch_helper(model, state.params, state.dual, batch, state.rng, step)
        prob1, probmixt = preds["g_mixture_prob"]
        if probmixt is not None:
            for k in range(probmixt.shape[-1]):
                row.append(prob1[0, k, :, :])
            if j == 0:
                colnames = ["true"] + [f"{c}" for c in range(probmixt.shape[-1])]
        else:
            row.append(prob1)
            if j == 0:
                colnames = ["true", "edge probs"]
        mats.append(row)
    rownames = list(range(len(mats)))

    # plotting
    ref_img = visualize_mats(mats, colnames=colnames, rownames=rownames, max_cols=max_cols,
                             max_rows=max_rows, size=size * (d ** 0.333), diff=diff)
    return ref_img


def evaluate(model, state, ds, ref_batches, step, detailed_log=False,
             log_metrics=("auprc", "auroc", "f1_0_5", "cyclic_0_5"),
             agg_metrics=("auprc", "f1_0_5"),
             visualize_diff=False,
             ):

    """Eval model on a full tree of data iterators at the given params/state"""

    results = {}
    df = pd.DataFrame()
    all_test_iters = ds.make_test_datasets()
    if len(all_test_iters) == 0:
        return {}

    # evaluate each test iterator
    for descr, test_iters in all_test_iters.items():
        for n_vars, it in test_iters.items():

            scalars_aux_stratified = defaultdict(lambda: defaultdict(float))
            t0 = time.time()
            key = state.rng

            # get batch of observations
            main_batch_size = None
            for batch in it:

                assert batch["x_obs"].ndim == 4 and batch["x_int"].ndim == 4 and batch["g"].ndim == 3

                # avoid recompilation on last remainder batch
                batch_size = batch["g"].shape[0] # size of batch
                if main_batch_size is not None and batch_size != main_batch_size:
                    break
                else:
                    main_batch_size = batch_size

                # evaluate batch for variable number of observations
                ratio_obs_int = batch["n_observations_obs"][0] / (batch["n_observations_obs"][0] + batch["n_observations_int"][0])
                n_obs_max = batch["n_observations_obs"][0] + batch["n_observations_int"][0]
                n_obs_l = [i for i in [50, 200, 1000] if i < n_obs_max] + [n_obs_max]
                # careful: every n_obs needs 1 compile

                for n in n_obs_l:

                    n_obs, n_int = math.floor(n * ratio_obs_int), math.ceil(n * (1 - ratio_obs_int))
                    subbatch = onp_subbatch(None, batch, n_obs, n_int)

                    scalars = defaultdict(float)

                    # predictions for normal checkpoint and for polyak checkpoint
                    for suffix, params in [("", state.params), ("_polyak", state.ave_params)]:

                        key, subk = random.split(key)
                        preds = eval_batch_helper(model, params, state.dual, subbatch, subk, step)
                        # n_graph_samples = preds["g_samples"].shape[-3]

                        # loss
                        scalars.update({
                            f"loss{suffix}": preds["loss"],
                        })

                        for j in range(batch_size):

                            g_edge_pairs_j = jnp.einsum("ij,jk->ijk", subbatch["g"][j], subbatch["g"][j])

                            ####### shd
                            # at thresholds
                            for thres, g_edges in preds["g_edges"].items():
                                scalars[f"shd_{thres}{suffix}"] += \
                                    shd(subbatch["g"][j], g_edges[j]) / batch_size

                            # # in expectation (empirical mean of samples)
                            # for ii in range(n_graph_samples):
                            #     scalars[f"e-shd{suffix}"] += \
                            #         shd(subbatch["g"][j], preds["g_samples"][j, ii]) / (batch_size * n_graph_samples)

                            ####### cyclicity
                            # at thresholds
                            for thres, g_edges in preds["g_edges"].items():
                                scalars[f"edges_{thres}{suffix}"] += \
                                    n_edges(g_edges[j]) / batch_size
                                scalars[f"cyclic_{thres}{suffix}"] += \
                                    is_cyclic(g_edges[j]) / batch_size

                            # # in expectation (empirical mean of samples)
                            # for ii in range(n_graph_samples):
                            #     scalars[f"e-edges{suffix}"] += \
                            #         n_edges(preds["g_samples"][j, ii]) / (batch_size * n_graph_samples)
                            #     scalars[f"e-cyclic{suffix}"] += \
                            #         is_cyclic(preds["g_samples"][j, ii]) / (batch_size * n_graph_samples)

                            ####### precision metrics
                            # at thresholds
                            for thres, g_edges in preds["g_edges"].items():
                                metrics = classification_metrics(subbatch["g"][j], g_edges[j])
                                for k, v in metrics.items():
                                    scalars[f"{k}_{thres}{suffix}"] += v / batch_size

                            for thres, g_edge_pairs in preds["g_edge_pairs"].items():
                                metrics = classification_metrics(g_edge_pairs_j, g_edge_pairs[j])
                                for k, v in metrics.items():
                                    scalars[f"pair-{k}_{thres}{suffix}"] += v / batch_size

                            # # in expectation (empirical mean of samples)
                            # for ii in range(n_graph_samples):
                            #     metrics = classification_metrics(subbatch["g"][j], preds["g_samples"][j, ii])
                            #     for k, v in metrics.items():
                            #         scalars[f"e-{k}{suffix}"] += v / (batch_size * n_graph_samples)

                            ####### probabilistic metrics
                            metrics = threshold_metrics(subbatch["g"][j], preds["g_edges_prob"][j])
                            for k, v in metrics.items():
                                scalars[f"{k}{suffix}"] += v / batch_size

                            metrics = threshold_metrics(g_edge_pairs_j, preds["g_edge_pair_prob"][j])
                            for k, v in metrics.items():
                                scalars[f"pair-{k}{suffix}"] += v / batch_size


                    ## save scalars
                    scalars_aux_stratified[n] = update_ave(scalars_aux_stratified[n], scalars, n=batch_size, is_mean=True)

            eval_time = time.time() - t0

            # compute averages and log scalars to df
            scalars_ave_stratified = {i: retrieve_ave(aux) for i, aux in scalars_aux_stratified.items()}
            for n, scalars_ave in scalars_ave_stratified.items():
                df = df.append({
                    **scalars_ave,
                    "descr": descr,
                    "d": n_vars,
                    "n": n,
                    "eval_time": eval_time,
                }, ignore_index=True)

            # visualize predictions
            refs = ref_batches[descr][n_vars]
            results.update({f"ref_preds/{descr}-d={n_vars}": visualize_batch(model, state, refs, step)})
            if visualize_diff:
                results.update({f"ref_preds_diff/{descr}-d={n_vars}": visualize_batch(model, state, refs, step, diff=True)})

            # visualize calibration


    # collect results for max number of observations
    metrics_strs = df.columns.drop(["descr", "n", "d"], errors="ignore")
    if log_metrics is not None and not detailed_log:
        metrics_strs = metrics_strs\
            .drop([s for s in metrics_strs if not any([s in k for k in log_metrics])], errors="ignore") \
            .drop([s for s in metrics_strs if "polyak" in s], errors="ignore")

    for descr in df["descr"].unique():
        df_descr = df.loc[df["descr"] == descr]
        n_max = int(df_descr["n"].max())
        d_min = int(df_descr["d"].min())
        d_max = int(df_descr["d"].max())
        for d in df_descr["d"].unique():
            if not detailed_log and d not in [d_min, d_max]:
                continue
            results_train_settings = df_descr[(df_descr["n"] == n_max) & (df_descr["d"] == d)]
            assert not results_train_settings.empty, f"No results for {descr} d={d} n_max={n_max}"
            for metric_str in metrics_strs:
                results.update({f"{descr}/d={d}/{metric_str}": results_train_settings[metric_str].item()})

    # visualize performance when varying n_vars and n_obs
    # do not use polyak or point estimate metrics
    metrics_strs_agg = df.columns\
        .drop(["descr", "n", "d"], errors="ignore")\
        .drop([s for s in df.columns if not any([s in k for k in agg_metrics])], errors="ignore")\
        .drop([s for s in df.columns if "polyak" in s], errors="ignore")

    for descr in df["descr"].unique():
        for metric_str in metrics_strs_agg.drop(["eval_time", "loss", "shd"], errors="ignore"):
            df_metric = df.loc[df["descr"] == descr, ["d", "n", metric_str]]

            # plot: metric(n_vars) for a given number of observations
            # for n_obs in df_metric["n"].unique():
            n_max = int(df_metric["n"].max())
            for n in [n_max]:
                n_vars_arr = df_metric[df_metric["n"] == n]["d"].to_numpy()
                metric_arr = df_metric[df_metric["n"] == n][metric_str].to_numpy()
                if len(n_vars_arr) == 1:
                    continue

                table = wandb.Table(data=[[x, y] for (x, y) in zip(n_vars_arr, metric_arr)],
                                    columns=["vars", metric_str])
                results.update({f"{descr}/agg/{metric_str}-for-n={int(n)}":
                    wandb.plot.bar(table, "vars", metric_str, title=f"{descr}  {metric_str}(vars=X, observations={int(n)})")})

            # plot: metric(n_observations) for a given number of variables
            d_max = int(df_metric["d"].max())
            for d in df_metric["d"].unique():
                if not detailed_log and d != d_max:
                    continue
                n_arr = df_metric[df_metric["d"] == d]["n"].to_numpy()
                metric_arr = df_metric[df_metric["d"] == d][metric_str].to_numpy()
                if len(n_arr) == 1:
                    continue

                table = wandb.Table(data=[[x, y] for (x, y) in zip(n_arr, metric_arr)],
                                    columns=["observations", metric_str])
                results.update({f"{descr}/agg/{metric_str}-for-d={int(d)}":
                    wandb.plot.bar(table, "observations", metric_str, title=f"{descr}  {metric_str}(vars={int(d)}, observations=X)")})

    return results