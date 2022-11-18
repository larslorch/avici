import math
import functools
import time
from collections import defaultdict

import jax
import jax.random as random
import jax.numpy as jnp
import pandas as pd

from avici.utils.data import onp_subbatch
from avici.utils.data_jax import jax_get_x

from avici.metrics import shd, threshold_metrics, is_cyclic, classification_metrics, n_edges


def print_header(strg):
    print("\n" + "=" * 25 + f" {strg} " + "=" * 25, flush=True)


def update_ave(ave_d, d, n=1, is_mean=False):
    for k, v in d.items():
        ave_d[("__ctr__", k)] += n
        if is_mean:
            ave_d[k] += v * n
        else:
            ave_d[k] += v
    return ave_d


def retrieve_ave(ave_d):
    out = {}
    for k, v in ave_d.items():
        # check if `k` is a ctr element
        if isinstance(k, tuple) and k[0] == "__ctr__":
            continue
        # process value `v`
        try:
            v_val = v.item()
        # distributed training case with `pmap`: retrieve 1 device replicate
        except TypeError:
            v_val = v[0].item()
        # not an array
        except AttributeError:
            v_val = v
        assert ("__ctr__", k) in ave_d.keys()
        out[k] = v_val / ave_d[("__ctr__", k)]
    return out


@functools.partial(jax.jit, static_argnums=(0,))
def eval_batch_helper(model, params, dual, batch, key, step, decision_thresholds=(0.5,)):
    """jit-forward passes of net on a batch"""

    # loss
    key, subk = random.split(key)
    loss = model.loss(params, dual, subk, batch, step, False)[0]

    # graph preds
    x = jax_get_x(batch)
    g_edges_prob = model.infer_edge_probs(params, x)
    g_edges = {f"{thres}".replace(".", "_"): (g_edges_prob > thres).astype(jnp.int32) for thres in decision_thresholds}

    return {
        f"loss": loss,
        f"dual": dual,
        f"g_edges_prob": g_edges_prob,
        f"g_edges": g_edges,
    }


def evaluate(model, state, ds, step, detailed_log=False, log_metrics=("auprc", "auroc", "f1_0_5", "cyclic_0_5")):

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

                        # loss
                        scalars.update({f"loss{suffix}": preds["loss"]})

                        for j in range(batch_size):

                            # shd
                            # at thresholds
                            for thres, g_edges in preds["g_edges"].items():
                                scalars[f"shd_{thres}{suffix}"] += \
                                    shd(subbatch["g"][j], g_edges[j]) / batch_size

                            # cyclicity
                            # at thresholds
                            for thres, g_edges in preds["g_edges"].items():
                                scalars[f"edges_{thres}{suffix}"] += \
                                    n_edges(g_edges[j]) / batch_size
                                scalars[f"cyclic_{thres}{suffix}"] += \
                                    float(is_cyclic(g_edges[j])) / batch_size

                            # precision metrics
                            # at thresholds
                            for thres, g_edges in preds["g_edges"].items():
                                metrics = classification_metrics(subbatch["g"][j], g_edges[j])
                                for k, v in metrics.items():
                                    scalars[f"{k}_{thres}{suffix}"] += v / batch_size

                            #probabilistic metrics
                            metrics = threshold_metrics(subbatch["g"][j], preds["g_edges_prob"][j])
                            for k, v in metrics.items():
                                scalars[f"{k}{suffix}"] += v / batch_size

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

    return results
