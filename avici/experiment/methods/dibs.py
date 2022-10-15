# from dibs.inference import MarginalDiBS
# from dibs.inference import JointDiBS

from dibs.inference_interventional import IntervMarginalDiBS
from dibs.inference_interventional import IntervJointDiBS

from dibs.models.graph import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, UniformDAGDistributionRejection
from dibs.graph_utils import graph_to_mat

from dibs.models import LinearGaussian, BGe, DenseNonlinearGaussian

import jax
import math
import jax.random as random
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
import jax.tree_util

import numpy as onp

def _dibs_plus(log_weights, gs):
    log_weights = log_weights - logsumexp(log_weights)
    log_g_edges_prob, log_g_edges_prob_sgn = logsumexp(
        log_weights, b=gs.astype(log_weights.dtype), axis=0, return_sign=True)
    g_edges_prob = log_g_edges_prob_sgn * jnp.exp(log_g_edges_prob)
    return g_edges_prob

def _run_dibs(seed, data, config, inference_model, log_lik_fun, marginal, heldout_split=0.0):

    key = random.PRNGKey(seed)

    # concatenate all observations
    x_concat = jnp.concatenate([data["x_obs"], data["x_int"]], axis=-3)
    x_full, interv_mask_full = x_concat[..., 0], x_concat[..., 1]

    # to calibrate hyperparameters, split the data before running the algorithm
    if heldout_split > 0.0:
        key = random.PRNGKey(seed)
        perm = random.permutation(key, x_full.shape[0])
        cutoff = math.floor(x_full.shape[0] * heldout_split)
        train_idx = perm[cutoff:]
        val_idx = perm[:cutoff]

        x = x_full[train_idx]
        interv_mask = interv_mask_full[train_idx]
        heldout_data = x_full[val_idx], interv_mask_full[val_idx]

        print(f"Splitting data into train size {train_idx.shape} and val size {val_idx.shape}")
    else:
        x = x_full
        interv_mask = interv_mask_full
        heldout_data = None

    # sample DAG and parameter particles
    key, subk = random.split(key)
    if marginal:
        dibs_alg = IntervMarginalDiBS(
            x=x,
            interv_mask=interv_mask,
            inference_model=inference_model,
            optimizer_param=dict(stepsize=config.get("stepsize", 0.005)),
            kernel_param=dict(h=config.get("h_latent", 5.0)),
            alpha_linear=config.get("alpha_linear", 1.0),
            grad_estimator_z=config.get("grad_estimator_z", "score"),
        )
        gs = dibs_alg.sample(key=subk, n_particles=config["n_particles"], steps=config["steps"])
        thetas = jnp.zeros(gs.shape[0]) #dummy
    else:
        dibs_alg = IntervJointDiBS(
            x=x,
            interv_mask=interv_mask,
            inference_model=inference_model,
            optimizer_param=dict(stepsize=config.get("stepsize", 0.005)),
            kernel_param=dict(h_latent=config.get("h_latent", 5.0), h_theta=config.get("h_theta", 500)),
            alpha_linear=config.get("alpha_linear", 1.0)
        )
        gs, thetas = dibs_alg.sample(key=subk, n_particles=config["n_particles"], steps=config["steps"])

    log_weights = vmap(log_lik_fun, (0, 0, None, None))(gs, thetas, x, interv_mask)

    if config.get("dibs_plus", False):
        g_edges_prob = _dibs_plus(log_weights[..., None, None], gs)
    else:
        g_edges_prob = gs.mean(0)

    pred = dict(g_edges_prob=onp.array(jax.device_get(g_edges_prob), dtype=onp.float32),
                g_edges=onp.array(jax.device_get((g_edges_prob > 0.5)), dtype=onp.int32))


    assert not jnp.isnan(gs).any(), f"Got NaNs in G: {gs}"
    assert not jnp.array(jax.tree_util.tree_map(lambda e: jnp.isnan(e).any(), jax.tree_util.tree_leaves(thetas))).any(), f"Got NaNs in Theta: {thetas}"

    # compute validation score if heldout data is given
    if heldout_data is not None:
        val_x, val_interv_mask = heldout_data
        holl = vmap(log_lik_fun, (0, 0, None, None))(gs, thetas, val_x, val_interv_mask)

        # sum_G p(G | D) log(p(x | G))
        if config.get("dibs_plus", False):
            ave_holl = _dibs_plus(log_weights, holl)
        else:
            ave_holl = holl.mean(0)

        pred["heldout_score"] = onp.array(jax.device_get(ave_holl), dtype=onp.float32)

    return pred

def run_dibs(seed, data, config, heldout_split=0.0):

    graph_dist = ScaleFreeDAGDistribution(n_vars=data["x_obs"].shape[-2], n_edges_per_node=2)
   
    if config["likelihood"] == "bge":
        inference_model = BGe(
            graph_dist=graph_dist,
            alpha_mu=config.get("alpha_mu", None),
        )
        marginal = True
        def log_lik_fun(g, _, x_, mask_):
            return inference_model.log_marginal_likelihood(x=x_, g=g, interv_targets=mask_)

    elif config["likelihood"] == "linear":
        inference_model = LinearGaussian(
            graph_dist=graph_dist,
            obs_noise=config.get("obs_noise", 0.1),
            mean_edge=0.0,
            sig_edge=config.get("sig_param", 1.0),
        )
        marginal = False
        def log_lik_fun(g, theta, x_, mask_):
            return inference_model.log_likelihood_mask(x=x_, theta=theta, g=g, interv_targets=mask_)

    elif config["likelihood"] == "nn":
        inference_model = DenseNonlinearGaussian(
            graph_dist=graph_dist,
            obs_noise=config.get("obs_noise", 0.1),
            activation=config.get("activation", "sigmoid"),
            sig_param=config.get("sig_param", 1.0),
            hidden_layers=[5, ]
        )
        marginal = False
        def log_lik_fun(g, theta, x_, mask_):
            return inference_model.log_likelihood_mask(x=x_, theta=theta, g=g, interv_targets=mask_)

    else:
        raise KeyError(f"Unknown DiBS likelihood `{config['likelihood']}`")

    return _run_dibs(seed, data, config, inference_model, log_lik_fun, marginal, heldout_split=heldout_split)



