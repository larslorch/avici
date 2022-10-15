import jax.numpy as jnp
from jax import random

from dibs.models.graph import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, UniformDAGDistributionRejection
from dibs.graph_utils import graph_to_mat

from dibs.models import LinearGaussian, BGe, DenseNonlinearGaussian

from typing import Any, NamedTuple


class Data(NamedTuple):
    """ NamedTuple for structuring simulated synthetic data and their ground
    truth generative model

    Args:
        passed_key (ndarray): ``jax.random`` key passed *into* the function generating this object
        n_vars (int): number of variables in model
        n_observations (int): number of observations in ``x`` and used to perform inference
        n_ho_observations (int): number of held-out observations in ``x_ho``
            and elements of ``x_interv`` used for evaluation
        g (ndarray): ground truth DAG
        theta (Any): ground truth parameters
        x (ndarray): i.i.d observations from the model of shape ``[n_observations, n_vars]``
        x_ho (ndarray): i.i.d observations from the model of shape ``[n_ho_observations, n_vars]``
        x_interv (list): list of (interv dict, i.i.d observations)

    """

    passed_key: Any

    n_vars: int
    n_observations: int
    n_ho_observations: int

    g: Any
    theta: Any
    x: Any
    x_ho :Any
    x_interv: Any


def make_synthetic_bayes_net(*,
    key,
    n_vars,
    graph_dist,
    generative_model,
    n_observations=100,
    n_ho_observations=100,
    n_intervention_sets=10,
    perc_intervened=0.1,
):
    """
    Returns an instance of :class:`~dibs.metrics.Target` for evaluation of a method on
    a ground truth synthetic causal Bayesian network

    Args:
        key (ndarray): rng key
        n_vars (int): number of variables
        graph_dist (Any): graph model object. For example: :class:`~dibs.models.ErdosReniDAGDistribution`
        generative_model (Any): BN model object for generating the observations. For example: :class:`~dibs.models.LinearGaussian`
        n_observations (int): number of observations generated for posterior inference
        n_ho_observations (int): number of held-out observations generated for evaluation
        n_intervention_sets (int): number of different interventions considered overall
            for generating interventional data
        perc_intervened (float): percentage of nodes intervened upon (clipped to 0) in
            an intervention.

    Returns:
        :class:`~dibs.target.Data`:
        synthetic ground truth generative DAG and parameters as well observations sampled from the model
    """

    # remember random key
    passed_key = key.copy()

    # generate ground truth observations
    key, subk = random.split(key)
    g_gt = graph_dist.sample_G(subk)
    g_gt_mat = jnp.array(graph_to_mat(g_gt))

    key, subk = random.split(key)
    theta = generative_model.sample_parameters(key=subk, n_vars=n_vars)

    key, subk = random.split(key)
    x = generative_model.sample_obs(key=subk, n_samples=n_observations, g=g_gt, theta=theta)

    key, subk = random.split(key)
    x_ho = generative_model.sample_obs(key=subk, n_samples=n_ho_observations, g=g_gt, theta=theta)

    # 10 random 0-clamp interventions where `perc_interv` % of nodes are intervened on
    # list of (interv dict, x)
    x_interv = []
    for idx in range(n_intervention_sets):
    
        # random intervention
        key, subk = random.split(key)
        n_interv = jnp.ceil(n_vars * perc_intervened).astype(jnp.int32)
        interv_targets = random.choice(subk, n_vars, shape=(n_interv,), replace=False)
        interv = {int(k): 0.0 for k in interv_targets}

        # observations from p(x | theta, G, interv) [n_samples, n_vars]
        key, subk = random.split(key)
        x_interv_ = generative_model.sample_obs(key=subk, n_samples=n_observations, g=g_gt, theta=theta, interv=interv)
        x_interv.append((interv, x_interv_))

    # return and save generated target object
    data = Data(
        passed_key=passed_key,
        n_vars=n_vars,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations,
        g=g_gt_mat,
        theta=theta,
        x=x,
        x_ho=x_ho,
        x_interv=x_interv,
    )
    return data
    

def make_graph_model(*, n_vars, graph_prior_str, edges_per_node=2):
    """
    Instantiates graph model

    Args:
        n_vars (int): number of variables in graph
        graph_prior_str (str): specifier for random graph model; choices: ``er``, ``sf``
        edges_per_node (int): number of edges per node (in expectation when applicable)

    Returns:
        Object representing graph model. For example :class:`~dibs.models.ErdosReniDAGDistribution` or :class:`~dibs.models.ScaleFreeDAGDistribution`
    """
    if graph_prior_str == 'er':
        graph_dist = ErdosReniDAGDistribution(
            n_vars=n_vars, 
            n_edges_per_node=edges_per_node)

    elif graph_prior_str == 'sf':
        graph_dist = ScaleFreeDAGDistribution(
            n_vars=n_vars,
            n_edges_per_node=edges_per_node)

    else:
        assert n_vars <= 5, "Naive uniform DAG sampling only possible up to 5 nodes"
        graph_dist = UniformDAGDistributionRejection(
            n_vars=n_vars)

    return graph_dist


def make_linear_gaussian_equivalent_model(*, key, n_vars=20, graph_prior_str='sf', 
    obs_noise=0.1, mean_edge=0.0, sig_edge=1.0, min_edge=0.5, n_observations=100,
    n_ho_observations=100):
    """
    Samples a synthetic linear Gaussian BN instance 
    with Bayesian Gaussian equivalent (BGe) marginal likelihood 
    as inference model to weight each DAG in an MEC equally

    By marginalizing out the parameters, the BGe model does not 
    allow inferring the parameters :math:`\\Theta`.
    
    Args:
        key (ndarray): rng key
        n_vars (int): number of variables i
        n_observations (int): number of iid observations of variables
        n_ho_observations (int): number of iid held-out observations of variables
        graph_prior_str (str): graph prior (``er`` or ``sf``)
        obs_noise (float): observation noise
        mean_edge (float): edge weight mean
        sig_edge (float): edge weight stddev
        min_edge (float): min edge weight enforced by constant shift of sampled parameter
    
    Returns:
        tuple(:class:`~dibs.models.BGe`, :class:`~dibs.target.Data`):
        BGe inference model and observations from a linear Gaussian generative process
    """

    # init models
    graph_dist = make_graph_model(n_vars=n_vars, graph_prior_str=graph_prior_str)

    generative_model = LinearGaussian(
        graph_dist=graph_dist, obs_noise=obs_noise,
        mean_edge=mean_edge, sig_edge=sig_edge,
        min_edge=min_edge)

    inference_model = BGe(graph_dist=graph_dist)

    # sample synthetic BN and observations
    key, subk = random.split(key)
    data = make_synthetic_bayes_net(
        key=subk, n_vars=n_vars,
        graph_dist=graph_dist,
        generative_model=generative_model,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations)

    return data, inference_model


def make_linear_gaussian_model(*, key, n_vars=20, graph_prior_str='sf', 
    obs_noise=0.1, mean_edge=0.0, sig_edge=1.0, min_edge=0.5, n_observations=100,
    n_ho_observations=100):
    """
    Samples a synthetic linear Gaussian BN instance 

    Args:
        key (ndarray): rng key
        n_vars (int): number of variables
        n_observations (int): number of iid observations of variables
        n_ho_observations (int): number of iid held-out observations of variables
        graph_prior_str (str): graph prior (`er` or `sf`)
        obs_noise (float): observation noise
        mean_edge (float): edge weight mean
        sig_edge (float): edge weight stddev
        min_edge (float): min edge weight enforced by constant shift of sampled parameter

    Returns:
        tuple(:class:`~dibs.models.LinearGaussian`, :class:`~dibs.target.Data`):
        linear Gaussian inference model and observations from a linear Gaussian generative process
    """

    # init models
    graph_dist = make_graph_model(n_vars=n_vars, graph_prior_str=graph_prior_str)

    generative_model = LinearGaussian(
        graph_dist=graph_dist, obs_noise=obs_noise,
        mean_edge=mean_edge, sig_edge=sig_edge,
        min_edge=min_edge)

    inference_model = LinearGaussian(
        graph_dist=graph_dist, obs_noise=obs_noise,
        mean_edge=mean_edge, sig_edge=sig_edge,
        min_edge=min_edge)

    # sample synthetic BN and observations
    key, subk = random.split(key)
    data = make_synthetic_bayes_net(
        key=subk, n_vars=n_vars,
        graph_dist=graph_dist,
        generative_model=generative_model,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations)

    return data, inference_model


def make_nonlinear_gaussian_model(*, key, n_vars=20, graph_prior_str='sf', 
    obs_noise=0.1, sig_param=1.0, hidden_layers=[5,], n_observations=100,
    n_ho_observations=100):
    """
    Samples a synthetic nonlinear Gaussian BN instance 
    where the local conditional distributions are parameterized
    by fully-connected neural networks.

    Args:
        key (ndarray): rng key
        n_vars (int): number of variables
        n_observations (int): number of iid observations of variables
        n_ho_observations (int): number of iid held-out observations of variables
        graph_prior_str (str): graph prior (`er` or `sf`)
        obs_noise (float): observation noise
        sig_param (float): stddev of the BN parameters,
            i.e. here the neural net weights and biases
        hidden_layers (list): list of ints specifying the hidden layer (sizes)
            of the neural nets parameterizatin the local condtitionals
    
    Returns:
        tuple(:class:`~dibs.models.DenseNonlinearGaussian`, :class:`~dibs.metrics.Target`):
        nonlinear Gaussian inference model and observations from a nonlinear Gaussian generative process
    """

    # init models
    graph_dist = make_graph_model(n_vars=n_vars, graph_prior_str=graph_prior_str)

    generative_model = DenseNonlinearGaussian(
        obs_noise=obs_noise, sig_param=sig_param,
        hidden_layers=hidden_layers, graph_dist=graph_dist)

    inference_model = DenseNonlinearGaussian(
        obs_noise=obs_noise, sig_param=sig_param,
        hidden_layers=hidden_layers, graph_dist=graph_dist)

    # sample synthetic BN and observations
    key, subk = random.split(key)
    data = make_synthetic_bayes_net(
        key=subk, n_vars=n_vars,
        graph_dist=graph_dist,
        generative_model=generative_model,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations)

    return data, inference_model
