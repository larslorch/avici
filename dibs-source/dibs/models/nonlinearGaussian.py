import os
import numpy as onp

import jax.numpy as jnp
from jax import vmap
from jax import random
from jax.scipy.stats import norm as jax_normal
from jax.tree_util import tree_map, tree_reduce
from jax.nn.initializers import normal
try:
    import jax.example_libraries.stax as stax
    from jax.example_libraries.stax import Dense, Sigmoid, LeakyRelu, Relu, Tanh
except ImportError:
    # for jax <= 2.24
    import jax.experimental.stax as stax
    from jax.experimental.stax import Dense, Sigmoid, LeakyRelu, Relu, Tanh

from dibs.graph_utils import graph_to_mat
from dibs.utils.tree import tree_shapes


def DenseNoBias(out_dim, W_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer _without_ bias"""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        W = W_init(rng, (input_shape[-1], out_dim))
        return output_shape, (W, )

    def apply_fun(params, inputs, **kwargs):
        W, = params
        return jnp.dot(inputs, W)

    return init_fun, apply_fun


def makeDenseNet(*, hidden_layers, sig_weight, sig_bias, bias=True, activation='relu'):
    """
    Generates functions defining a fully-connected NN
    with Gaussian initialized parameters

    Args:
        hidden_layers (list): list of ints specifying the dimensions of the hidden sizes
        sig_weight: std dev of weight initialization
        sig_bias: std dev of weight initialization
        bias: bias of linear layer
        activation: activation function str; choices: `sigmoid`, `tanh`, `relu`, `leakyrelu`
    
    Returns:
        stax.serial neural net object
    """

    # features: [hidden_layers[0], hidden_layers[0], ..., hidden_layers[-1], 1]
    if activation == 'sigmoid':
        f_activation = Sigmoid
    elif activation == 'tanh':
        f_activation = Tanh
    elif activation == 'relu':
        f_activation = Relu
    elif activation == 'leakyrelu':
        f_activation = LeakyRelu
    else:
        raise KeyError(f'Invalid activation function `{activation}`')

    modules = []
    if bias:
        for dim in hidden_layers:
            modules += [
                Dense(dim, W_init=normal(stddev=sig_weight),
                        b_init=normal(stddev=sig_bias)),
                f_activation
            ]
        modules += [Dense(1, W_init=normal(stddev=sig_weight),
                            b_init=normal(stddev=sig_bias))]
    else:
        for dim in hidden_layers:
            modules += [
                DenseNoBias(dim, W_init=normal(stddev=sig_weight)),
                f_activation
            ]
        modules += [DenseNoBias(1, W_init=normal(stddev=sig_weight))]

    return stax.serial(*modules)
    

class DenseNonlinearGaussian:
    """
    Nonlinear Gaussian BN model corresponding to a nonlinaer structural equation model (SEM)
    with additive Gaussian noise.

    Each variable distributed as Gaussian with mean parameterized by a dense neural network (MLP)
    whose weights and biases are sampled from a Gaussian prior.
    The noise variance at each node is equal by default.

    Refer to http://proceedings.mlr.press/v108/zheng20a/zheng20a.pdf

    Args:
        graph_dist: Graph model defining prior :math:`\\log p(G)`. Object *has to implement the method*:
            ``unnormalized_log_prob_soft``.
            For example: :class:`~dibs.graph.ErdosReniDAGDistribution`
            or :class:`~dibs.graph.ScaleFreeDAGDistribution`
        hidden_layers (list): list of integers specifying the number of layers as well as their widths.
            For example: ``[8, 8]`` would correspond to 2 hidden layers with 8 neurons
        obs_noise (float, optional): variance of additive observation noise at nodes
        sig_param (float, optional): std dev of Gaussian parameter prior
        activation (str, optional): identifier for activation function.
            Choices: ``sigmoid``, ``tanh``, ``relu``, ``leakyrelu``

    """
    def __init__(self, *, graph_dist, hidden_layers, obs_noise=0.1, sig_param=1.0, activation='relu', bias=True):
        super(DenseNonlinearGaussian, self).__init__()

        self.graph_dist = graph_dist
        self.n_vars = graph_dist.n_vars
        self.obs_noise = obs_noise
        self.sig_param = sig_param
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.bias = bias

        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)

        # init single neural net function for one variable with jax stax
        self.nn_init_random_params, nn_forward = makeDenseNet(
            hidden_layers=self.hidden_layers, 
            sig_weight=self.sig_param,
            sig_bias=self.sig_param,
            activation=self.activation,
            bias=self.bias)
        
        # [?], [N, d] -> [N,]
        self.nn_forward = lambda theta, x: nn_forward(theta, x).squeeze(-1)
        
        # vectorize init and forward functions
        self.eltwise_nn_init_random_params = vmap(self.nn_init_random_params, (0, None), 0)
        self.double_eltwise_nn_init_random_params = vmap(self.eltwise_nn_init_random_params, (0, None), 0)
        self.triple_eltwise_nn_init_random_params = vmap(self.double_eltwise_nn_init_random_params, (0, None), 0)
        
        # [d2, ?], [N, d] -> [N, d2]
        self.eltwise_nn_forward = vmap(self.nn_forward, (0, None), 1)

        # [d2, ?], [d2, N, d] -> [N, d2]
        self.double_eltwise_nn_forward = vmap(self.nn_forward, (0, 0), 1)


    def get_theta_shape(self, *, n_vars):
        """Returns tree shape of the parameters of the neural networks

        Args:
            n_vars (int): number of variables in model

        Returns:
            PyTree of parameter shape
        """
        
        dummy_subkeys = jnp.zeros((n_vars, 2), dtype=jnp.uint32)
        _, theta = self.eltwise_nn_init_random_params(dummy_subkeys, (n_vars, )) # second arg is `input_shape` of NN forward pass

        theta_shape = tree_shapes(theta)
        return theta_shape


    def sample_parameters(self, *, key, n_vars, n_particles=0, batch_size=0):
        """Samples batch of random parameters given dimensions of graph from :math:`p(\\Theta | G)`

        Args:
            key (ndarray): rng
            n_vars (int): number of variables in BN
            n_particles (int): number of parameter particles sampled
            batch_size (int): number of batches of particles being sampled

        Returns:
            Parameter PyTree with leading dimension(s) ``batch_size`` and/or ``n_particles``,
            dropping either dimension when equal to 0
        """
        shape = [d for d in (batch_size, n_particles, n_vars) if d != 0]
        subkeys = random.split(key, int(onp.prod(shape))).reshape(*shape, 2)

        if len(shape) == 1:
            _, theta = self.eltwise_nn_init_random_params(subkeys, (n_vars, ))

        elif len(shape) == 2:
            _, theta = self.double_eltwise_nn_init_random_params(subkeys, (n_vars, ))

        elif len(shape) == 3:
            _, theta = self.triple_eltwise_nn_init_random_params(subkeys, (n_vars, ))

        else:
            raise ValueError(f"invalid shape size for nn param initialization {shape}")
            
        # to float64
        prec64 = 'JAX_ENABLE_X64' in os.environ and os.environ['JAX_ENABLE_X64'] == 'True'
        theta = tree_map(lambda arr: arr.astype(jnp.float64 if prec64 else jnp.float32), theta)
        return theta


    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None):
        """Samples ``n_samples`` observations given graph ``g`` and parameters ``theta``
        by doing single forward passes in topological order

        Args:
            key (ndarray): rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta (Any): parameters
            interv (dict): intervention specification of the form ``{intervened node : clamp value}``

        Returns:
            observation matrix of shape ``[n_samples, n_vars]``
        """
        if interv is None:
            interv = {}
        if toporder is None:
            toporder = g.topological_sorting()

        n_vars = len(g.vs)
        x = jnp.zeros((n_samples, n_vars))

        key, subk = random.split(key)
        z = jnp.sqrt(self.obs_noise) * random.normal(subk, shape=(n_samples, n_vars))

        g_mat = graph_to_mat(g)

        # ancestral sampling
        # for simplicity, does d full forward passes for simplicity, which avoids indexing into python list of parameters
        for j in toporder:

            # intervention
            if j in interv.keys():
                x = x.at[:, j].set(interv[j])
                continue

            # regular ancestral sampling
            parents = g_mat[:, j].reshape(1, -1)

            has_parents = parents.sum() > 0

            if has_parents:
                # [N, d] = [N, d] * [1, d] mask non-parent entries of j
                x_msk = x * parents

                # [N, d] full forward pass
                means = self.eltwise_nn_forward(theta, x_msk)

                # [N,] update j only
                x = x.at[:, j].set(means[:, j] + z[:, j])
            else:
                x = x.at[:, j].set(z[:, j])

        return x

    """
    The following functions need to be functionally pure and @jit-able
    """

    def log_prob_parameters(self, *, theta, g):
        """Computes parameter prior :math:`\\log p(\\Theta | G)``
        In this model, the prior over weights and biases is zero-centered Gaussian.

        Arguments:
            theta (Any): parameter pytree
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``

        Returns:
            log prob
        """
        # compute log prob for each weight
        logprobs = tree_map(lambda leaf_theta: jax_normal.logpdf(x=leaf_theta, loc=0.0, scale=self.sig_param), theta)

        # mask logprobs of first layer weight matrix [0][0] according to graph
        # [d, d, dim_first_layer] = [d, d, dim_first_layer] * [d, d, 1]
        if self.bias:
            first_weight_logprobs, first_bias_logprobs = logprobs[0]
            logprobs[0] = (first_weight_logprobs * g.T[:, :, None], first_bias_logprobs)
        else:
            first_weight_logprobs,  = logprobs[0]
            logprobs[0] = (first_weight_logprobs * g.T[:, :, None],)

        # sum logprobs of every parameter tensor and add all up 
        return tree_reduce(jnp.add, tree_map(jnp.sum, logprobs))


    def log_likelihood(self, *, x, theta, g, interv_targets):
        """Computes likelihood :math:`p(D | G, \\Theta)`.
        In this model, the noise per observation and node is additive and Gaussian.

        Arguments:
            x (ndarray): observations of shape ``[n_observations, n_vars]``
            theta (Any): parameters PyTree
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
            interv_targets (ndarray): binary intervention indicator vector of shape ``[n_vars, ]``

        Returns:
            log prob
        """

        # [d2, N, d] = [1, N, d] * [d2, 1, d] mask non-parent entries of each j
        all_x_msk = x[None] * g.T[:, None]

        # [N, d2] NN forward passes for parameters of each param j 
        all_means = self.double_eltwise_nn_forward(theta, all_x_msk)

        # sum scores for all nodes and data
        return jnp.sum(
            jnp.where(
                # [1, n_vars]
                interv_targets[None, ...],
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=x, loc=all_means, scale=jnp.sqrt(self.obs_noise))
            )
        )

    def log_likelihood_mask(self, *, x, theta, g, interv_targets):
        """Computes likelihood :math:`p(D | G, \\Theta)`.
        In this model, the noise per observation and node is additive and Gaussian.

        Arguments:
            x (ndarray): observations of shape ``[n_observations, n_vars]``
            theta (Any): parameters PyTree
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
            interv_targets (ndarray): binary intervention indicator vector of shape ``[n_observations, n_vars]``

        Returns:
            log prob
        """
        assert interv_targets.ndim == 2

        # [d2, N, d] = [1, N, d] * [d2, 1, d] mask non-parent entries of each j
        all_x_msk = x[None] * g.T[:, None]

        # [N, d2] NN forward passes for parameters of each param j
        all_means = self.double_eltwise_nn_forward(theta, all_x_msk)

        # sum scores for all nodes and data
        return jnp.sum(
            jnp.where(
                # [n_observations, n_vars]
                interv_targets,
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=x, loc=all_means, scale=jnp.sqrt(self.obs_noise))
            )
        )

    """
    Distributions used by DiBS for inference:  prior and joint likelihood 
    """

    def log_graph_prior(self, g_prob):
        """ Computes graph prior :math:`\\log p(G)` given matrix of edge probabilities.
        This function simply binds the function of the provided ``self.graph_dist``.

        Arguments:
            g_prob (ndarray): edge probabilities in G of shape ``[n_vars, n_vars]``

        Returns:
            log prob
        """
        return self.graph_dist.unnormalized_log_prob_soft(soft_g=g_prob)


    def observational_log_joint_prob(self, g, theta, x, rng):
        """Computes observational joint likelihood :math:`\\log p(\\Theta, D | G)``

        Arguments:
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
            theta (Any): parameter PyTree
            x (ndarray): observational data of shape ``[n_observations, n_vars]``
            rng (ndarray): rng; skeleton for minibatching (TBD)

        Returns:
            log prob of shape ``[1,]``
        """
        log_prob_theta = self.log_prob_parameters(g=g, theta=theta)
        log_likelihood = self.log_likelihood(g=g, theta=theta, x=x, interv_targets=self.no_interv_targets)
        return log_prob_theta + log_likelihood



    def interventional_log_joint_prob(self, g, theta, x, interv_targets, rng):
        """Computes interventional joint likelihood :math:`\\log p(\\Theta, D | G)``

        Arguments:
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
            theta (Any): parameter PyTree
            x (ndarray): observational data of shape ``[n_observations, n_vars]``
            interv_targets (ndarray): indicator mask of interventions of shape ``[n_observations, n_vars]``
            rng (ndarray): rng; skeleton for minibatching (TBD)

        Returns:
            log prob of shape ``[1,]``
        """
        log_prob_theta = self.log_prob_parameters(g=g, theta=theta)
        log_likelihood = self.log_likelihood_mask(g=g, theta=theta, x=x, interv_targets=interv_targets)
        return log_prob_theta + log_likelihood

