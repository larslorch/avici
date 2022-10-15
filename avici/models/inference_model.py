import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
from jax.scipy.special import logsumexp

import haiku as hk
import functools

from avici.modules.architectures_refactored import BaseModel

from avici.synthetic.utils_jax import jax_get_train_x, jax_get_x

class InferenceModelG:
    """
    Model backbone for amortized BN inference
    Function shape:
        [N, d] -> [d, d]

    """

    def __init__(self, *, model_kwargs,
                 bernoulli="sigmoid",
                 loss="nll", label_smoothing=0.0,
                 pos_weight=1.0,
                 train_p_obs_only=0.0,
                 model_class='naive',
                 acyclicity=None,
                 acyclicity_pow_iters=5,
                 mixture_net=False,
                 mask_diag=True,
                 mixture_k=8,
                 log_likelihood=None,
                 kl_mixt_pen=False,
                 kl_mixt_wgt=0.0,
                 standardize_v=None, # deprecated; kept for legacy
                 ):

        self.bernoulli = bernoulli
        self._loss_type = loss
        self._label_smoothing = label_smoothing
        self._pos_weight = pos_weight
        self._train_p_obs_only = jnp.array(train_p_obs_only)
        self._acyclicity_weight = acyclicity
        self._acyclicity_power_iters = acyclicity_pow_iters

        self.model_class = model_class
        self.mixture_net = mixture_net
        self.mixture_k = mixture_k
        self.kl_mixt_pen = kl_mixt_pen
        self.kl_mixt_wgt = kl_mixt_wgt

        # target log log_likelihood: log p(D | G, theta)
        self.log_likelihood = log_likelihood

        # forward pass transforms
        # (since we don't use BatchNorm, we don't use `with_state`)
        if model_class == 'BaseModel':
            # keep rng because we have dropout
            self.net = hk.transform(lambda *args: BaseModel(**model_kwargs)(*args))
            self.mixture_net = True

        else:
            raise KeyError(f"False `model_class` option {model_class}")

        self.mask_diag = mask_diag


    @functools.partial(jax.jit, static_argnums=(0, 1))
    def sample_graphs(self, n_samples, params, rng, x):
        """
        Args:
            n_samples: number of samples
            params: hk.Params
            rng
            x: [..., N, d, 2]
            is_count_data [...] bool

        Returns:
            graph samples of shape [..., n_samples, d, d]
        """
        n_vars = x.shape[-2]
        is_training = False

        if self.mixture_net:
            # [..., mixture_k, d, d], [..., mixture_k]
            logit_ij, logit_mixtures = self.net.apply(params, rng, x, is_training)
            prob1 = jax.nn.sigmoid(logit_ij)

            # sample mixture membership
            # [n_samples, ...]
            key, subk = random.split(rng)
            assmt = random.categorical(subk, logit_mixtures, axis=-1, shape=(n_samples,) + logit_mixtures.shape[:-1])

            # [..., n_samples, mixture_k] with mixture_k dimension being one-hot
            onehot_assmt = jnp.eye(logit_mixtures.shape[-1])[jnp.moveaxis(assmt, 0, -1)]

            # select adjacency matrices of probabilities of sampled mixture components using mask
            # [..., mixture_k, d, d], [..., n_samples, mixture_k]  -> [... n_samples, d, d]
            sel_prob1 = jnp.einsum('...kab,...nk->...nab', prob1, onehot_assmt)

            # sample graphs
            # [..., n_samples, d, d]
            key, subk = random.split(rng)
            samples = random.bernoulli(subk, sel_prob1).astype(jnp.int32)

        else:
            # [..., d, d]
            scores = self.net.apply(params, rng, x, is_training)
            prob1 = jax.nn.sigmoid(scores)

            # sample graphs
            # [..., n_samples, d, d]
            key, subk = random.split(rng)
            samples = jnp.moveaxis(random.bernoulli(subk, prob1, shape=(n_samples,) + prob1.shape),
                                   0, -3).astype(jnp.int32)

        if self.mask_diag:
            samples = samples.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].set(0)
        return samples


    @functools.partial(jax.jit, static_argnums=(0, 4))
    def infer_edge_logprobs(self, params, rng, x, is_training: bool):
        """
        Args:
            params: hk.Params
            rng
            x: [..., N, d, 2]
            is_training
            is_count_data [...] bool

        Returns:
            logprobs of graph adjacency matrix prediction of shape [..., d, d]
        """
        n_vars = x.shape[-2]

        if self.mixture_net:
            # [..., mixture_k, d, d], [..., mixture_k]
            logit_ij, logit_mixtures = self.net.apply(params, rng, x, is_training)
            logp1 = jax.nn.log_sigmoid(logit_ij)
            logmixt = jax.nn.log_softmax(logit_mixtures, axis=-1)

            # [..., d, d]
            logp_edges = logsumexp(jnp.expand_dims(logmixt, axis=(-2, -1)) + logp1, axis=-3)

        else:
            # [..., d, d]
            scores = self.net.apply(params, rng, x, is_training)
            logp_edges = jax.nn.log_sigmoid(scores)

        if self.mask_diag:
            logp_edges = logp_edges.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].set(-jnp.inf)
        return logp_edges


    @functools.partial(jax.jit, static_argnums=(0, 4))
    def infer_edge_pair_logprobs(self, params, rng, x, is_training: bool):
        """
        Args:
            params: hk.Params
            rng
            x: [..., N, d, 2]
            is_training
            is_count_data [...] bool

        Returns:
            logprobs of edge pairs of shape [..., d, d, d] representing log p(i->j, j->k)
        """
        n_vars = x.shape[-2]

        if self.mixture_net:
            # [..., mixture_k, d, d], [..., mixture_k]
            logit_ij, logit_mixtures = self.net.apply(params, rng, x, is_training)
            logp1 = jax.nn.log_sigmoid(logit_ij)
            logmixt = jax.nn.log_softmax(logit_mixtures, axis=-1)

            # log p(i->j, j->k | mixture)
            # this is the equivalent of einsum "ij,jk->ijk" in log space
            # [..., mixture_k, d, d, d]
            logp_edge_pair_mixture = logp1[..., :, :, None] + logp1[..., None, :, :]

            # [..., d, d, d]
            logp_edge_pairs = logsumexp(jnp.expand_dims(logmixt, axis=(-3, -2, -1)) + logp_edge_pair_mixture, axis=-4)

        else:
            # [..., d, d]
            scores = self.net.apply(params, rng, x, is_training)
            logp1 = jax.nn.log_sigmoid(scores)

            # log p(i->j, j->k)
            # this is the equivalent of einsum "ij,jk->ijk" in log space
            # [..., d, d, d]
            logp_edge_pairs = logp1[..., :, :, None] + logp1[..., None, :, :]

        if self.mask_diag:
            # mask self-loops
            logp_edge_pairs = logp_edge_pairs.at[..., jnp.arange(n_vars), jnp.arange(n_vars), :].set(-jnp.inf)
            logp_edge_pairs = logp_edge_pairs.at[..., :, jnp.arange(n_vars), jnp.arange(n_vars)].set(-jnp.inf)

            # mask 2-loops
            logp_edge_pairs = logp_edge_pairs.at[..., jnp.arange(n_vars), :, jnp.arange(n_vars)].set(-jnp.inf)

        return logp_edge_pairs


    @functools.partial(jax.jit, static_argnums=(0, 4))
    def infer_mixture_logprobs(self, params, rng, x, is_training: bool):
        """
        For test time inference

        Args:
           params: hk.Params
            rng
            x: [..., N, d, 2]
            is_training
            is_count_data [...] bool

        Returns:
            full specification of mixture distribution
                [..., mixture_k, d, d], [..., mixture_k]
        """
        n_vars = x.shape[-2]

        if self.mixture_net:
            # [..., mixture_k, d, d], [..., mixture_k]
            logit_ij, logit_mixtures = self.net.apply(params, rng, x, is_training)
            logprob1 = jax.nn.log_sigmoid(logit_ij)
            logprobmixt = jax.nn.log_softmax(logit_mixtures, axis=-1)

        else:
            # [..., d, d]
            scores = self.net.apply(params, rng, x, is_training)
            logprob1 = jax.nn.log_sigmoid(scores)
            logprobmixt = jnp.zeros(scores.shape[:-2])

        if self.mask_diag:
            logprob1 = logprob1.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].set(-jnp.inf)
        return logprob1, logprobmixt

    def infer_edge_probs(self, params, x):
        """
        For test time inference

        Args:
            params: hk.Params
            x: [..., N, d, 1]
            is_count_data [...] bool

        Returns:
            probabilities of graph adjacency matrix prediction of shape [..., d, d]
        """
        is_training_, dummy_rng_ = False, random.PRNGKey(0) # assume test time
        logp_edges = self.infer_edge_logprobs(params, dummy_rng_, x, is_training_)
        p_edges = jnp.exp(logp_edges)
        return p_edges

    def infer_edge_pair_probs(self, params, x):
        """
        For test time inference

        Args:
            params: hk.Params
            x: [..., N, d, 1]
            is_count_data [...] bool

        Returns:
            probabilities of edge pairs of shape [..., d, d, d] representing p(i->j, j->k)
        """
        is_training_, dummy_rng_ = False, random.PRNGKey(0) # assume test time
        logp_edge_pairs = self.infer_edge_pair_logprobs(params, dummy_rng_, x, is_training_)
        p_edge_pairs = jnp.exp(logp_edge_pairs)
        return p_edge_pairs


    def infer_mixture_probs(self, params, x):
        """
        For test time inference

        Args:
            params: hk.Params
            x: [..., N, d, 1]
            is_count_data [...] bool

        Returns:
            full specification of mixture distribution
                [..., mixture_k, d, d], [..., mixture_k]
        """
        is_training_, dummy_rng_ = False, random.PRNGKey(0)  # assume test time
        logprob1, logprobmixt = self.infer_mixture_logprobs(params, dummy_rng_, x, is_training_)
        p_edges = jnp.exp(logprob1)
        p_mixt = jnp.exp(logprobmixt)
        return p_edges, p_mixt


    def exp_matmul(self, _logmat, _vec, _axis):
        if _axis == -1:
            _ax_unsqueeze, _ax_sum = -2, -1
        elif _axis == -2:
            _ax_unsqueeze, _ax_sum = -1, -2
        else:
            raise ValueError(f"invalid axis inside exp_matmul")

        _logvec, _logvec_sgn = logsumexp(_logmat, b=jnp.expand_dims(_vec, axis=_ax_unsqueeze),
                                         axis=_ax_sum, return_sign=True)
        return _logvec_sgn * jnp.exp(_logvec)


    def acyclicity_spectral(self, logmat, key, power_iterations):
        """No Bears acyclicity constraint by https://psb.stanford.edu/psb-online/proceedings/psb20/Lee.pdf """

        # left/right power iteration
        key, subk = random.split(key)
        u = random.normal(subk, shape=logmat.shape[:-1])
        key, subk = random.split(key)
        v = random.normal(subk, shape=logmat.shape[:-1])

        for t in range(power_iterations):

            # u_new = jnp.einsum('...i,...ij->...j', u, mat)
            u_new = self.exp_matmul(logmat, u, -2) # u @ exp(mat)

            # v_new = jnp.einsum('...ij,...j->...i', mat, v)
            v_new = self.exp_matmul(logmat, v, -1) # exp(mat) @ v

            u = u_new / jnp.linalg.norm(u_new, ord=2, axis=-1, keepdims=True)
            v = v_new / jnp.linalg.norm(v_new, ord=2, axis=-1, keepdims=True)

        u = jax.lax.stop_gradient(u)
        v = jax.lax.stop_gradient(v)

        # largest_eigenvalue = (u @ exp(mat) @ v) / u.dot(v)
        largest_eigenvalue = (
                jnp.einsum('...j,...j->...', u, self.exp_matmul(logmat, v, -1)) /
                jnp.einsum('...j,...j->...', u, v)
        )

        return largest_eigenvalue


    """Training"""
    def cross_entropy(self, params, dual, key, data, t, is_training: bool):
        """Cross entropy loss for pairwise edge prediction (i.e. classification)"""

        key, subk = random.split(key)
        if is_training:
            x = jax_get_train_x(subk, data, p_obs_only=self._train_p_obs_only)
        else:
            x = jax_get_x(data)

        n_vars = data["g"].shape[-1]

        # inference model q(G | D)
        if self.mixture_net:

            # [..., n_observations, d, 1] --> [...,  mixture_k, d, d], [...,  mixture_k]
            # rng is needed in `apply` because of dropout
            key, subk = random.split(key)
            logit_ij, logit_mixtures = self.net.apply(params, subk, x, is_training)

            # get logprobs
            # [..., mixture_k, d, d]
            logp1 = jax.nn.log_sigmoid(  logit_ij)
            logp0 = jax.nn.log_sigmoid(- logit_ij)
            # [..., mixture_k]
            logmixt = jax.nn.log_softmax(logit_mixtures, axis=-1)

            # labels [..., 1, d, d]
            y_soft = (1 - self._label_smoothing) * jnp.expand_dims(data["g"], axis=-3) + self._label_smoothing / 2.0

            # binary cross entropy, summed over edges and skipping diagonal (no self-loops)
            # [..., mixture_k, d, d]
            xent_mixture_eltwise = - (self._pos_weight * y_soft * logp1 + (1 - y_soft) * logp0)

            # [..., mixture_k]
            if self.mask_diag:
                xent_mixture = xent_mixture_eltwise.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].multiply(0.0).sum((-1, -2))
            else:
                xent_mixture = xent_mixture_eltwise.sum((-1, -2))

            # [...]
            # scale down for approx. equal loss scale across n_vars (has to be done *after* logsumexp)
            if self.mask_diag:
                batch_xent = logsumexp(logmixt + xent_mixture, axis=-1) / (n_vars * (n_vars - 1))
            else:
                batch_xent = logsumexp(logmixt + xent_mixture, axis=-1) / (n_vars * n_vars)

            # []
            loss_raw = batch_xent.mean()  # mean over all available batch dims

        else:
            # [..., n_observations, d, 1] --> [..., d, d]
            # rng is needed in `apply` because of dropout
            key, subk = random.split(key)
            logit_ij = self.net.apply(params, subk, x, is_training)

            # get logits  [d, d]
            logp1 = jax.nn.log_sigmoid(  logit_ij)
            logp0 = jax.nn.log_sigmoid(- logit_ij)

            # labels
            y_soft = (1 - self._label_smoothing) * data["g"] + self._label_smoothing/2.0

            # binary cross entropy, suumed over edges and skipping diagonal (no self-loops)
            # [..., d, d]
            xent_eltwise = - (self._pos_weight * y_soft * logp1 + (1 - y_soft) * logp0)

            # [...] where `...` is usually batch size
            # scale down for approx. equal loss scale across n_vars
            if self.mask_diag:
                batch_xent = xent_eltwise.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].multiply(0.0).sum((-1, -2)) / (n_vars * (n_vars - 1))
            else:
                batch_xent = xent_eltwise.sum((-1, -2)) / (n_vars * n_vars)

            # [] scalar
            loss_raw = batch_xent.mean() # mean over all available batch dims

        ### regularization
        if self.kl_mixt_pen:

            # mean pairwise KL divergence between mixture components
            # [..., n_mixtures, 1, d, d]
            logp0_ax0 = jnp.expand_dims(logp0, axis=-3)
            logp1_ax0 = jnp.expand_dims(logp1, axis=-3)

            # [..., 1, n_mixtures, d, d]
            logp0_ax1 = jnp.expand_dims(logp0, axis=-4)
            logp1_ax1 = jnp.expand_dims(logp1, axis=-4)

            # [..., n_mixtures, n_mixtures, d, d]
            # KL between Bern(p) and Bern(q) = p(log(p) - log(q)) + (1-p)(log(1-p) - log(1-q))
            pairwise_kl = jnp.exp(logp0_ax0) * (logp0_ax0 - logp0_ax1) + jnp.exp(logp1_ax0) * (logp1_ax0 - logp1_ax1)

            # mean over edges and over components
            # [...]
            pairwise_mean_kl = pairwise_kl.mean((-4, -3, -2, -1))

            # R(q) = - KL because we want to encourage/maximize KL between components
            regularizer = - self.kl_mixt_wgt * pairwise_mean_kl.mean()
        else:
            regularizer = jnp.array(0.0)

        ### acyclicity
        key, subk = random.split(key)
        if self._acyclicity_weight is not None:

            # if mixture_net:   [..., mixture_k, d, d]
            # else:             [..., d, d]
            if self.mask_diag:
                logp_edges = logp1.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].set(-jnp.inf)
            else:
                logp_edges = logp1


            # if mixture_net:   [..., mixture_k]
            # else:             [...]
            spectral_radii = self.acyclicity_spectral(logp_edges, subk, power_iterations=self._acyclicity_power_iters)

            # [] scalars
            ave_acyc_penalty = spectral_radii.mean()
            wgt_acyc_penalty = self._acyclicity_weight(ave_acyc_penalty, t, dual)

        else:
            # [] scalars
            ave_acyc_penalty = jnp.array(0.0)
            wgt_acyc_penalty = jnp.array(0.0)

        # [] scalar
        loss = loss_raw + regularizer + wgt_acyc_penalty

        aux = {
            "loss_raw": loss_raw,
            "reg": regularizer,
            "acyc": ave_acyc_penalty,
            "wgt_acyc": wgt_acyc_penalty,
            "mean_z_norm": jnp.abs(logit_ij).mean(),
        }
        return loss, aux


    """Loss function"""
    def loss(self, params, dual, key, batch, t, is_training: bool):
        # batch leaves have leading dimension [1, batch_size_device, ...]
        if self._loss_type == "xent":
            return self.cross_entropy(params, dual, key, batch, t, is_training)
        else:
            raise KeyError(f"Invalid loss `{self._loss_type}`")

