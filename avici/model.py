import inspect
import jax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import logsumexp

import haiku as hk
import functools

from avici.utils.data_jax import jax_get_train_x, jax_get_x


def layer_norm(*, axis, name=None):
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True, name=name)


def set_diagonal(arr, val):
    n_vars = arr.shape[-1]
    return arr.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].set(val)


class BaseModel(hk.Module):

    def __init__(self,
                 layers=8,
                 dim=128,
                 key_size=32,
                 num_heads=8,
                 widening_factor=4,
                 dropout=0.1,
                 out_dim=None,
                 logit_bias_init=-3.0,
                 cosine_temp_init=0.0,
                 ln_axis=-1,
                 name="BaseModel",
                 ):
        super().__init__(name=name)
        self.dim = dim
        self.out_dim = out_dim or dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.ln_axis = ln_axis
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.logit_bias_init = logit_bias_init
        self.cosine_temp_init = cosine_temp_init
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")  # kaiming uniform


    def __call__(self, x, is_training: bool):
        dropout_rate = self.dropout if is_training else 0.0
        z = hk.Linear(self.dim)(x)

        for _ in range(self.layers):
            # mha
            q_in = layer_norm(axis=self.ln_axis)(z)
            k_in = layer_norm(axis=self.ln_axis)(z)
            v_in = layer_norm(axis=self.ln_axis)(z)
            z_attn = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init_scale=2.0,
                model_size=self.dim,
            )(q_in, k_in, v_in)
            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_attn)

            # ffn
            z_in = layer_norm(axis=self.ln_axis)(z)
            z_ffn = hk.Sequential([
                hk.Linear(self.widening_factor * self.dim, w_init=self.w_init),
                jax.nn.relu,
                hk.Linear(self.dim, w_init=self.w_init),
            ])(z_in)
            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_ffn)

            # flip N and d axes
            z = jnp.swapaxes(z, -3, -2)

        z = layer_norm(axis=self.ln_axis)(z)
        assert z.shape[-2] == x.shape[-2] and z.shape[-3] == x.shape[-3], "Do we have an odd number of layers?"

        # [..., n_vars, dim]
        z = jnp.max(z, axis=-3)

        # u, v dibs embeddings for edge probabilities
        u = hk.Sequential([
            layer_norm(axis=self.ln_axis),
            hk.Linear(self.out_dim, w_init=self.w_init),
        ])(z)
        v = hk.Sequential([
            layer_norm(axis=self.ln_axis),
            hk.Linear(self.out_dim, w_init=self.w_init),
        ])(z)

        # edge logits
        # [..., n_vars, dim], [..., n_vars, dim] -> [..., n_vars, n_vars]
        u /= jnp.linalg.norm(u, axis=-1, ord=2, keepdims=True)
        v /= jnp.linalg.norm(v, axis=-1, ord=2, keepdims=True)
        logit_ij = jnp.einsum("...id,...jd->...ij", u, v)
        temp = hk.get_parameter("learned_temp", (1, 1, 1), logit_ij.dtype,
                                init=hk.initializers.Constant(self.cosine_temp_init)).squeeze()
        logit_ij *= jnp.exp(temp)
        logit_ij_bias = hk.get_parameter("final_matrix_bias", (1, 1, 1), logit_ij.dtype,
                                         init=hk.initializers.Constant(self.logit_bias_init)).squeeze()
        logit_ij += logit_ij_bias

        assert logit_ij.shape[-1] == x.shape[-2] and logit_ij.shape[-2] == x.shape[-2]
        return logit_ij



class InferenceModel:

    def __init__(self, *,
                 model_class,
                 model_kwargs,
                 train_p_obs_only=0.0,
                 acyclicity=None,
                 acyclicity_pow_iters=10,
                 mask_diag=True,
                 ):

        self._train_p_obs_only = jnp.array(train_p_obs_only)
        self._acyclicity_weight = acyclicity
        self._acyclicity_power_iters = acyclicity_pow_iters
        self.mask_diag = mask_diag

        # filter deprecated network kwargs
        sig = inspect.signature(model_class.__init__).parameters
        deprec = list(filter(lambda key: key not in sig, model_kwargs.keys()))
        for k in deprec:
            del model_kwargs[k]
            # print(f"Ignoring deprecated kwarg `{k}` loaded from `model_kwargs` in checkpoint")

        # init forward pass transform
        self.net = hk.transform(lambda *args: model_class(**model_kwargs)(*args))


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
        # [..., d, d]
        is_training = False
        logits = self.net.apply(params, rng, x, is_training)
        prob1 = jax.nn.sigmoid(logits)

        # sample graphs
        # [..., n_samples, d, d]
        key, subk = random.split(rng)
        samples = jnp.moveaxis(random.bernoulli(subk, prob1, shape=(n_samples,) + prob1.shape), 0, -3).astype(jnp.int32)
        if self.mask_diag:
            samples = set_diagonal(samples, 0.0)

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
        # [..., d, d]
        logits = self.net.apply(params, rng, x, is_training)
        logp_edges = jax.nn.log_sigmoid(logits)
        if self.mask_diag:
            logp_edges = set_diagonal(logp_edges, -jnp.inf)

        return logp_edges


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


    def exp_matmul(self, _logmat, _vec, _axis):
        """
        Matrix-vector multiplication with matrix in log-space
        """
        if _axis == -1:
            _ax_unsqueeze, _ax_sum = -2, -1
        elif _axis == -2:
            _ax_unsqueeze, _ax_sum = -1, -2
        else:
            raise ValueError(f"invalid axis inside exp_matmul")

        _logvec, _logvec_sgn = logsumexp(_logmat, b=jnp.expand_dims(_vec, axis=_ax_unsqueeze),
                                         axis=_ax_sum, return_sign=True)
        return _logvec_sgn * jnp.exp(_logvec)


    def acyclicity_spectral_log(self, logmat, key, power_iterations):
        """
        No Bears acyclicity constraint by
        https://psb.stanford.edu/psb-online/proceedings/psb20/Lee.pdf

        Performed in log-space
        """

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
    def loss(self, params, dual, key, data, t, is_training: bool):
        # `data` leaves have leading dimension [1, batch_size_device, ...]

        key, subk = random.split(key)
        if is_training:
            x = jax_get_train_x(subk, data, p_obs_only=self._train_p_obs_only)
        else:
            x = jax_get_x(data)

        n_vars = data["g"].shape[-1]

        ### inference model q(G | D)
        # [..., n_observations, d, 2] --> [..., d, d]
        key, subk = random.split(key)
        logits = self.net.apply(params, subk, x, is_training)

        # get logits [..., d, d]
        logp1 = jax.nn.log_sigmoid(  logits)
        logp0 = jax.nn.log_sigmoid(- logits)

        # labels [..., d, d]
        y_soft = data["g"]

        # mean over edges and skip diagonal (no self-loops)
        # [...]
        loss_eltwise = - (y_soft * logp1 + (1 - y_soft) * logp0)
        if self.mask_diag:
            batch_loss = set_diagonal(loss_eltwise, 0.0).sum((-1, -2)) / (n_vars * (n_vars - 1))
        else:
            batch_loss = loss_eltwise.sum((-1, -2)) / (n_vars * n_vars)

        # [] scalar
        loss_raw = batch_loss.mean() # mean over all available batch dims

        ### acyclicity
        key, subk = random.split(key)
        if self._acyclicity_weight is not None:
            # [..., d, d]
            if self.mask_diag:
                logp_edges = set_diagonal(logp1, -jnp.inf)
            else:
                logp_edges = logp1

            # [...]
            spectral_radii = self.acyclicity_spectral_log(logp_edges, subk, power_iterations=self._acyclicity_power_iters)

            # [] scalars
            ave_acyc_penalty = spectral_radii.mean()
            wgt_acyc_penalty = self._acyclicity_weight(ave_acyc_penalty, t, dual)

        else:
            # [] scalars
            ave_acyc_penalty = jnp.array(0.0)
            wgt_acyc_penalty = jnp.array(0.0)

        # [] scalar
        loss = loss_raw + wgt_acyc_penalty
        aux = {
            "loss_raw": loss_raw,
            "acyc": ave_acyc_penalty,
            "wgt_acyc": wgt_acyc_penalty,
            "mean_z_norm": jnp.abs(logits).mean(),
        }
        return loss, aux
