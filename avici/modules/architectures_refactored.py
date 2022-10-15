import math
import jax
import jax.numpy as jnp
import haiku as hk

def layer_norm(*, axis, name=None):
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True, name=name)


class BaseModel(hk.Module):

    def __init__(self,
                 n_mixtures=1,
                 dim=128,
                 layers=8,
                 dropout=0.1,
                 ln_axis=-1,
                 widening_factor=4,
                 num_heads=8,
                 key_size=32,
                 logit_bias_init=-3.0,
                 out_dim=None,
                 mixture_drop=None,
                 cosine_sim=False,
                 relational_encoder=False,
                 attn_only_n=False,
                 attn_only_d=False,
                 cosine_temp_init=0.0,
                 name="BaseModel",
                 ):
        super().__init__(name=name)
        self.n_mixtures = n_mixtures
        self.dim = dim
        self.out_dim = out_dim or dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.mixture_drop = mixture_drop or dropout
        self.ln_axis = ln_axis
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.logit_bias_init = logit_bias_init
        self.cosine_temp_init = cosine_temp_init
        self.cosine_sim = cosine_sim
        self.relational_encoder = relational_encoder
        self.attn_only_n = attn_only_n
        self.attn_only_d = attn_only_d
        assert not (attn_only_n and attn_only_d)
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")  # kaiming uniform

    def __call__(self, x, is_training: bool):
        dropout_rate = self.dropout if is_training else 0.0
        z = hk.Linear(self.dim)(x)

        if self.attn_only_n:
            z = jnp.swapaxes(z, -3, -2)

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
            if not self.attn_only_n and not self.attn_only_d:
                z = jnp.swapaxes(z, -3, -2)

        if self.attn_only_n:
            z = jnp.swapaxes(z, -3, -2)

        z = layer_norm(axis=self.ln_axis)(z)
        assert z.shape[-2] == x.shape[-2] and z.shape[-3] == x.shape[-3], "Do we have an odd number of layers?"

        # [..., n_vars, dim]
        z = jnp.max(z, axis=-3)

        # u, v dibs embeddings for edge probabilities
        u = hk.Sequential([
            layer_norm(axis=self.ln_axis),
            hk.Linear(self.n_mixtures * self.out_dim, w_init=self.w_init),
            hk.Reshape(output_shape=(self.n_mixtures, self.out_dim), preserve_dims=-1),
        ])(z)
        v = hk.Sequential([
            layer_norm(axis=self.ln_axis),
            hk.Linear(self.n_mixtures * self.out_dim, w_init=self.w_init),
            hk.Reshape(output_shape=(self.n_mixtures, self.out_dim), preserve_dims=-1),
        ])(z)

        # edge logits
        # [..., n_vars, n_mixtures, dim], [..., n_vars, n_mixtures, dim] -> [..., n_mixtures, n_vars, n_vars]
        if self.relational_encoder:
            # expand into [..., n_vars, 1, n_mixtures, dim], [..., 1, n_vars, n_mixtures, dim]
            # concat into [..., n_vars, n_vars, n_mixtures, dim]
            paired_z = jnp.concatenate(jnp.broadcast_arrays(jnp.expand_dims(u, axis=-3),
                                                            jnp.expand_dims(v, axis=-4)), axis=-1)
            # concat into [..., n_vars, n_vars, n_mixtures]
            logit_ij = hk.Sequential([
                hk.Linear(self.out_dim, w_init=self.w_init),
                jnp.tanh,
                hk.Linear(1, w_init=self.w_init, b_init=hk.initializers.Constant(self.logit_bias_init)),
            ], name="final_rel")(paired_z).squeeze(-1)

            # transpose into [..., n_mixtures, n_vars, n_vars]
            logit_ij = jnp.swapaxes(logit_ij, -2, -1)
            logit_ij = jnp.swapaxes(logit_ij, -3, -2)

        else:
            if self.cosine_sim:
                u /= jnp.linalg.norm(u, axis=-1, ord=2, keepdims=True)
                v /= jnp.linalg.norm(v, axis=-1, ord=2, keepdims=True)
                logit_ij = jnp.einsum("...ikd,...jkd->...kij", u, v)
                temp = hk.get_parameter("learned_temp", (self.n_mixtures, 1, 1),
                    logit_ij.dtype, init=hk.initializers.Constant(self.cosine_temp_init))
                logit_ij *= jnp.exp(temp)
            else:
                logit_ij = jnp.einsum("...ikd,...jkd->...kij", u, v) / math.sqrt(self.out_dim)
            logit_ij_bias = hk.get_parameter("final_matrix_bias", (self.n_mixtures, 1, 1),
                                             logit_ij.dtype, init=hk.initializers.Constant(self.logit_bias_init))
            logit_ij += logit_ij_bias

        assert logit_ij.shape[-1] == x.shape[-2] and logit_ij.shape[-2] == x.shape[-2] \
               and logit_ij.shape[-3] == self.n_mixtures

        # uniform mixture weights
        # [..., n_mixtures]
        logit_mixtures = jnp.zeros(logit_ij.shape[:-2])

        # dropout of mixture components (no re-scaling as in classic dropout because we softmax anyway)
        # keep at least one mixture component
        p_drop = self.mixture_drop if is_training else 0.0
        keep = jax.random.bernoulli(hk.next_rng_key(), jnp.array(1.0 - p_drop), shape=logit_mixtures.shape)
        keep_certain = jax.random.choice(hk.next_rng_key(), self.n_mixtures, shape=logit_mixtures.shape[:-1])
        keep = keep | jnp.eye(self.n_mixtures, dtype=jnp.int32)[keep_certain]
        logit_mixtures = jnp.where(keep, logit_mixtures, -jnp.inf)  # -inf only possible bc we keep at least 1 component
        if self.n_mixtures == 1:
            logit_mixtures = jnp.zeros_like(logit_mixtures)

        return logit_ij, logit_mixtures