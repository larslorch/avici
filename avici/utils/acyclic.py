import functools
import jax.numpy as jnp
from jax import jit, vmap

@functools.partial(jit, static_argnums=(1,))
def acyclic_constr(mat, n_vars):
    """
        Differentiable acyclicity constraint from
        Yu et al 2019
        http://proceedings.mlr.press/v97/yu19a/yu19a.pdf

        mat:  [n_vars, n_vars]
        out:  [1,] constraint value

    """

    alpha = 1.0 / n_vars
    # M = jnp.eye(n_vars) + alpha * mat * mat # [original version]
    M = jnp.eye(n_vars) + alpha * mat

    M_mult = jnp.linalg.matrix_power(M, n_vars)
    h = jnp.trace(M_mult) - n_vars
    return h

eltwise_acyclic_constr = jit(vmap(acyclic_constr, (0, None), 0), static_argnums=(1,))
double_eltwise_acyclic_constr = jit(vmap(eltwise_acyclic_constr, (0, None), 0), static_argnums=(1,))