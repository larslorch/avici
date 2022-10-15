import functools

import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_map, tree_reduce


def expand_by(arr, n):
    """
    Expands jnp.array by n dimensions at the end
    Args:
        arr: shape [...]
        n (int)
    
    Returns:
        arr of shape [..., 1, ..., 1] with `n` ones
    """
    return jnp.expand_dims(arr, axis=tuple(arr.ndim + j for j in range(n)))


@jit
def sel(mat, mask):
    """
    jit/vmap helper function

    Args:
        mat: [N, d]
        mask: [d, ]  boolean 

    Returns:
        [N, d] with columns of `mat` with `mask` == 1 non-zero a
        and the columns with `mask` == 0 are zero

    Example: 
        mat 
        1 2 3
        4 5 6
        7 8 9

        mask
        1 0 1

        out
        1 0 3
        4 0 6
        7 0 9
    """
    return jnp.where(mask, mat, 0)

@jit
def leftsel(mat, mask, maskval=0.0):
    """
    jit/vmap helper function

    Args:
        mat: [N, d]
        mask: [d, ]  boolean 

    Returns:
        [N, d] [N, d] with columns of `mat` with `mask` == 1 non-zero a
        and pushed leftmost; the columns with `mask` == 0 are zero

    Example: 
        mat 
        1 2 3
        4 5 6
        7 8 9

        mask
        1 0 1

        out
        1 3 0
        4 6 0
        7 9 0
    """
    valid_indices = jnp.where(
        mask, jnp.arange(mask.shape[0]), mask.shape[0])
    padded_mat = jnp.concatenate(
        [mat, maskval * jnp.ones((mat.shape[0], 1))], axis=1)
    padded_valid_mat = padded_mat[:, jnp.sort(valid_indices)]
    return padded_valid_mat


@functools.partial(jit, static_argnums=(1,))
def mask_topk(x, topkk):
    """
    Returns indices of `topk` entries of `x` in decreasing order

    Args:
        x: [N, ]
        topk (int)

    Returns:
        array of shape [topk, ]
        
    """
    mask = x.argsort()[-topkk:][::-1]
    return mask


def squared_norm_pytree(x, y):
    """Computes squared euclidean norm between two pytrees

    Args:
        x:  PyTree
        y:  PyTree

    Returns:
        shape []
    """

    diff = tree_map(jnp.subtract, x, y)
    squared_norm_ind = tree_map(lambda leaf: jnp.square(leaf).sum(), diff)
    squared_norm = tree_reduce(jnp.add, squared_norm_ind)
    return squared_norm


