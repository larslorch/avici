import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_reduce
from jax import random

def tree_index(pytree, idx):
    """
    Indexes pytree leaves and returns resulting pytree
    """
    return tree_map(lambda leaf: leaf[idx], pytree)


def tree_select(pytree, bool_mask):
    """
    Indexes pytree leading dimension with boolean mask
    """
    return tree_map(lambda leaf: leaf[bool_mask, ...], pytree)


def tree_zip_leading(pytree_list):
    """
    Converts n pytrees without leading dimension into one pytree with leading dim [n, ...]
    """
    return tree_map(lambda *args: jnp.stack([*args]) if len(args) > 1 else tree_expand_leading_by(*args, 1), *pytree_list)
    

def tree_unzip_leading(pytree, n):
    """
    Converts pytree with leading dim [n, ...] into n pytrees without the leading dimension
    """
    leaves, treedef = tree_flatten(pytree)
    
    return [
        tree_unflatten(treedef, [leaf[i] for leaf in leaves])
        for i in range(n)
    ]

def tree_expand_leading_by(pytree, n):
    """
    Converts pytree with leading pytrees with additional `n` leading dimensions
    """
    return tree_map(lambda leaf: jnp.expand_dims(leaf, axis=tuple(range(n))), pytree)


def tree_shapes(pytree):
    """
    Returns pytree with same tree but leaves replaced by original shapes
    """
    return tree_map(lambda leaf: jnp.array(leaf.shape), pytree)


def tree_key_split(key, pytree):
    """
    Generates one subkey from `key` for each leaf of `pytree` and returns it in tree of shape `pytree`
    """

    tree_flat, treedef = tree_flatten(pytree)
    subkeys_flat = random.split(key, len(tree_flat))
    subkeys_tree = tree_unflatten(treedef, subkeys_flat)
    return subkeys_tree


def tree_mul(pytree, c):
    """
    Multiplies every leaf of pytree with `c`
    """
    return tree_map(lambda leaf: leaf * c, pytree)

