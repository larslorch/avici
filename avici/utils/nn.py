import jax
import jax.numpy as jnp
import haiku as hk

def eltwise_softmax_cross_entropy(logits, labels):
    """
    Computes cross entropy loss for logits
    Arguments:
        logits:     [N, d]
        labels:     [N,]    integers in 0...d-1
    
    """
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)


def ridge_regularizer(params):
    """
    Computes squared norm of parameters for L2 regularization
    Arguments:
        params:     hk.Params
    
    """
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


class Identity(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        return x
