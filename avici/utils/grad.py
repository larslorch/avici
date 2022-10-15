import jax
import haiku as hk

from optax._src import base


def scale_batch_grads(*, n_vars, n_observations,
                      layer_id="RelateAggSplitModel/~/local_embed"):

    """Rescales gradients of
        node embeddings (applied to all node rows, batched over observations)
           by 1/observations
        observations embeddings (applied to all observation columns, batched over nodes)
           by 1/nodes

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return base.EmptyState()

    def update_fn(updates, state, params=None):
        del params

        # create pytree with same structure as updates containing correct scaling at each leaf
        scales = {}
        for layer_key, sub_updates in updates.items():
            scales_layer = {}
            for k in sub_updates.keys():
                if layer_id in layer_key and "mha_obs" in layer_key:
                    scales_layer[k] = 1 / n_vars
                elif layer_id in layer_key and "mha_node" in layer_key:
                    scales_layer[k] = 1 / n_observations
                else:
                    scales_layer[k] = 1.0

            scales[layer_key] = hk.data_structures.to_immutable_dict(scales_layer)
        scales = hk.data_structures.to_immutable_dict(scales)

        # apply scaling
        updates = jax.tree_map(lambda g, s: g * s, updates, scales)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
