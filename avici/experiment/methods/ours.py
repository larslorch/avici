import jax
import jax.random as random
import jax.numpy as jnp
import numpy as onp
import haiku as hk

from avici.models.inference_model import InferenceModelG
from avici.experiment.utils import load_checkpoint
from avici.utils.eval import eval_batch_helper


def run_ours(seed, data, checkpoint_dir):
    """Runs our method from a loaded checkpoint"""

    # load checkpoint
    state, loaded_config = load_checkpoint(checkpoint_dir)

    # init model
    neural_net_kwargs = loaded_config["neural_net_kwargs"]
    inference_model_kwargs = loaded_config["inference_model_kwargs"]
    model = InferenceModelG(**inference_model_kwargs, **neural_net_kwargs)

    # generate predictions
    pred = eval_batch_helper(model, state.params, state.dual, data, random.PRNGKey(seed), 0)

    # count params
    num_params = hk.data_structures.tree_size(state.params)
    byte_size = hk.data_structures.tree_bytes(state.params)

    return {
        "num_params": num_params,
        "byte_size_f32": byte_size,
        "mbyte_size_f32": byte_size / 1e6,
        "neural_net_kwargs": neural_net_kwargs,
        "inference_model_kwargs": inference_model_kwargs,
        f"g_edges_prob": onp.array(jax.device_get(pred["g_edges_prob"]), dtype=onp.float32),
        f"g_edges": onp.array(jax.device_get(pred["g_edges"]["0_5"]), dtype=onp.int32),
    }

