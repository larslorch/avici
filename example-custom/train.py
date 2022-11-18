import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf # avoids dataloader crash after execution finishes
tf.config.set_visible_devices([], 'GPU') # hide gpus to tf to avoid OOM and interference with jax

import warnings
warnings.filterwarnings("ignore", message="Some donated buffers were not usable") # long jax warning when not on gpu

import argparse
import jax
import jax.random as random
import jax.numpy as jnp
import haiku as hk
import psutil
import math
import time
import pprint
import optax
from collections import defaultdict
from pathlib import Path

from avici.definitions import CHECKPOINT_SUBDIR

from avici.model import BaseModel, InferenceModel
from avici.data import AsyncBufferDataset
from avici.buffer import FIFOBuffer
from avici.backbone import CheckpointingUpdater, SuperUpdater, get_first

from avici.train import evaluate, update_ave, retrieve_ave, print_header
from avici.utils.parse import load_data_config
from avici.utils.version_control import str2bool


DIR = Path(__file__).parent


def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help="Integer for random seeding")
    parser.add_argument("--config", type=Path, required=True, help="Absolute path to domain config `.yaml` file")
    parser.add_argument("--smoke_test", default=True, type=str2bool,
                        help="Flag for testing that sets network and training to minimal size")

    # data and buffer
    parser.add_argument("--train_n_obs", default=150, type=int,
                        help="Number of observational data points in training datasets")
    parser.add_argument("--train_n_int", default=50, type=int,
                        help="Number of interventional data points in training datasets")
    parser.add_argument("--p_obs_only", default=0.5, type=float,
                        help="Probability of only sampling `n_obs + n_int` observational data points in a given "
                             "training step")
    parser.add_argument("--buffer_size", default=200, type=int,
                        help="Number of training datasets buffered for each `n_vars`, where `n_vars` are the different "
                             "numbers of nodes in the trainining datasets as specified in the `config_domain`")
    parser.add_argument("--n_workers", type=int,
                        help="Number of parallel workers that generate fresh training datasets. Defaults to number of "
                             "available cpus")
    parser.add_argument("--n_listeners", default=8, type=int,
                        help="Number of processes that insert the produced datasets into buffers from queue in "
                             "producer-consumer workflow")

    # nn
    parser.add_argument("--dim", default=128, type=int, help="Feature size in transformer blocks")
    parser.add_argument("--out_dim", type=int,
                        help="Feature size used to compute final inner product predicting the edge probabilities. "
                             "Defaults to `dim`")
    parser.add_argument("--n_loc", default=8, type=int,
                        help="Number of alternating multi-head self attention layers, "
                             "where `k` implies `2k` layers with `k` attending over each of the two axes")
    parser.add_argument("--SAB_num_heads", default=8, type=int, help="Head count in multi-head attention")
    parser.add_argument("--key_size", default=32, type=int, help="Key, query, and value size in multi-head attention")
    parser.add_argument("--widening_factor", default=4, type=int,
                        help="Widening factor in hidden layer of intermediate feedforward layers")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--cosine_temp_init", default=2.0, type=float,
                        help="Log initialization value for learned temperature inside sigmoid")
    parser.add_argument("--mask_diag", default=True, type=str2bool,
                        help="Whether to mask diagonal (self-loops) in predicted probabilities")

    # optimizer
    parser.add_argument("--n_steps", default=300000, type=int, help="Number of primal update steps")
    parser.add_argument("--bsu", default=300, type=int,
                        help="Unit for estimating adequate batch sizes for the different training `n_vars`. "
                             "The batch size for a given `n_vars` is estimated roughly as `bsu` / `n_vars`, "
                             "assuming linear scaling of memory with the number of variables in a dataset")
    parser.add_argument("--lr", default=3e-5, type=float,
                        help="Base learning rate. Scaled by the square-root of the effective batch size "
                             "(maximum over all `n_vars`) as in You et al., 2019 for multi-device training")
    parser.add_argument("--grad_clip_value", default=1.0, type=float, help="Gradient clipping maximum value")

    # acyclicity
    parser.add_argument("--acyc", default=True, type=str2bool,
                        help="Whether to include the acyclicity constraint. "
                             "If not, the following kwargs starting with `acyc_` are ignored")
    parser.add_argument("--acyc_powit", default=10, type=int,
                        help="Number of power iterations in the acyclicity constraint")
    parser.add_argument("--acyc_dual_lr", default=1e-4, type=float, help="Dual learning rate \\eta")
    parser.add_argument("--acyc_inner_step", default=500, type=int, help="Primal steps performed between dual steps")
    parser.add_argument("--acyc_polyak", default=1e-4, type=float,
                        help="Polyak updating rate for acyclicity penalty used in primal step")
    parser.add_argument("--acyc_warmup", default=True, type=str2bool,
                        help="Whether to burnin primal updates before first dual step")
    parser.add_argument("--acyc_burnin", default=50000, type=int,
                        help="Number of primal burnin steps before first dual step")

    # logging
    parser.add_argument("--log_every", default=1000, type=int, help="Logging frequency in primal steps")
    parser.add_argument("--eval_every", default=10000, type=int, help="Evaluation frequency in primal steps")
    parser.add_argument("--checkpoint_every", default=10000, type=int, help="Checkpoint frequency in primal steps")
    parser.add_argument("--checkpoint", default=True, type=str2bool, help="Whether to save model checkpoints")
    parser.add_argument("--checkpoint_dir", type=Path, help="Checkpoint directory. Defaults to `./checkpoints`")
    parser.add_argument("--run_name", default="custom", help="Folder name for checkpoints inside `checkpoint_dir`")
    return parser



if __name__ == "__main__":

    jnp.set_printoptions(precision=4, suppress=True)
    kwargs = make_parser().parse_args()

    # smoke test arguments
    if kwargs.smoke_test:
        kwargs.dim, kwargs.n_loc, kwargs.SAB_num_heads, kwargs.key_size = 4, 1, 2, 2
        kwargs.n_steps, kwargs.log_every, kwargs.eval_every, kwargs.checkpoint_every = 10, 1, 10, 5
        kwargs.buffer_size, kwargs.train_n_obs, kwargs.train_n_int, kwargs.bsu = 10, 10, 5, 50

    # load data config with custom data-generating functions if specified
    config_domain = load_data_config(kwargs.config, abspath=True)

    # distributed training
    device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    try:
        cpu_count = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpu_count = psutil.cpu_count(logical=True)
    print(f"jax backend:   {jax.default_backend()} ")
    print(f"devices:       {device_count}")
    print(f"local_devices: {local_device_count}")

    # estimate maximum device batch size for datasets of the various numbers of variables `d` used for training
    # assumes linear scaling of memory as function of d, damping slightly for small nvars, skipping when batch size < 1
    def bs(d):
        return math.floor(0.1 + kwargs.bsu * (1 - math.exp(- 0.1 * d)) / d) # approx. = bsu / d

    batch_sizes = {d: {"device": [local_device_count, bs(d)],
                       "effective": device_count * bs(d)}
                   for d in config_domain["train_n_vars"] if not bs(d) < 1.0}
    batch_sizes_test = {d: {"device": [1]} for d in config_domain["test_n_vars"]}

    n_vars_choices = jnp.array(list(batch_sizes.keys()))
    n_vars_probs = n_vars_choices / jnp.array([batch_sizes[d]["effective"] for d in n_vars_choices.tolist()])
    n_vars_probs /= n_vars_probs.sum()

    # effective batch size is the batch size accross all hosts and devices.
    # in a multi-host training setup each host will only see a batch size of `effective batch size / host_count`.
    max_batch_size_effective = max(b["effective"] for _, b in batch_sizes.items())

    """ ========== Data ========== """
    print_header("Data")

    # init async dataset with `n_workers` filling several data buffers for the various `d`
    # launches async executors with `n_workers` filling a data buffer
    buffer_kwargs = dict(
        seed=kwargs.seed,
        config=config_domain,
        train_n_observations_obs=kwargs.train_n_obs,
        train_n_observations_int=kwargs.train_n_int,
        buffer_class=FIFOBuffer,
        buffer_size=kwargs.buffer_size,
        batch_dims_train=batch_sizes,
        batch_dims_test=batch_sizes_test,
        n_listeners=kwargs.n_listeners,
        n_workers=kwargs.n_workers,
    )
    ds = AsyncBufferDataset(**buffer_kwargs)
    print(f"buffer kwargs:\n{pprint.pformat({k: v for k, v in buffer_kwargs.items() if k != 'config'}, indent=4)}")

    train_iters = ds.make_datasets("train")
    print(f"Instantiated datasets", flush=True)

    # define reference data batch for visualization (without pmap dim, with batch dim)
    print(f"Instantiated reference batches", flush=True)
    print(f"train n_obs: {kwargs.train_n_obs}")
    print(f"      n_int: {kwargs.train_n_int}")
    print(f"           = {kwargs.train_n_obs + kwargs.train_n_int}")

    """ ========== Model, loss, and updater ========== """
    print_header("Model")

    print("Acyclicity: ")
    if kwargs.acyc:
        acyc_constr = lambda _penalty, _tt, _dual: _penalty * _dual
        print(f"\tdual: warmup = {kwargs.acyc_warmup}, mask_diag = {kwargs.mask_diag}")
    else:
        acyc_constr = None
        print(f"\tnone: mask_diag = {kwargs.mask_diag}")

    # net and loss
    neural_net_kwargs = dict(
        dim=kwargs.dim,
        out_dim=kwargs.out_dim,
        layers=kwargs.n_loc,
        key_size=kwargs.key_size,
        num_heads=kwargs.SAB_num_heads,
        widening_factor=kwargs.widening_factor,
        dropout=kwargs.dropout_rate,
        cosine_temp_init=kwargs.cosine_temp_init,
    )
    print(f"Net kwargs:\n{pprint.pformat(neural_net_kwargs, indent=4)}", flush=True)

    inference_model_kwargs = dict(
        train_p_obs_only=kwargs.p_obs_only,
        acyclicity=acyc_constr,
        acyclicity_pow_iters=kwargs.acyc_powit,
        mask_diag=kwargs.mask_diag,
    )
    print(f"Inference model kwargs:\n{pprint.pformat(inference_model_kwargs, indent=4)}", flush=True)

    model = InferenceModel(**inference_model_kwargs, model_class=BaseModel, model_kwargs=neural_net_kwargs)

    # optimizer
    print_header("Optimizer")
    learning_rate = math.sqrt(max_batch_size_effective) * kwargs.lr
    optimizer_modules = [
        optax.clip_by_global_norm(kwargs.grad_clip_value),
        optax.lamb(lambda _: jnp.array(learning_rate))
    ]
    optimizer = optax.chain(*optimizer_modules)

    print("Optimizer: LAMB")
    print(f"\tlearning rate = {learning_rate} = sqrt({max_batch_size_effective}) * {kwargs.lr} (sqrt. batch size scaling)")
    print(f"\tgrad_clip at {kwargs.grad_clip_value}")

    # updater
    updater_kwargs = dict(
        acyclicity_dual_lr=kwargs.acyc_dual_lr,
        acyclicity_inner_step=kwargs.acyc_inner_step,
        acyclicity_burnin=kwargs.acyc_burnin,
        acyclicity_warmup=kwargs.acyc_warmup,
        polyak_rate=kwargs.acyc_polyak,
        local_device_count=local_device_count
    )
    updater = SuperUpdater(net_init=model.net.init, loss_fn=model.loss, opt=optimizer, **updater_kwargs)

    if kwargs.checkpoint:
        CHECKPOINT_DIR = kwargs.checkpoint_dir or (DIR / CHECKPOINT_SUBDIR / kwargs.run_name)

        save_kwargs = dict(
            updater=updater_kwargs,
            neural_net_kwargs=neural_net_kwargs,
            inference_model_kwargs=inference_model_kwargs,
            buffer_kwargs=dict(buffer_kwargs, config=kwargs.config),
            train_script_kwargs=vars(kwargs),
        )
        updater = CheckpointingUpdater(inner=updater,
                                       checkpoint_dir=CHECKPOINT_DIR,
                                       checkpoint_every_n=kwargs.checkpoint_every,
                                       save_kwargs=save_kwargs)

    print(f"Updater instantiated", flush=True)
    print(f"{pprint.pformat(updater_kwargs, indent=4)}", flush=True)

    """ ========== Initialize state ========== """
    print_header("State")

    # broadcast same rng to every device
    key = random.PRNGKey(kwargs.seed)
    super_subk = jnp.broadcast_to(key, (local_device_count,) + key.shape)

    # data batch for initialization (with pmap dim, without batch dim)
    ref_x_train = jnp.zeros((local_device_count, 10, 5, 2))
    state = updater.init(super_subk, ref_x_train)
    print(f"State initialized", flush=True)

    state_n_params = hk.data_structures.tree_size(state.params)
    state_n_bytes = hk.data_structures.tree_bytes(state.params)
    print(f"param count: {state_n_params:,}", flush=True)
    print(f"param size:  {(state_n_bytes / 1024):,.2f} kB", flush=True)

    """Training loop"""
    print_header("Train loop")
    ave_aux = defaultdict(float)
    buffer_dict_aux = None
    step_init = int(state.step[0]) # get step number from device 0
    t_time = time.time()
    print(f"Entering training loop before step {step_init}", flush=True)
    print("\tSampling probs ")
    print("\t" + ", ".join([f'd={d}: {n_vars_probs[j].item():4.3f}' for j, d in enumerate(n_vars_choices)]), flush=True)

    key_loop = random.PRNGKey(kwargs.seed)
    for t in range(step_init, kwargs.n_steps + 1):

        """Update step"""
        key_loop, subk = random.split(key_loop)
        next_d = random.choice(subk, n_vars_choices, p=n_vars_probs)
        batch = next(train_iters[int(next_d)])

        assert batch['x_obs'].shape[-3] == kwargs.train_n_obs + kwargs.train_n_int, \
            f"Dataset/buffer returned wrong number of observational data. Got {batch['x_obs'].shape[-3]} " \
            f"for train_n_obs=`{kwargs.train_n_obs}` and train_n_int=`{kwargs.train_n_int}`. " \
            f"Double check that train `config_domain` samples sufficient obserations, more than requested by kwargs. "
        assert batch['x_int'].shape[-3] == kwargs.train_n_int, \
            f"Dataset/buffer returned wrong number of interventional data. Got {batch['x_int'].shape[-3]} " \
            f"for train_n_int=`{kwargs.train_n_int}`. "\
            f"Double check that train `config_domain` samples sufficient obserations, more than requested by kwargs. "

        # update step
        state, logs = updater.update(state, batch, t)

        """Logging"""
        # periodic checks use `t` instead of `state.step`, preserving JAX async dispatch by not accessing `state`
        ave_aux = update_ave(ave_aux, logs)
        log_dict = {}
        if (kwargs.log_every and ((t + 1) % kwargs.log_every) == 0):
            # training scalars
            ave = retrieve_ave(ave_aux)
            ave_aux = defaultdict(float)
            train_metrics = {
                **{f"train/step_{_k}": _v for _k, _v in ave.items()},
                "train/step_time":  (time.time() - t_time) / kwargs.log_every, # to capture async dispatch
            }
            log_dict.update(jax.device_get(train_metrics))

            # buffer scalars
            buffer_dict, buffer_dict_aux = ds.log_scalars_dict(aux=buffer_dict_aux, steps=kwargs.log_every)
            log_dict.update(buffer_dict)
            t_time = time.time()

        # evaluation on validation data
        if (kwargs.eval_every and ((t + 1) % kwargs.eval_every) == 0):
            # state is sharded per-device during training; retrieve copy from device 0 for evaluation
            state_chief = get_first(state)
            eval_metrics = evaluate(model, state_chief, ds, step=t)
            log_dict.update(jax.device_get(eval_metrics))

        # log step
        if len(log_dict) > 0:
            print(f"\nLogged after step {t}:", flush=True)
            pprint.pprint(log_dict)

        # check for nans
        for k, v in {**batch, **logs}.items():
            if jax.device_get(jnp.isnan(v).any()).item():
                raise ValueError(f"Got NaNs: {k}; \nBatch: {pprint.pformat(batch)} \nLogs: {pprint.pformat(logs)} ")

    # final checkpoint before exiting
    print(f"Exited training loop after step {int(jax.device_get(state.step[0])) - 1}")
    if kwargs.checkpoint:
        updater.checkpoint(state, int(jax.device_get(state.step[0])))

    # finish and clear up the async data workers and queue of the buffer
    ds.finish()
    print(f"\nFinished training. ", flush=True)
    exit(0)
