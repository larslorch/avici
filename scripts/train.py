import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf # avoids dataloader crash after execution finishes
tf.config.set_visible_devices([], 'GPU') # hide gpus to tf to avoid OOM and cuda errors by conflicting with jax

import warnings
warnings.filterwarnings("ignore", message="Some donated buffers were not usable") # long jax warning when not on gpu

import argparse
import jax
import jax.random as random
import jax.numpy as jnp
import haiku as hk
import psutil
import subprocess

import wandb
import math
import functools
import shutil
import time
import pprint
import optax
from collections import defaultdict
from tabulate import tabulate

import multiprocessing
from pathlib import Path

from avici.utils.version_control import str2bool
from avici.definitions import WANDB_ENTITY, PROJECT_DIR, CHECKPOINT_SUBDIR

from avici.models.inference_model import InferenceModelG
from avici.utils.lr import noam_schedule, const_schedule

from avici.modules.backbone import CheckpointingUpdater, SuperUpdater, get_first, estimate_memory

from avici.utils.experiment import update_ave, retrieve_ave
from avici.utils.sugar import kaiming_uniform_init, kaiming_uniform_init_scaled

from avici.utils.eval import evaluate
from avici.utils.plot import visualize_data

from avici.synthetic.data import AsyncBufferDataset
from avici.synthetic.utils_jax import jax_get_train_x, jax_get_x
from avici.synthetic.buffer import FIFOBuffer
from avici.utils.parse import load_data_config

DEBUG_STR = "DEBUG"

def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEBUG_STR)
    parser.add_argument("--descr", default="run")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--online", default=False, type=str2bool)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--wandb_id")
    parser.add_argument("--store_wandb_locally", default=False, type=str2bool)
    parser.add_argument("--detailed_log", default=False, type=str2bool)
    parser.add_argument("--only_eval", default=False, type=str2bool)
    parser.add_argument("--group_scratch", default=False, type=str2bool)
    parser.add_argument("--visualize_diff", default=False, type=str2bool)

    # data
    # parser.add_argument("--config", default="config/debug-linear_additive-0.yaml", type=Path)
    # parser.add_argument("--config", default="config/debug-rff_additive-0.yaml", type=Path)
    parser.add_argument("--config", default="config/debug-sergio-0.yaml", type=Path)
    parser.add_argument("--train_n_obs", default=7, type=int)
    parser.add_argument("--train_n_int", default=9, type=int)
    parser.add_argument("--p_obs_only", default=0.0, type=float) # prob only sampling  (n_obs + n_int) observ. data
    parser.add_argument("--buffer_size", default=10, type=int)
    parser.add_argument("--n_listeners", default=8, type=int)
    parser.add_argument("--p_small_data", default=0.0, type=float) # prob using less data input
    parser.add_argument("--p_small_factor", default=0.1, type=float)
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--estimate_eval_memory", default=False, type=str2bool)

    parser.add_argument("--mem_check", default=False, type=str2bool)
    parser.add_argument("--mem_check_N", default=100, type=int)
    parser.add_argument("--mem_check_d", default=100, type=int)
    parser.add_argument("--mem_check_fwd", default=True, type=str2bool) # jax default
    parser.add_argument("--mem_alloc")
    parser.add_argument("--preallocate_gpu", default=True, type=str2bool) # jax default
    parser.add_argument("--eval_remat", default=False, type=str2bool)
    parser.add_argument("--optimize_stack", default=False, type=str2bool)
    parser.add_argument("--standardize_v") # deprecated; kept for legacy

    # inference model
    parser.add_argument("--bernoulli", default="sigmoid", choices=["sigmoid", "rbf"])
    parser.add_argument("--loss", default="xent", choices=["xent", "elbo", "nll"])
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--pos_wgt", default=1.0, type=float)
    parser.add_argument("--mask_diag", default=True, type=str2bool)

    # nn
    parser.add_argument("--nn", default="BaseModel")
    parser.add_argument("--block", default="inter-SAB")
    parser.add_argument("--cross", default="Ndd")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--pooling", default="max")
    parser.add_argument("--agg", default="max")
    parser.add_argument("--dim", default=8, type=int)
    parser.add_argument("--out_dim", type=int)
    parser.add_argument("--batch_n", default=3, type=int) # batch size when subsampling N in SAB-subN or SAB-stratN
    parser.add_argument("--chunk_query_size", default=1024, type=int)
    parser.add_argument("--chunk_key_size", default=1024, type=int)
    parser.add_argument("--n_per_block", default=1, type=int)
    parser.add_argument("--n_loc", default=2, type=int)
    parser.add_argument("--ln_loc", default=True, type=str2bool)
    parser.add_argument("--n_glob", default=0, type=int)
    parser.add_argument("--ln_glob", default=True, type=str2bool)
    parser.add_argument("--n_split", default=0, type=int)
    parser.add_argument("--ln_split", default=True, type=str2bool)
    parser.add_argument("--ln_final", default=True, type=str2bool)
    parser.add_argument("--pre_ln", default=True, type=str2bool)
    parser.add_argument("--ln_axis", default="last", type=str, choices=["last", "lasttwo"])
    parser.add_argument("--SAB_num_heads", default=2, type=int)
    parser.add_argument("--key_size", default=5, type=int)
    parser.add_argument("--isab_k", default=2, type=int)
    parser.add_argument("--widening_factor", default=2, type=int)
    parser.add_argument("--long_init", default=False, type=str2bool)
    parser.add_argument("--long_final", default=False, type=str2bool)
    parser.add_argument("--ieee", default=False, type=str2bool)
    parser.add_argument("--intermediate_ffn", default=True, type=str2bool)
    parser.add_argument("--isab_ffn", default=True, type=str2bool)
    parser.add_argument("--skip_connection_e", default=False, type=str2bool)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--final_init_scaling", default=False, type=str2bool)
    parser.add_argument("--relation_net", default=False, type=str2bool)
    parser.add_argument("--identity_embedding", default="IModule")
    parser.add_argument("--matrix_bias", default=True, type=str2bool)
    parser.add_argument("--scan_eval", default=False, type=str2bool)
    parser.add_argument("--scan_eval_size", default=500, type=int)
    parser.add_argument("--cosine_sim", default=False, type=str2bool)
    parser.add_argument("--cosine_temp_init", default=2.0, type=float)
    parser.add_argument("--relational_encoder", default=False, type=str2bool)
    parser.add_argument("--attn_only_n", default=False, type=str2bool)
    parser.add_argument("--attn_only_d", default=False, type=str2bool)

    parser.add_argument("--mixture_net", default=True, type=str2bool)
    parser.add_argument("--mixture_k", default=1, type=int)
    parser.add_argument("--mixture_drop", default=0.0, type=float)
    parser.add_argument("--kl_mixt_pen", default=False, type=str2bool)
    parser.add_argument("--kl_mixt_wgt", default=1.0, type=float)

    # acyclicity
    parser.add_argument("--acyc", default="dual", choices=["const", "lin", "dual", "none"])
    parser.add_argument("--acyc_const", default=1.0, type=float)
    parser.add_argument("--acyc_lin", default=1.0, type=float)
    parser.add_argument("--acyc_powit", default=10, type=int)
    parser.add_argument("--acyc_dual_lr", default=1e-4, type=float)
    parser.add_argument("--acyc_inner_step", default=500, type=int)
    parser.add_argument("--acyc_polyak", default=1e-4, type=float)
    parser.add_argument("--acyc_burnin", default=50000, type=int)
    parser.add_argument("--acyc_warmup", default=True, type=str2bool)

    # optimizer
    parser.add_argument("--optimizer", default="lamb", choices=["adam", "lamb"])
    parser.add_argument("--n_steps", default=50, type=int)
    parser.add_argument("--bsu", default=50, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--acc_grad", type=int)
    parser.add_argument("--schedule", default="piecewise_const_200k_300k",
        choices=["const", "noam", "piecewise_const_70k_120k", "piecewise_const_150k_250k", "piecewise_const_200k_300k"])
    parser.add_argument("--grad_clip", default=True, type=str2bool)
    parser.add_argument("--grad_clip_value", default=1.0, type=float)
    parser.add_argument("--scaled_init", default=False, type=str2bool, help="scales kamining init by 1/layers")
    parser.add_argument("--lr_scaling", default="sqrt", choices=["sqrt", "linear"])
    parser.add_argument("--curriculum", default="equal-nvars")

    # logging
    parser.add_argument("--log_every", default=1, type=int)
    parser.add_argument("--eval_every", default=10, type=int)
    parser.add_argument("--checkpoint_every", default=20000, type=int)
    parser.add_argument("--checkpoint", default=True, type=str2bool)
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--visualize_data_distribution", default=False, type=str2bool)
    # parser.add_argument("--debug_data", default=False, type=str2bool)

    parser.add_argument("--lim_n_obs", type=int) # dummy

    # relaunching
    parser.add_argument("--relaunch", default=False, type=str2bool,
                        help="If True, ends training after `relaunch_bsub` minutes and executes `relaunch_script`"
                             "which should be a `bsub ...` command")
    parser.add_argument("--relaunch_bsub", type=str)
    parser.add_argument("--relaunch_after", type=float, help="time in minutes after which to relaunch script")

    return parser



def get_nn(args):
    if args.nn == "BaseModel":
        nn_args = dict(
            model_class=args.nn,
            model_kwargs={"dim": args.dim,
                          "out_dim": args.out_dim,
                          "layers": args.n_loc,
                          "key_size": args.key_size,
                          "num_heads": args.SAB_num_heads,
                          "widening_factor": args.widening_factor,
                          "dropout": args.dropout_rate,
                          "ln_axis": {"last": -1, "lasttwo": (-2, -1)}[args.ln_axis],
                          "n_mixtures": args.mixture_k or 1,
                          "mixture_drop": args.mixture_drop,
                          "cosine_sim": args.cosine_sim,
                          "cosine_temp_init": args.cosine_temp_init,
                          "relational_encoder": args.relational_encoder,
                          "attn_only_n": args.attn_only_n,
                          "attn_only_d": args.attn_only_d,
                          })

    else:
        raise ValueError()

    print(f"model kwargs:\n{pprint.pformat(nn_args, indent=4)}", flush=True)

    return nn_args


def print_header(strg):
    print("\n" + "=" * 25 + f" {strg} " + "=" * 25, flush=True)


if __name__ == "__main__":

    # jnp.set_printoptions(precision=4, suppress=True)
    kwargs = make_parser().parse_args()
    run_name = kwargs.descr
    if kwargs.ieee:
        raise NotImplementedError("ieee got unhooked at various locations for now")

    if not kwargs.preallocate_gpu:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

    t_init_script = time.time()
    assert not kwargs.relaunch or (kwargs.relaunch_after is not None and kwargs.relaunch_bsub is not None)

    # pmap debugging locally
    # # simulate pmap devices with local cpus
    # os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count=2"

    DIR = PROJECT_DIR

    if kwargs.only_eval:
        kwargs.project += "_EVAL"
    if kwargs.wandb_id is None:
        kwargs.wandb_id = wandb.util.generate_id()

    wandb.init(
        id=kwargs.wandb_id,
        resume="allow",
        project=kwargs.project,
        group=run_name,
        name=run_name,
        config=vars(kwargs),
        entity=WANDB_ENTITY,
        dir=DIR,
        save_code=True,
        job_type="dev",
        mode="online" if (kwargs.online and not kwargs.smoke_test) else "disabled",
    )
    print("Working directory at ", DIR, flush=True)

    if kwargs.mem_check:
        # os.environ['XLA_FLAGS'] = f"--xla_python_client_preallocate=2"
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

    if kwargs.mem_alloc:
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = kwargs.mem_alloc


    # load config
    config = load_data_config(kwargs.config)
    if config is None:
        raise SyntaxError("config is None; make sure there are no false tabs and indents in the .yaml file")

    # distributed training
    host_count = jax.process_count()
    devices = jax.devices() # if host_count=1, device_count==local_device_count
    device_count = jax.device_count()
    local_devices = jax.local_devices()
    local_device_count = jax.local_device_count()
    try:
        # only on linux but only correct call on cluster (psutil.cpu_count is false)
        cpu_count = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpu_count = psutil.cpu_count(logical=True)

    print(f"jax backend:   {jax.default_backend()} ")
    print(f"hosts:         {host_count}")
    print(f"devices:       {device_count}  {devices}")
    print(f"local_devices: {local_device_count}  {local_devices}")
    print(f"cpu count:     {cpu_count} ")
    print(f"cpu count (logical):     {psutil.cpu_count(logical=True)} ")
    # print(f"tf cpu count:  {len(tf.config.get_visible_devices('CPU'))}")
    assert len(tf.config.get_visible_devices('GPU')) == 0, "tf shouldn't have access to GPUs as it interferes with jax"
    try:
        cuda_version_str = subprocess.check_output(['nvcc','--version']).decode('ascii')
        print(f"cuda version:\n{cuda_version_str}")
    except FileNotFoundError:
        print(f"cuda not available")
        pass

    # python venv
    print(f"python venv prefix: {sys.prefix}")
    print(f"python venv base prefix: {sys.base_prefix}")

    wandb.log({"cpu_count": cpu_count, "device_count": device_count}, step=0)

    if host_count > 1:
        raise NotImplementedError("Not implemented")
        # checkout https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/dataset.py for data handling

    # The effective batch size is the batch size accross all hosts and devices. In a
    # multi-host training setup each host will only see a batch size of
    # `effective batch size / host_count`.

    # assume linear scaling of memory as function of nvars; skip when batch size < 1
    def bs(_d_, bsu=kwargs.bsu):
        # return math.floor(0.1 + bsu / _d_)
        return math.floor(0.1 + bsu * (1 - math.exp(- 0.1 * _d_)) / _d_) # damping 0.4-1.0x for small nvars

    batch_sizes = {
        d: {"device": [local_device_count, bs(d)], "effective": device_count * bs(d)}
        for d in config["train_n_vars"] if not bs(d) < 1.0
    }
    batch_sizes_test = {
        d: {"device": [1]} for d in config["test_n_vars"]
    }

    min_batch_size_device = min(b_["device"][-1] for _, b_ in batch_sizes.items())
    max_batch_size_effective = max(b_["effective"] for _, b_ in batch_sizes.items())
    if kwargs.acc_grad is not None:
        max_batch_size_effective *= kwargs.acc_grad

    """ ========== Data ========== """
    print_header("Data")

    # init async dataset with `n_workers` filling several data buffers
    # launches async executors with `n_workers` filling a data buffer
    t_buffer_start = time.time()
    buffer_kwargs = dict(
        seed=kwargs.seed,
        config=config,
        train_n_observations_obs=kwargs.train_n_obs,
        train_n_observations_int=kwargs.train_n_int,
        buffer_class=FIFOBuffer,
        buffer_size=kwargs.buffer_size,
        batch_dims_train=batch_sizes,
        batch_dims_test=batch_sizes_test,
        double_cache_train=True, # jax.default_backend() == 'gpu',
        n_listeners=kwargs.n_listeners,
        n_workers=kwargs.n_workers,
    )
    ds = AsyncBufferDataset(**buffer_kwargs)
    print(f"buffer kwargs:\n{pprint.pformat({k: v for k, v in buffer_kwargs.items() if k != 'config'}, indent=4)}", flush=True)

    wandb.log({"time_fill_buffer": time.time() - t_buffer_start}, step=0)

    train_iters = ds.make_datasets("train")
    print(f"Instantiated datasets", flush=True)

    # define reference data batch for visualization (without pmap dim, with batch dim)
    ref_batches_test = ds.make_test_ref_batches(n_batches=10)
    print(f"Instantiated reference batches", flush=True)
    print(f"train n_obs: {kwargs.train_n_obs}")
    print(f"      n_int: {kwargs.train_n_int}")
    print(f"           = {kwargs.train_n_obs + kwargs.train_n_int}")

    print("train batch sizes")
    pprint.pprint(batch_sizes)

    """ ========== Model, loss, and updater ========== """
    print_header("Model")

    print("Acyclicity: ")
    if kwargs.acyc == "const":
        acyc_constr = lambda _penalty, _tt, _dual: _penalty * jnp.array(kwargs.acyc_const)
        acyc_constr.__name__ = kwargs.acyc
        print(f"\tconst: {kwargs.acyc_const} * t")

    elif kwargs.acyc == "lin":
        acyc_constr = lambda _penalty, _tt, _dual: _penalty * jnp.maximum(jnp.array(0.0), jnp.array((_tt - kwargs.acyc_burnin)  * kwargs.acyc_lin))
        acyc_constr.__name__ = kwargs.acyc
        print(f"\tlinear: {kwargs.acyc_lin} * t")

    elif kwargs.acyc == "dual":
        acyc_constr = lambda _penalty, _tt, _dual: _penalty * _dual
        acyc_constr.__name__ = kwargs.acyc
        print(f"\tdual; warmup = {kwargs.acyc_warmup}")

    elif kwargs.acyc == "none":
        acyc_constr = None
        kwargs.mask_diag = False
        print(f"\tnone; fixing mask_diag=False")

    else:
        raise KeyError(kwargs.acyc)

    # net and loss
    neural_net_kwargs = get_nn(kwargs)
    inference_model_kwargs = dict(
        loss=kwargs.loss,
        mixture_net=kwargs.mixture_net,
        mixture_k=kwargs.mixture_k,
        label_smoothing=kwargs.label_smoothing,
        pos_weight=kwargs.pos_wgt,
        train_p_obs_only=kwargs.p_obs_only,
        acyclicity=acyc_constr,
        acyclicity_pow_iters=kwargs.acyc_powit,
        mask_diag=kwargs.mask_diag,
        bernoulli=kwargs.bernoulli,
        kl_mixt_pen=kwargs.kl_mixt_pen,
        kl_mixt_wgt=kwargs.kl_mixt_wgt,
    )
    model = InferenceModelG(**inference_model_kwargs, **neural_net_kwargs)
    print(f"inference model kwargs:\n{pprint.pformat(inference_model_kwargs, indent=4)}", flush=True)

    # summary = hk.experimental.eval_summary(model.net.apply)(get_first(super_ref_obs_single))

    # optimizer
    print_header("Optimizer")
    print("Optimizer: ")
    if kwargs.lr_scaling == "sqrt":
        lr = math.sqrt(max_batch_size_effective) * kwargs.lr
        print(f"\tlr scaling: base_lr = {lr} = sqrt({max_batch_size_effective}) * {kwargs.lr} (sqrt batch size scaling)")

    elif kwargs.lr_scaling == "linear":
        lr = max_batch_size_effective * kwargs.lr
        print(f"\tlr scaling: base_lr = {lr} = {max_batch_size_effective} * {kwargs.lr} (linear batch size scaling)")

    else:
        lr = kwargs.lr

    if kwargs.schedule == "const":
        lr_sched = const_schedule(lr)
        print(f"\tconstant schedule with lr = {lr}")
    elif kwargs.schedule == "noam":
        lr_sched = noam_schedule(max_lr=lr, warmup=20000)
        print(f"\tnoam schedule with lr = {lr}")
    elif kwargs.schedule == "piecewise_const_70k_120k":
        lr_sched = optax._src.schedule.piecewise_constant_schedule(lr, {70000: 0.1, 120000: 0.1})
        print(f"\tpiecewise_const_70k_120k schedule with lr = {lr}")
    elif kwargs.schedule == "piecewise_const_150k_250k":
        lr_sched = optax._src.schedule.piecewise_constant_schedule(lr, {150000: 0.1, 250000: 0.1})
        print(f"\tpiecewise_const_150k_250k schedule with lr = {lr}")
    elif kwargs.schedule == "piecewise_const_200k_300k":
        lr_sched = optax._src.schedule.piecewise_constant_schedule(lr, {200000: 0.1, 300000: 0.1})
        print(f"\tpiecewise_const_200k_300k schedule with lr = {lr}")
    else:
        raise KeyError(kwargs.schedule)

    optimizer_modules = []

    if kwargs.grad_clip:
        optimizer_modules.append(optax.clip_by_global_norm(kwargs.grad_clip_value))
        print(f"\tgrad_clip at {kwargs.grad_clip_value}:")

    if kwargs.optimizer == "adam":
        optimizer_modules.append(optax.adam(lr_sched))
        print("\tADAM")
    elif kwargs.optimizer == "lamb":
        optimizer_modules.append(optax.lamb(lr_sched, weight_decay=kwargs.weight_decay))
        print("\tLAMB")
    else:
        raise KeyError(f"Invalid optimizer {kwargs.optimizer}")

    optimizer = optax.chain(*optimizer_modules)

    if kwargs.acc_grad is not None:
        # accumulate gradient for `kwargs.acc_grad` steps before doing an actual update step
        optimizer = optax.MultiSteps(opt=optimizer, every_k_schedule=kwargs.acc_grad, use_grad_mean=True)
        print(f"\tAccumulating grads over {kwargs.acc_grad} steps")

    # updater
    updater_kwargs = dict(
        acyclicity_dual_lr=kwargs.acyc_dual_lr,
        acyclicity_inner_step=kwargs.acyc_inner_step,
        acyclicity_burnin=kwargs.acyc_burnin,
        acyclicity_warmup=kwargs.acyc_warmup,
        polyak_rate=kwargs.acyc_polyak,
        local_device_count=local_device_count
    )
    print(f"updater kwargs:\n{pprint.pformat(updater_kwargs, indent=4)}", flush=True)
    updater = SuperUpdater(net_init=model.net.init, loss_fn=model.loss, opt=optimizer, **updater_kwargs)

    if kwargs.checkpoint and not kwargs.mem_check:
        CHECKPOINT_DIR = kwargs.checkpoint_dir or (DIR / CHECKPOINT_SUBDIR / kwargs.project / run_name)

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

    """ ========== Initialize state ========== """
    print_header("State")

    # same rng on every device
    key = random.PRNGKey(kwargs.seed)
    super_subk = jnp.broadcast_to(key, (local_device_count,) + key.shape)

    # data batch for initialization (with pmap dim, without batch dim)
    # ref_batch_train = next(train_iters[config["train_n_vars"][0]])
    ref_x_train = jnp.zeros((local_device_count, 10, 5, 2))

    state = updater.init(super_subk, ref_x_train)
    print(f"State initialized", flush=True)

    # # print summary
    # summary = hk.experimental.tabulate(updater._inner.update)(state, dict(x_obs=ref_x_train,
    #                                                                       x_int=ref_x_train,
    #                                                                       g=jnp.ones((local_device_count, 5, 5))), 0)
    # for line in summary.split('\n'):
    #     print(line, flush=True)

    """Training loop"""
    print_header("Train loop")
    ave_aux = defaultdict(float)
    buffer_dict_aux = None
    step_init = int(state.step[0]) # get step number from device 0
    t_time = time.time()
    print(f"Entering training loop before step {step_init}", flush=True)

    def get_n_vars_probs(_):
        n_vars_ = jnp.array(list(batch_sizes.keys()))
        eff_bs_ = jnp.array([batch_sizes[n_vars]["effective"] for n_vars in n_vars_.tolist()])

        if kwargs.curriculum == "uniform":
            p_ = jnp.ones_like(eff_bs_)
        elif kwargs.curriculum == "equal":
            p_ = 1.0 / eff_bs_
        elif kwargs.curriculum == "equal-sqrt":
            p_ = 1.0 / (eff_bs_ * jnp.sqrt(eff_bs_))
        elif kwargs.curriculum == "equal-lin":
            p_ = 1.0 / (eff_bs_ * eff_bs_)
        elif kwargs.curriculum == "equal-nvars":
            p_ = n_vars_ / eff_bs_
        else:
            raise ValueError(f"Unknown curriculum {kwargs.curriculum}")
        return n_vars_, p_


    # debugging
    # unique_data_seen = {n_vars: set() for n_vars in config["train_n_vars"]}
    # steps_ctr = {n_vars: 0 for n_vars in config["train_n_vars"]}
    key_loop = random.PRNGKey(kwargs.seed)
    for t in range(step_init, kwargs.n_steps + 1):

        """Update step"""
        t_data_load = time.time()
        key_loop, subk = random.split(key_loop)
        n_vars_choices, selection_probs = get_n_vars_probs(t)
        d = random.choice(subk, n_vars_choices, p=selection_probs)
        batch = next(train_iters[int(d)])  # batch has type `jax.interpreters.pxla.ShardedDeviceArray` when using gpu backend, otherwise `np.ndarray`
        ave_aux = update_ave(ave_aux, {"time_data": time.time() - t_data_load})

        assert batch['x_obs'].shape[-3] == kwargs.train_n_obs + kwargs.train_n_int, \
            f"Dataset/buffer returned wrong number of observational data. Got {batch['x_obs'].shape[-3]} " \
            f"for train_n_obs=`{kwargs.train_n_obs}` and train_n_int=`{kwargs.train_n_int}`." \
            f"Double check that train config samples more obserations than requested by kwargs "
        assert batch['x_int'].shape[-3] == kwargs.train_n_int, \
            f"Dataset/buffer returned wrong number of interventional data. Got {batch['x_int'].shape[-3]} " \
            f"for train_n_int=`{kwargs.train_n_int}`"\
            f"Double check that train config samples more obserations than requested by kwargs "

        # randomly choose whether to use less data (suboptimal here since not sharded on device inside inference_model)
        if kwargs.p_small_data > 0.0:
            key_loop, subk = random.split(key_loop)
            if random.bernoulli(subk, p=jnp.array(kwargs.p_small_data)):
                train_n_obs_small = math.ceil(kwargs.train_n_obs * kwargs.p_small_factor)
                train_n_int_small = math.ceil(kwargs.train_n_int * kwargs.p_small_factor)
                key_loop, subk = random.split(key_loop)
                batch["x_obs"] = batch["x_obs"][..., random.permutation(subk, batch["x_obs"].shape[-3])[:train_n_obs_small + train_n_int_small], :, :]
                key_loop, subk = random.split(key_loop)
                batch["x_int"] = batch["x_int"][..., random.permutation(subk, batch["x_int"].shape[-3])[:train_n_int_small], :, :]

        # update step
        state, logs = updater.update(state, batch, t)

        """Bookkeeping"""
        if t == 0:
            # validate initial loss
            print(f"Loss at t = {t}: {jax.device_get(logs['loss'][0]).item()}", flush=True)
            print(f"Curriculum `{kwargs.curriculum}` weights: {[f'd={d}: {(selection_probs[j]/selection_probs.sum()).item():4.3f}' for j, d in enumerate(n_vars_choices)]}", flush=True)
            wandb.log({"time_init": t_time - t_init_script}, step=0)
            print("time_init", time.time() - t_init_script, flush=True)

        # check relaunch
        if kwargs.relaunch:
            assert '--relaunch_bsub' in sys.argv, f"Need to provide `relaunch_bsub` but got argv: {sys.argv}"
            # currently the dataset iterator state is not saved, so relaunch is not completely identical
            if time.time() - t_init_script > kwargs.relaunch_after * 60 and t < kwargs.n_steps:
                # checkpoint state
                if kwargs.checkpoint:
                    print(f"Relaunch checkpoint at step {t}")
                    updater.checkpoint(state, t)
                print(f"Exiting training after {(time.time() - t_init_script) / 60.0:.0f} mins")

                # modify string args to parse correctly in bsub command
                _bsub_cmd = kwargs.relaunch_bsub.replace("\\\"", "\"")
                _argv = sys.argv
                _argv[sys.argv.index("--relaunch_bsub") + 1] = _argv[sys.argv.index("--relaunch_bsub") + 1].replace("\"", "\\\"")
                for key_, value_ in kwargs.__dict__.items():
                    if type(value_) == str and ("--" + key_) in sys.argv:
                        _argv[sys.argv.index("--" + key_) + 1] = "\'" + _argv[sys.argv.index("--" + key_)  + 1] + "\'"
                if kwargs.checkpoint:
                    _argv += ["--checkpoint_dir", "\'" + str(CHECKPOINT_DIR) + "\'"]
                if kwargs.wandb_id is not None:
                    _argv += ["--wandb_id", "\'" + kwargs.wandb_id + "\'"]

                # send relaunch job to cluster
                slurm_addon = " --wrap " if kwargs.relaunch_bsub[:6] == "sbatch" else ""
                _cmd = _bsub_cmd + f"{slurm_addon} \"python " + " ".join(_argv) + "\""
                print(f"Execute command: \n>\n{_cmd}\n")
                os.system(_cmd)
                wandb.finish()

                # finish and clear up the async data workers and queue to avoid errors
                ds.finish()
                print("Successfully finished.", flush=True)
                exit(0)

        # for checks use `t` instead of `state.step` because it preserves JAX async dispatch (by not accessing `state`)
        # logging
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
            # debugging
            # train_metrics.update({f"buffer/unique-d={n_vars}": len(d) for n_vars, d in unique_data_seen.items()})
            # train_metrics.update({f"buffer/prop_unique-d={n_vars}": len(d)/(steps_ctr[n_vars] * batch_sizes[n_vars]["effective"]) for n_vars, d in unique_data_seen.items()})

            log_dict.update(jax.device_get(train_metrics))

            # parameter scalars
            # NOTE: if this breaks here, forgot to change `name` in new model class
            param_scalars = {}
            if kwargs.nn in state.params.keys() and "final_matrix_bias" in state.params[kwargs.nn].keys():
                param_scalars["params/mean_matrix_bias"] = state.params[kwargs.nn]["final_matrix_bias"][0].mean().item()

            if kwargs.nn in state.params.keys() and "learned_temp" in state.params[kwargs.nn].keys():
                param_scalars["params/learned_temp"] = state.params[kwargs.nn]["learned_temp"][0].mean().item()

            log_dict.update(param_scalars)

            # buffer scalars
            buffer_dict, buffer_dict_aux = ds.log_scalars_dict(aux=buffer_dict_aux, steps=kwargs.log_every)
            log_dict.update(buffer_dict)

            # # debugging
            # strg = ""
            # for k, v in log_dict.items():
            #     if "unique" in k or "sampled_per_step" in k:
            #         strg += f"{k.split('/')[1:]}:{v} "
            # print(strg, flush=True)

            t_time = time.time()
            print(f"Logged at {t}", flush=True)

        # evaluation (only on test data)
        if (kwargs.eval_every and ((t + 1) % kwargs.eval_every) == 0):
            # state is sharded per-device during training; retrieve copy from device 0
            state_chief = get_first(state)
            eval_metrics = evaluate(model, state_chief, ds, ref_batches_test, step=t,
                                    detailed_log=kwargs.detailed_log, visualize_diff=kwargs.visualize_diff)
            log_dict.update(jax.device_get(eval_metrics))
            print(f"Evaluated at {t}", flush=True)

        # log step to wandb
        if len(log_dict) > 0:
            wandb.log(log_dict, step=t + 1) # state.step = t + 1


        # check for nans
        for k in batch.keys():
            if jax.device_get(jnp.isnan(batch[k]).any()).item():
                raise ValueError(f"Got NaN in batch: {k}; \nLogs: {pprint.pformat(batch)} ")

        for k in logs.keys():
            if jax.device_get(jnp.isnan(logs[k]).any()).item():
                raise ValueError(f"Got NaN in logs: {k}; \nLogs: {pprint.pformat(logs)} ")

        # for k in log_dict.keys():
        #     if "/agg/" in k or "ref_preds" in k:
        #         continue
        #     if jax.device_get(jnp.isnan(log_dict[k]).any()).item():
        #         raise ValueError(f"Got NaN in logs: {k}; \nLogs: {pprint.pformat(logs)} ")

        if kwargs.smoke_test:
            break

    # final checkpoint
    print(f"Exited training loop after step {int(state.step[0])}")
    if kwargs.checkpoint and kwargs.project != DEBUG_STR:
        updater.checkpoint(state, int(state.step[0]))

    # finish wandb process
    wandb_run_dir = os.path.abspath(os.path.join(wandb.run.dir, os.pardir))
    print("wandb run saved in ", wandb_run_dir, flush=True)
    wandb.finish()

    # clean up
    if not kwargs.store_wandb_locally:
        print("Attempting wandb deletion", wandb_run_dir, flush=True)
        try:
            shutil.rmtree(wandb_run_dir)
            print(f"deleted wandb directory `{wandb_run_dir}`")

        except OSError as e:
            # print("Error: %s - %s." % (e.filename, e.strerror))
            print("wandb dir not deleted.")
            pass

    # finish and clear up the async data workers and queue to avoid errors
    ds.finish()
    print(f"\nFinished training. ", flush=True)
    exit(0)
