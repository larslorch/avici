import os
import json
import pickle
import functools
import jax
import jax.random as random
import jax.numpy as jnp
import haiku as hk
import optax
from optax._src.linear_algebra import global_norm
from jax.tree_util import tree_map

from deepdiff import DeepDiff
from typing import Any, NamedTuple
from pathlib import Path

import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

from avici.definitions import CHECKPOINT_KWARGS


def get_first(super_pytree):
    """Gets values from the first device."""
    return tree_map(lambda leaf: leaf[0], super_pytree)


def make_serializable(d):
    if isinstance(d, dict):
        return {k: make_serializable(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [make_serializable(v) for v in d]
    elif callable(d):
        if hasattr(d, "__name__"):
            return d.__name__
        else:
            return type(d).__name__
    elif isinstance(d, Path):
        return str(d)
    else:
        return d


def fix_json_loading(d):
    if isinstance(d, dict):
        return {((int(k) if k.isdigit() else k) if isinstance(k, str) else k):
                    fix_json_loading(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [fix_json_loading(v) for v in d]
    elif isinstance(d, Path):
        return str(d)
    else:
        return d


class ModelState(NamedTuple):
    step: int
    rng: Any
    opt_state: Any
    params: Any
    dual: Any
    dual_penalty_polyak: Any
    ave_params: Any  # polyak average


class Updater:
    """
    A stateless abstraction around an init_fn/update_fn pair.
    Creates a `ModelState` object and updates it.
    """

    def __init__(self, *, net_init, loss_fn, opt,
                 acyclicity_dual_lr,
                 acyclicity_inner_step,
                 acyclicity_burnin,
                 acyclicity_warmup,
                 polyak_rate=1e-4):

        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = opt
        self.distributed = False
        self.acyclicity_dual_lr = acyclicity_dual_lr
        self.acyclicity_inner_step = acyclicity_inner_step
        self.acyclicity_burnin = acyclicity_burnin
        self.acyclicity_warmup = acyclicity_warmup
        self.polyak_rate = polyak_rate

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, rng, x):
        """Initializes model state of the updater."""
        out_rng, init_rng = random.split(rng)
        params = self._net_init(init_rng, x, True) # is_training = True
        opt_state = self._opt.init(params)
        return ModelState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=params,
            dual=jnp.array(0.0),
            dual_penalty_polyak=jnp.array(0.0),
            ave_params=params,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state, batch, t):
        """Updates the ModelState `state` using data `batch` and returns metrics."""

        new_rng, step_rng = random.split(state.rng)

        # compute loss and gradient
        (loss, aux), loss_grads = jax.value_and_grad(self._loss_fn, has_aux=True)(state.params, state.dual, step_rng, batch, t, True)  # is_training = True

        # optimizer step on params
        opt_update, opt_state = self._opt.update(loss_grads, state.opt_state, state.params) # some optimizers need params
        params = optax.apply_updates(state.params, opt_update)

        # track polyak average of params
        ave_params = optax.incremental_update(
            params, state.ave_params, step_size=0.001)

        # dual step
        dual_penalty = jnp.maximum(aux["acyc"], jnp.array(0.0))
        dual_penalty_ave = jax.lax.cond(
            t == 0,
            lambda _: dual_penalty,
            lambda _: optax.incremental_update(dual_penalty, state.dual_penalty_polyak, step_size=self.polyak_rate),
            operand=None)

        if self.acyclicity_warmup:
            dual_lr = jnp.minimum(self.acyclicity_dual_lr, t * self.acyclicity_dual_lr / self.acyclicity_burnin)
            effective_burnin = 0
        else:
            dual_lr = self.acyclicity_dual_lr
            effective_burnin = self.acyclicity_burnin

        dual = jax.lax.cond(
            (jnp.mod(t, self.acyclicity_inner_step) == 0) & (t > effective_burnin),
            lambda _: state.dual + dual_lr * dual_penalty_ave,
            lambda _: state.dual,
            operand=None)

        # state
        new_state = ModelState(
            step=state.step + 1,
            rng=new_rng,
            opt_state=opt_state,
            params=params,
            dual=dual,
            dual_penalty_polyak=dual_penalty_ave,
            ave_params=ave_params,
        )
        # log scalars
        metrics = {
            "loss": loss,
            "dual": dual,
            "grad_norm": global_norm(loss_grads),
            **aux,
        }
        return new_state, metrics


class SuperUpdater:
    """
    `Updater` with distributed training (pmap) functionality.
    """

    def __init__(self, *, net_init, loss_fn, opt,
                 acyclicity_dual_lr,
                 acyclicity_inner_step,
                 acyclicity_burnin,
                 acyclicity_warmup,
                 local_device_count,
                 polyak_rate=1e-4):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = opt
        self.distributed = True
        self.local_device_count = local_device_count
        self.acyclicity_dual_lr = acyclicity_dual_lr
        self.acyclicity_inner_step = acyclicity_inner_step
        self.acyclicity_burnin = acyclicity_burnin
        self.acyclicity_warmup = acyclicity_warmup
        self.polyak_rate = polyak_rate

    @functools.partial(jax.pmap, static_broadcasted_argnums=0)
    def init(self, rng, x):
        """Initializes model state of the updater."""
        out_rng, init_rng = random.split(rng)
        params = self._net_init(init_rng, x, True)  # is_training = True
        opt_state = self._opt.init(params)
        return ModelState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=params,
            dual=jnp.array(0.0),
            dual_penalty_polyak=jnp.array(0.0),
            ave_params=params,
        )

    # @jax.jit # comment this in, on top of pmap, to debug NaN's inside pmap
    @functools.partial(jax.pmap, in_axes=(0, 0, 0, None), axis_name="i", static_broadcasted_argnums=0, donate_argnums=(1, 2))
    def update(self, state, batch, t):
        """Updates the ModelState `state` using data `batch` and returns metrics."""

        new_rng, step_rng = random.split(state.rng)

        # compute loss and gradient
        (loss, aux), loss_grads = jax.value_and_grad(self._loss_fn, has_aux=True)(state.params, state.dual, step_rng, batch, t, True)  # is_training = True

        # take the mean of the gradients across all data-parallel replicas
        loss_grads = jax.lax.pmean(loss_grads, axis_name="i")

        # optimizer step on params (will perform same step on every device due to above line)
        opt_update, opt_state = self._opt.update(loss_grads, state.opt_state, state.params) # some optimizers need params
        params = optax.apply_updates(state.params, opt_update)

        # track polyak average of params
        ave_params = optax.incremental_update(
            params, state.ave_params, step_size=0.001)

        # dual step
        dual_penalty = jnp.maximum(jax.lax.pmean(aux["acyc"], axis_name="i"), jnp.array(0.0))
        dual_penalty_ave = jax.lax.cond(
            t == 0,
            lambda _: dual_penalty,
            lambda _: optax.incremental_update(dual_penalty, state.dual_penalty_polyak, step_size=self.polyak_rate),
            operand=None)

        if self.acyclicity_warmup:
            dual_lr = jnp.minimum(self.acyclicity_dual_lr, t * self.acyclicity_dual_lr / self.acyclicity_burnin)
            effective_burnin = 0
        else:
            dual_lr = self.acyclicity_dual_lr
            effective_burnin = self.acyclicity_burnin

        dual = jax.lax.cond(
            (jnp.mod(t, self.acyclicity_inner_step) == 0) & (t > effective_burnin),
            lambda _: state.dual + dual_lr * dual_penalty_ave,
            lambda _: state.dual,
            operand=None)

        # state
        new_state = ModelState(
            step=state.step + 1,
            rng=new_rng,
            opt_state=opt_state,
            params=params,
            dual=dual,
            dual_penalty_polyak=dual_penalty_ave,
            ave_params=ave_params,
        )

        # log scalars
        # note: we log the mean across all hosts/devices
        metrics = {
            "loss": loss,
            "dual": dual,
            "grad_norm": global_norm(loss_grads),
            **aux,
        }
        metrics = jax.lax.pmean(metrics, axis_name="i")

        return new_state, metrics



class CheckpointingUpdater:
    """A didactic checkpointing wrapper around an `Updater` or `SuperUpdater`."""

    def __init__(self, inner, checkpoint_dir, checkpoint_every_n=10000, base_str="checkpoint_", save_kwargs=None):
        self._inner = inner
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n
        self._base_str = base_str
        self._save_kwargs = save_kwargs
        self._just_loaded = False # avoids serialization immediately after loading

    def _checkpoint_paths(self):
        return sorted([p for p in os.listdir(self._checkpoint_dir) if self._base_str in p])

    def checkpoint(self, state, step):
        path = os.path.join(self._checkpoint_dir, f'{self._base_str}{step:07d}.pkl')
        if self._inner.distributed:
            # For distributed training, only save 1 copy of model state
            state = tree_map(lambda leaf: leaf[0], state)
        checkpoint_state = jax.device_get(state)
        print(f'Serializing experiment state to {path}', flush=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint_state, f)
        return

    def init(self, rng, data):
        """Initialize experiment state. """
        if not os.path.exists(self._checkpoint_dir) or not self._checkpoint_paths():
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            print(f'Checkpoint directory at {self._checkpoint_dir}', flush=True)
            if self._save_kwargs is not None:
                with open(os.path.join(self._checkpoint_dir, CHECKPOINT_KWARGS), "w") as file:
                    json.dump(make_serializable(self._save_kwargs), file, indent=4, sort_keys=True)
            return self._inner.init(rng, data)
        else:
            last_checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_paths()[-1])
            print(f'Loading latest checkpoint from {last_checkpoint}', flush=True)
            if self._save_kwargs is not None:
                with open(os.path.join(self._checkpoint_dir, CHECKPOINT_KWARGS), "r") as file:
                    loaded_kwargs = json.load(file)
                    loaded_kwargs = fix_json_loading(loaded_kwargs)
                    if loaded_kwargs != make_serializable(self._save_kwargs):
                        diff = DeepDiff(loaded_kwargs, make_serializable(self._save_kwargs)).pretty()
                        warn_str = f"Specified save_kwargs and those found in checkpoint directory don't match: \n"
                        warnings.warn(warn_str + diff)

            with open(last_checkpoint, 'rb') as f:
                state = pickle.load(f)
                if self._inner.distributed:
                    # For distributed training, replicate copy of model state on each device
                    devices = jax.local_devices()
                    state = tree_map(lambda leaf: jax.device_put_sharded(len(devices) * [leaf], devices), state)
                self._just_loaded = True
                return state

    def update(self, state, data, step):
        """Update experiment state. """
        # `step` is maintained separately to allow for JAX async dispatch inside `state`
        # https://jax.readthedocs.io/en/latest/async_dispatch.html
        if (step % self._checkpoint_every_n) == 0 and step != 0:
            if self._just_loaded:
                self._just_loaded = False
            else:
                self.checkpoint(state, step)

        state, out = self._inner.update(state, data, step)
        return state, out
