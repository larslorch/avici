import time
import math
import numpy as onp
import pyarrow.plasma as plasma

from typing import NamedTuple, Any
from avici.utils.data import onp_subbatch
from avici.utils.parse import init_custom_classes


class SharedArray(NamedTuple):
    max_size: Any
    state: Any
    mem: Any
    state_mem_lock: Any
    metrics: Any
    metrics_lock: Any


class SharedState(NamedTuple):
    queues: Any
    buffers: Any
    is_alive: Any
    next_job: Any
    plasma_store_name: Any


class Sampler:

    dtype = dict(
        g=onp.int32,
        n_vars=onp.int32,
        x_obs=onp.float32,
        x_int=onp.float32,
        n_observations_obs=onp.int32,
        n_observations_int=onp.int32,
        is_count_data=bool,
    )

    @staticmethod
    def generate_data(rng, *, n_vars, spec_list, spec=None):
        # sample random spec
        if spec is None:
            spec = spec_list[rng.choice(len(spec_list))]

        # sample graph sampling
        g = spec.graph(rng=rng, n_vars=n_vars)

        # sample mechanism mechanism and data
        data = spec.mechanism(rng=rng,
                              g=g,
                              n_observations_obs=spec.n_observations_obs,
                              n_observations_int=spec.n_observations_int)

        # collect all data
        collection = dict(
            g=g,
            n_vars=n_vars,
            x_obs=data.x_obs,
            x_int=data.x_int,
            n_observations_obs=spec.n_observations_obs,
            n_observations_int=spec.n_observations_int,
            is_count_data=data.is_count_data,
        )

        # check dims
        if collection["x_obs"].ndim == 2:
            collection["x_obs"] = onp.expand_dims(collection["x_obs"], axis=-1)
        if collection["x_int"].ndim == 2:
            collection["x_int"] = onp.expand_dims(collection["x_int"], axis=-1)

        return collection

    @staticmethod
    def async_worker(j, state, worker_seed, it, descr, n_vars, config):
        """Runs `f` repeatedly and sends results to the q.
        Arguments to worker must not be functions or generators.
        To see errors for debugging, must call job.get() """

        rng = onp.random.default_rng(worker_seed)

        # initialize custom classes inside handle pickling with multiprocessing and custom modules
        config = init_custom_classes(config)

        # init plasma client
        plasma_client = plasma.connect(state.plasma_store_name)

        ctr = j
        for _ in it or infinite_iterator():
            ctr += 1
            if not state.is_alive.is_set():
                break

            # get buffer for which to generate data
            assert n_vars is not None or descr == "train"
            if n_vars is None:
                d = state.next_job.value
            else:
                d = n_vars

            with state.buffers[descr][d].metrics_lock:
                state.buffers[descr][d].metrics.items_added += 1

            # generate data
            data = Sampler.generate_data(
                rng,
                n_vars=d,
                spec_list=config["data"][descr],
            )

            # commit data to plasma store
            object_id = plasma_client.put(data)

            # push object id on random processing queue
            # avoiding querying load status
            q = state.queues[ctr % len(state.queues)]
            q.put(dict(descr=descr, n_vars=d, id=object_id))

        # disconnect plasma client
        plasma_client.disconnect()


class FIFOBuffer(Sampler):

    @staticmethod
    def init(manager, buffer_size):
        max_size = buffer_size

        state = manager.Namespace()
        state.index = 0
        state.size = 0

        metrics = manager.Namespace()
        metrics.items_added = 0
        metrics.items_sampled = 0

        mem = manager.list([None] * buffer_size)

        state_mem_lock = manager.Lock()
        metrics_lock = manager.Lock()

        return SharedArray(
            max_size=max_size,
            state=state,
            mem=mem,
            state_mem_lock=state_mem_lock,
            metrics=metrics,
            metrics_lock=metrics_lock,
        )

    @staticmethod
    def load_balanc_async(state, t_sleep=0.001):
        # small worker that continually updates buffer with highest sample-insert ratio
        while True and state.is_alive.is_set():
            si = {d: get_soft_sample_insert_ratio(buffer) for d, buffer in state.buffers["train"].items()}
            state.next_job.value = max(si, key=si.get)
            time.sleep(t_sleep)

    @staticmethod
    def insert_async(j, state):

        # init plasma client
        plasma_client = plasma.connect(state.plasma_store_name)
        q = state.queues[j]

        while True:
            # get next item from q
            item = q.get()
            if item == 'kill':
                q.task_done()
                break  # kill signal for listener

            b = state.buffers[item["descr"]][item["n_vars"]]

            # FIFO buffer update
            with b.state_mem_lock:
                old_object_id = b.mem[b.state.index]
                b.mem[b.state.index] = item["id"]

                b.state.size = min(b.state.size + 1, b.max_size)
                b.state.index = (b.state.index + 1) % b.max_size

            # remove old data from plasma store
            plasma_client.delete([old_object_id])

            q.task_done()

        # disconnect plasma client
        plasma_client.disconnect()

    @staticmethod
    def index_into_buffer(j, *, client, buffer):
        with buffer.state_mem_lock:
            return client.get(buffer.mem[j])

    @staticmethod
    def uniform_sample_from_buffer(_, *, rng, client, buffer, n_obs, n_int):
        with buffer.state_mem_lock:
            idx = int(rng.choice(buffer.state.size))
            item_id = buffer.mem[idx]
            assert client.contains(item_id)
            item = client.get(item_id, timeout_ms=100)
            assert item != plasma.ObjectNotAvailable

        with buffer.metrics_lock:
            buffer.metrics.items_sampled += 1

        subitem = onp_subbatch(rng, item, n_obs=n_obs, n_int=n_int)
        return subitem


def infinite_iterator():
    """An infinite iterator (enabling for loop like `while True`)"""
    while True:
        yield None


def get_soft_sample_insert_ratio(buffer, const=1e3):
    # softness const ensures smooth load balancing initially
    if buffer.metrics.items_added:
        return (buffer.metrics.items_sampled + const) / buffer.metrics.items_added
    else:
        return math.inf


def is_buffer_filled(buffer):
    return all([item is not None for item in buffer.mem])