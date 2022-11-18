import multiprocessing
import time
import psutil
from collections import defaultdict
import functools

import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import numpy as onp

from avici.definitions import RNG_ENTROPY_TRAIN
from avici.buffer import SharedState, get_soft_sample_insert_ratio, is_buffer_filled
from avici.utils.tf import structured_py_function
import pyarrow.plasma as plasma

def _device_put_sharded(sharded_tree, devices):
    """Taken from  https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/dataset.py """
    leaves, treedef = jax.tree_util.tree_flatten(sharded_tree)
    n = leaves[0].shape[0]
    return jax.device_put_sharded([jax.tree_util.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)],
                                  devices)

def _double_cache(ds):
    """Taken from https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/dataset.py

    Keeps at least two batches on the accelerator.
    The current GPU allocator design reuses previous allocations. For a training
    loop this means batches will (typically) occupy the same region of memory as
    the previous batch. An issue with this is that it means we cannot overlap a
    host->device copy for the next batch until the previous step has finished and
    the previous batch has been freed.
    By double buffering we ensure that there are always two batches on the device.
    This means that a given batch waits on the N-2'th step to finish and free,
    meaning that it can allocate and copy the next batch to the accelerator in
    parallel with the N-1'th step being executed.
    Args:
    ds: Iterable of batches of numpy arrays.
    Yields:
    Batches of sharded device arrays.
    """
    batch = None
    devices = jax.local_devices()
    for next_batch in ds:
        assert next_batch is not None
        next_batch = _device_put_sharded(next_batch, devices)
        if batch is not None:
            yield batch
        batch = next_batch
    if batch is not None:
        yield batch


class AsyncBufferDataset:
    """
    Class for maintaining and sampling data batches from several buffers that are continually asynchronously updated

    If there are exceptions and errors after the main program exit()'s
    it is often because multiprocessing elements do not get cleaned up properly.
    Make sure pools, queues, processes are finished and emptied properly
    at the end of the main program.
    """

    def __init__(self, *,
                 seed,
                 config,
                 buffer_class,
                 buffer_size,
                 train_n_observations_obs,
                 train_n_observations_int,
                 batch_dims_train=None,
                 batch_dims_test=None,
                 double_cache_train=True,
                 prefetch=tf.data.experimental.AUTOTUNE,
                 n_workers=None,
                 n_listeners=1,
                 verbose=True,
                 queue_max_size=100,
                 object_store_gb=10.0,
                 ):

        self.config = config
        self.buffer_class = buffer_class
        self.train_n_observations_obs = train_n_observations_obs
        self.train_n_observations_int = train_n_observations_int
        self.batch_dims_train = batch_dims_train
        self.batch_dims_test = batch_dims_test
        self.double_cache_train = double_cache_train
        self.prefetch = prefetch
        self.verbose = verbose

        if self.verbose:
            print(f"AsyncBufferDataset", flush=True)

        # create plasma object store server
        # this emulates `with plasma.start_plasma_store as self._plasma_store: ...` and is closed in self.finish())
        self._plasma_store = plasma.start_plasma_store(int(object_store_gb * 1024 * 1024 * 1024))
        self._plasma_store_name, self._proc = type(self._plasma_store).__enter__(self._plasma_store)

        self._main_plasma_client = plasma.connect(self._plasma_store_name)

        # setup workers via async multiprocessing
        try:
            # only on linux but only correct call on cluster (psutil.cpu_count is false)
            cpu_count = len(psutil.Process().cpu_affinity())
        except AttributeError:
            cpu_count = psutil.cpu_count(logical=True)

        self.n_workers = n_workers or cpu_count
        self.n_listeners = n_listeners if self.n_workers >= 16 else 1 # with few CPUs don't need more than one listener

        n_processes = self.n_workers + self.n_listeners + 1
        self._pool = multiprocessing.Pool(processes=n_processes)
        self._endless_jobs = []
        self._listener_jobs = []
        if self.verbose:
            print(f"cpu count: {cpu_count}", flush=True)
            print(f"processes: {n_processes} "
                  f"({self.n_workers} workers,"
                  f" {self.n_listeners} listeners,"
                  f" {1} load balancer)", flush=True)

        # init shared state
        manager = multiprocessing.Manager()

        self._state = SharedState(
            queues=[manager.Queue(maxsize=queue_max_size) for _ in range(self.n_listeners)],
            buffers=defaultdict(dict),
            is_alive=manager.Event(),
            next_job=manager.Value(int, self.config["train_n_vars"][0]),
            plasma_store_name=self._plasma_store_name,
        )
        self._state.is_alive.set()

        # buffers
        for descr, spec in self.config["data"].items():
            is_train = descr == "train"
            for n_vars in self.config["train_n_vars"] if is_train else self.config["test_n_vars"]:
                size = buffer_size if is_train else self.config["test_n_datasets"]
                self._state.buffers[descr][n_vars] = self.buffer_class.init(manager, size)

        # launch workers that fill the buffers
        self._launch(seed)


    """Buffer"""

    def get_qsize(self):
        return [q.qsize() for q in self._state.queues]

    def get_max_size(self):
        return self._tree_map(lambda b: b.max_size, self._state.buffers)

    def get_size(self):
        return self._tree_map(lambda b: b.state.size, self._state.buffers)

    def get_items_added(self):
        return self._tree_map(lambda b: b.metrics.items_added, self._state.buffers)

    def get_items_sampled(self):
        return self._tree_map(lambda b: b.metrics.items_sampled, self._state.buffers)

    def get_soft_sample_insert_ratio(self):
        return self._tree_map(get_soft_sample_insert_ratio, self._state.buffers["train"])

    def get_is_filled(self):
        return self._tree_map(is_buffer_filled, self._state.buffers)

    def log_scalars_dict(self, aux=None, steps=1):
        assert steps > 0
        items_added_ = self.get_items_added()["train"]
        items_sampled_ = self.get_items_sampled()["train"]
        sample_insert_ratio_ = self.get_soft_sample_insert_ratio()

        # init running counts of items added and sampled
        if aux is None:
            aux = {}
            aux.update({f"added_d={n_vars}": 0 for n_vars in items_added_.keys()})
            aux.update({f"sampled_d={n_vars}": 0 for n_vars in items_sampled_.keys()})

        # make scalars
        d = {
            f"buffer/qsize": sum(self.get_qsize()) / len(self._state.queues),
            f"buffer/plasma_store": len(self._main_plasma_client.list()),
        }
        for n_vars in items_added_.keys():
            d.update({f"buffer/added_per_step-d={n_vars}": (items_added_[n_vars] - aux[f"added_d={n_vars}"]) / steps})
            d.update({f"buffer/sampled_per_step-d={n_vars}": (items_sampled_[n_vars] - aux[f"sampled_d={n_vars}"]) / steps})
            d.update({f"buffer/sample_insert_ratio-d={n_vars}": sample_insert_ratio_[n_vars]})

            # update aux counts
            aux[f"added_d={n_vars}"] = items_added_[n_vars]
            aux[f"sampled_d={n_vars}"] = items_sampled_[n_vars]

        return d, aux

    def _tree_map(self, f, tree, *args, **kwargs):
        """Apply f to every leaf and return result"""
        if type(tree) in [dict, defaultdict, multiprocessing.managers.DictProxy]:
            return {k: self._tree_map(f, subtree, *args, **kwargs) for k, subtree in tree.items()}
        else:
            return f(tree, *args, **kwargs)

    def _flatten(self, tree):
        """Flatten nested dict into list"""
        if type(tree) in [dict, defaultdict, multiprocessing.managers.DictProxy]:
            for _, subtree in tree.items():
                yield from self._flatten(subtree)
        else:
            yield tree


    """Async multiprocessing"""

    def _launch_async_workers(self, descr, n_vars, seed, size=None, endless=False):
        """launches async workers that fill buffer `buffer_id`"""
        assert endless or (size is not None)
        n_workers_needed = self.n_workers if size is None else min(self.n_workers, size)
        worker_seeds = seed.spawn(n_workers_needed)
        jobs = []
        for j, worker_seed in enumerate(worker_seeds):
            it = None if endless else range(j, size, n_workers_needed)
            p = self._pool.apply_async(
                self.buffer_class.async_worker,
                (j, self._state, worker_seed, it, descr, n_vars, self.config))

            p.daemon = True
            jobs.append(p)

        return jobs


    def _launch(self, seed, t_print=20.0):
        """
        Launch producer pipeline. Fill all buffers once and wait until done. Then launch
        endless workers for train buffers
        """
        if self.verbose:
            print(f"Filling all buffers...", flush=True)

        self.ds_seed, loop_seed, *init_seeds = \
            onp.random.SeedSequence(entropy=(RNG_ENTROPY_TRAIN, seed)).spawn(len(self._state.buffers) + 2)

        # put load balancer to work
        plb = self._pool.apply_async(self.buffer_class.load_balanc_async, (self._state,))
        plb.daemon = True
        self._endless_jobs.append(plb)

        # put listeners to work (filling the buffer with produced data on queues)
        for j in range(self.n_listeners):
            pli = self._pool.apply_async(self.buffer_class.insert_async, (j, self._state))
            pli.daemon = True
            self._listener_jobs.append(pli)

        # sequentially fill all buffers using `n_workers` workers that produce for the queues
        init_jobs = []
        for (descr, buffer_dict), seed in zip(self._state.buffers.items(), init_seeds):
            for n_vars, buffer in buffer_dict.items():
                init_jobs += self._launch_async_workers(descr, n_vars, seed, size=buffer.max_size, endless=False)

        # wait for buffers to be full
        t_ref = time.time()
        while True and self.verbose:
            if all(list(self._flatten(self.get_is_filled()))):
                break
            if time.time() > t_ref + t_print:
                sizes, max_sizes = self.get_size(), self.get_max_size()
                strg = f"buffers: "
                for descr, buffer_dict in self._state.buffers.items():
                    strg += f"{descr}: "
                    for n_vars, _ in buffer_dict.items():
                        strg += f"{sizes[descr][n_vars]:>3d}/{max_sizes[descr][n_vars]}  "
                print(strg, flush=True)
                t_ref = time.time()

                # check for raised exceptions in workers
                try:
                    if any([not job.successful() for job in init_jobs]):
                        break
                except ValueError as e:
                    if "multiprocessing.pool.ApplyResult" in str(e) and "not ready" in str(e):
                        pass  # job is not done yet
                    else:
                        raise e

        for job in init_jobs:
            _ = job.get()  # if verbose == True, returns immediately
        if self.verbose:
            print(f"Buffers filled.", flush=True)
            
        # launch endless workers for train buffers
        self._endless_jobs += self._launch_async_workers("train", None, loop_seed, endless=True)

        if self.verbose:
            print(f"Launched endless background workers for train buffers.", flush=True)


    def finish(self):
        # do not attempt to change the ordering of these calls
        self._state.is_alive.clear()

        # wait to collect results from the train workers through the pool result queue
        for job in self._endless_jobs:
            _ = job.get()

        # when here, all async workers exited and we are done, so kill the listeners
        for j in range(self.n_listeners):
            self._state.queues[j].put('kill')
        for job in self._listener_jobs:
            _ = job.get()

        # clear the queue to avoid any post-exit errors when the script finishes
        for j in range(self.n_listeners):
            while not self._state.queues[j].empty():
                self._state.queues[j].get()
            self._state.queues[j].join()

        self._pool.close()
        self._pool.terminate()
        self._pool.join()

        # close plasma server
        # this emulates the end of indented `with` block (see when initialized)
        self._main_plasma_client.disconnect()
        type(self._plasma_store).__exit__(self._plasma_store, None, None, None)

        if self.verbose:
            print("Terminated AsyncBufferDataset", flush=True)

    """Dataset"""

    def _make_dataset(self, buffer, is_train: bool, batch_dims=None):
        """
        Launch consumer (buffer consumer) tf.data.Dataset iterator to support batching, prefetching, etc.
        for buffers `descr`
        """

        if is_train:
            # plasma client
            client = plasma.connect(self._plasma_store_name)

            # infinite dataset that samples continually from updating buffer
            ds = tf.data.Dataset.from_tensor_slices([0]).repeat(None)  # loop indefinitely
            rng = onp.random.default_rng(self.ds_seed)

            f = functools.partial(self.buffer_class.uniform_sample_from_buffer,
                                  rng=rng,
                                  client=client,
                                  buffer=buffer,
                                  n_obs=self.train_n_observations_obs + self.train_n_observations_int,
                                  n_int=self.train_n_observations_int)

        else:
            # dataset is simply the buffer
            ds = tf.data.Dataset.from_tensor_slices(list(range(buffer.max_size)))

            f = functools.partial(self.buffer_class.index_into_buffer, client=self._main_plasma_client, buffer=buffer)

        ds = ds.map(lambda *args: structured_py_function(func=f,
                                                         inp=[*args],
                                                         Tout=self.buffer_class.dtype),
                    # num_parallel_calls is super important once we have multiple devices
                    # on 4 gpus, empirically this gave 10x data loading performance
                    # (0.05 sec instead of 0.5sec for next(iter) in train loop)
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # create batches
        if batch_dims is not None:
            for i, batch_size in enumerate(reversed(batch_dims)):
                ds = ds.batch(batch_size)

        ds = ds.prefetch(self.prefetch)
        ds = tfds.as_numpy(ds)

        if self.double_cache_train and is_train:
            ds = _double_cache(ds)

        yield from ds


    def make_datasets(self, descr):
        """Generate dict of datasets {n_vars: tf.data.Dataset} for config `descr`"""
        ds = {}
        is_train = descr == "train"
        for n_vars, buffer in self._state.buffers[descr].items():
            batch_dims = self.batch_dims_train[n_vars]["device"] if is_train else self.batch_dims_test[n_vars]["device"]
            ds[n_vars] = self._make_dataset(buffer, is_train=is_train, batch_dims=batch_dims)

            # if not is_train:
            #     self._dump_test_data(descr, n_vars, buffer)

        return ds

    def make_test_datasets(self):
        test_set_ids = [k for k in self.config["data"].keys() if k != "train"]
        return {descr: self.make_datasets(descr) for descr in test_set_ids}
