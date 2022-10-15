import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import functools
import multiprocessing
import time
import numpy as onp
import tensorflow as tf

from avici.definitions import WANDB_ENTITY, PROJECT_DIR, CLUSTER_GROUP_DIR, CHECKPOINT_SUBDIR, DATA_SUBDIR
from avici.utils.version_control import get_datetime
from avici.utils.tf import structured_py_function



class AsyncExecutor:

    def __init__(self, n_workers=1, listener=None):

        self._listener = listener # must only take q as argument
        if listener is not None:
            n_workers += 1

        self.n_workers = n_workers if n_workers > 0 else multiprocessing.cpu_count()

        self._manager = multiprocessing.Manager()
        self._q = self._manager.Queue()
        self._pool = multiprocessing.Pool(self.n_workers)
        self._jobs = []

    def run(self, target, *args_iter):
        self.launch(target, *args_iter)
        self.finish()

    def launch(self, target, *args_iter):
        # put listener to work first
        if self._listener is not None:
            watcher = self._pool.apply_async(self._listener, (self._q,))

        # fire off workers
        for args in zip(*args_iter):
            job = self._pool.apply_async(_worker, (self._q, target, *args))
            self._jobs.append(job)

    def finish(self):
        # collect results from the workers through the pool result queue
        for job in self._jobs:
            _ = job.get()  # same result as processed by listener

        # when here, all jobs finished and we are done, so kill the listener
        self._q.put('kill')
        self._pool.close()
        self._pool.join()

    def ready(self):
        return all([job.ready() for job in self._jobs])


def _worker(q, target, *args):
    # runs `target` and sends result to the q
    res = target(*args)
    q.put(res)
    return res


def example_listener(q):
    """Listens for messages/results on the q
    Allows easy use of context managers
    """
    while True:
        m = q.get()
        if m == 'kill':
            break # kill signal for listener

        # print(f"listener got result {m}")


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(**kwargs):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """


    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    data = {}
    for k, v in kwargs.items():

        if type(v) is onp.ndarray:
            # Non-scalar features need to be converted into binary-strings using tf.io.serialize_tensor function.
            # To convert the binary-string back to tensor, use tf.io.parse_tensor function:
            b = tf.io.serialize_tensor(v)
            serial = _bytes_feature(b)
        elif type(v) in {float}:
            serial = _float_feature(v)
        elif type(v) in {int, bool}:
            serial = _int64_feature(v)
        else:
            raise KeyError(f"`{type(v)}` is an invalid type for tfrecord conversion")

        data[k] = serial

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=data))
    return example_proto.SerializeToString()


def deserialize_example(serial):
    """
    Takes a tf.train.Example message and returns dict of numpy arrays
    """

    example = tf.train.Example.FromString(serial.numpy())
    result = {}

    # example.features.feature is the dictionary
    for key, feature in example.features.feature.items():

        # The values are the Feature objects which contain a `kind` which contains:
        # one of three fields: bytes_list, float_list, int64_list
        kind = feature.WhichOneof('kind')
        processed = getattr(feature, kind).value

        if kind == "bytes_list":
            # convert byte string back to tensor
            # assumes every byte string is a tensor (and not a simple string)
            result[key] = tf.io.parse_tensor(tf.constant(processed[0]), out_type=onp.float64).numpy()
        else:
            result[key] = onp.array(processed)

    return result


def sample_data(c):
    rng = onp.random.default_rng(onp.random.SeedSequence(entropy=0, spawn_key=(0, int(c))))
    time.sleep(1)
    time.sleep(0.1)
    out = (c+1) * (1.0 + onp.arange(3).astype(onp.float))
    out = onp.hstack([out, rng.normal(size=(2,))])
    # out = {"g": out, "c": c}
    out = {"g": out}
    return out


def write_listener(q, *, path, update_freq=30.0):
    """Listens for results on the q and writes it to TFRecord file at path
    Restarts writer every `update_freq` secs to update data set in memory for reading
    """
    t_loop = time.time()
    with tf.io.TFRecordWriter(path) as writer:
        while True:
            m = q.get()
            if m == 'kill':
                break  # kill signal for listener

            # write the `tf.train.Example` observations to the file.
            serialized = serialize_example(**m)
            writer.write(serialized)

            # flush writer
            if time.time() > t_loop + update_freq:
                writer.flush()
                t_loop = time.time()


if __name__ == '__main__':

    DIR = PROJECT_DIR
    DATA_DIR = os.path.join(DIR, DATA_SUBDIR, "online")
    os.makedirs(DATA_DIR, exist_ok=True)

    onp.set_printoptions(suppress=True, precision=4)

    filepath = os.path.join(DATA_DIR, f"example_df{get_datetime()}")
    print("path ", filepath)


    # write data
    n_cpus = 4
    N = 40
    ds_seeds = range(N)
    print("initialized dataset", flush=True)


    listen = functools.partial(write_listener, path=filepath, update_freq=1.0)
    # listen = example_listener

    executor = AsyncExecutor(n_workers=n_cpus, listener=listen)
    executor.launch(sample_data, ds_seeds)


    # read during the process of filling the data set
    while not executor.ready():
        time.sleep(1)

        # read
        try:
            raw_dataset = tf.data.TFRecordDataset([filepath])
            map_types = dict(g=onp.float32)
            ds = raw_dataset.map(lambda *args: structured_py_function(func=deserialize_example, inp=[*args], Tout=map_types))
            ds = ds.shuffle(100)
            ds = ds.batch(2, drop_remainder=True)

            # collect all gs
            gs_rec = []
            for ex in ds:
                gs_rec.append(ex["g"])

            if len(gs_rec) == 0:
                print("gs_rec empty ")
                continue

            gs_rec = onp.stack(gs_rec)
            print(gs_rec)
            print(gs_rec.shape, flush=True)

        except tf.errors.NotFoundError:
        # except:
            print("No example done yet")
            continue

    executor.finish()

    print("finished")

