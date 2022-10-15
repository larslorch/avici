import jax.random as random
import copy
from collections import defaultdict
from avici.utils.dataset import make_dataset, make_linear_gaussian_dataset_generator


def _make_data_identifier(*, strg, n_vars, n_obs, seed, key):
    return f"-{strg}-d={n_vars}-n={n_obs}-seed={seed}-key={key.sum().item()}"


def make_data(*, key, seed, path, batch_size_effective=1,
              n_train_sets=500000, n_observations_train=100, n_vars_train=20,
              n_test_sets=500, n_observations_test=100, n_vars_test=None):

    if n_vars_test is None:
        n_vars_test = [5, 10, 20, 30, 50, 100]

    key, key_data_train, key_data_test = random.split(key, 3)

    data = defaultdict(lambda: defaultdict(dict))

    """train"""
    ds_train_cache_path = path + _make_data_identifier(
        strg="train", n_vars=n_vars_train, n_obs=n_observations_train, seed=seed, key=key_data_train)

    x_train_generator, x_train_generator_types = make_linear_gaussian_dataset_generator(
        key=key_data_train, n_vars=n_vars_train, n_observations=n_observations_train,
        epoch_size=n_train_sets)

    data["train"]["loop"] = dict(
        generator=copy.deepcopy(x_train_generator), generator_types=x_train_generator_types,
        cache_filename=ds_train_cache_path, batch_size=batch_size_effective) # training batch size
    data["train"]["single"] = dict(
        generator=copy.deepcopy(x_train_generator), generator_types=x_train_generator_types,
        cache_filename=ds_train_cache_path, batch_size=1, repeat=1) # same as loop but batch_size 1 for convenience

    """test"""
    for d in n_vars_test:
        ds_test_cache_path = path + _make_data_identifier(
            strg="test", n_vars=d, n_obs=n_observations_test, seed=seed, key=key_data_test)

        x_test_generator, x_test_generator_types = make_linear_gaussian_dataset_generator(
            key=key_data_test, n_vars=d, n_observations=n_observations_test,
            epoch_size=n_test_sets)

        data["test"][d] = dict(generator=x_test_generator, generator_types=x_test_generator_types,
            cache_filename=ds_test_cache_path, batch_size=1, repeat=1) # no batch to avoid OOM for large d

    return data