import functools
import inspect
import warnings
from pathlib import Path

import numpy as onp

import jax
import jax.numpy as jnp
from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils


from avici.model import BaseModel, InferenceModel
from avici.utils.load import load_checkpoint
from avici.utils.data_jax import jax_standardize_default_simple, jax_standardize_count_simple
from avici.definitions import CACHE_SUBDIR, CHECKPOINT_KWARGS, HUGGINGFACE_REPO, \
    MODEL_NEURIPS_LINEAR, MODEL_NEURIPS_RFF, MODEL_NEURIPS_GRN, MODEL_NEURIPS_SCM_V0

from huggingface_hub import hf_hub_download


class AVICIModel:
    """
    Wrapper class for `avici.model.InferenceModel` that allows downloading and storing parameters
    and making predictions while exposing less functionality of the inference model.

    Args:
        params (pytree): parameters for the infernece model, extracted from `avici.backbone.ModelState.params`
        model (avici.model.InferenceModel`)
        expects_counts (bool): whether or not model expects counts. Since we pass the standardizer manually,
            this is technically not needed, but it is used to issue a warning in case the model expects counts
            but gets non-integer data (in case, for example, the data is accidentally standardized before `__call__`.
        standardizer: function standardizing the data prior to the forward pass.
            Default is `jax_standardize_default_simple` from `avici.utils.data_jax`, which performs the usual
            z-standardization for real-valued data.

    """

    def __init__(self, *,
                 params,
                 model,
                 expects_counts,
                 standardizer=jax_standardize_default_simple,
                 ):

        self.params = params
        self.expects_counts = expects_counts
        self._model = model
        self._standardizer = standardizer


    @functools.partial(jax.jit, static_argnums=(0, 3))
    def __call_main__(self, x, interv=None, experimental_chunk_size=None):
        # concatenate intervention indicators
        if interv is None:
            x = jnp.stack([x, jnp.zeros_like(x)], axis=-1)
        else:
            assert x.shape == interv.shape
            x = jnp.stack([x, interv.astype(x.dtype)], axis=-1)

        # standardize data
        x = self._standardizer(x)

        # forward pass through inference model
        g_edges_prob = self._model.infer_edge_probs(self.params, x, experimental_chunk_size=experimental_chunk_size)
        return g_edges_prob


    def __call__(self,
                 x,
                 interv=None,
                 return_probs=True,
                 devices=None,
                 shard_if_possible=True,
                 experimental_chunk_size=None,
                 ):
        """
        Wraps __call_main__ to do some tests and warnings before jax.jit

        Args:
            x: `[n, d]` Real-valued data matrix
            interv(optional): Binary matrix of the same shape as `x`
                with `interv[i,j] == 1` iff node `j` was intervened upon in observation `i`
            return_probs(optional): Whether to return probability estimates for each edge. Defaults to `True` as
                the computational cost is the same. `False` simply clips the predictions to 0 and 1 using a decision
                threshold of 0.5. Other thresholds achieve a different true positive vs
                false positive trade-off and can be computed using the probabilities returned with `True`.
            devices: String definining the backend to use for computation (e.g., "cpu", "gpu", "tpu") with all available
                devices, or list of explicit JAX devices to use for computation. If `None`, uses all available devices
                of the default JAX backend.
            shard_if_possible: Whether to shard the computation across the observations axis (`n`) of the input when
                multiple devices are available. This may improve the memory footprint on device. Discards the last
                 `n mod len(devices)` observations to allow sharding observations equally across devices.
                If `False`, does not shard the data and places input and computation on the first device in `devices`.
                Defaults to `True`.
            experimental_chunk_size (int, optional): [Experimental] If 0 < `experimental_chunk_size` < `n`,
                processes observation rows in chunks of size `experimental_chunk_size` until the max-pooling operation
                to save memory. This changes the output, because attention is no longer applied over the all
                observations jointly. `experimental_chunk_size` should probably be set as high as possible given the
                available memory. If `experimental_chunk_size` is `None` or falls outside of this range,
                attention is applied over the full observations axis. Defaults to `None`.
        Returns:
            `[d, d]` adjacency matrix of predicted edge probabilities
        """
        x_type = type(x)
        assert x.ndim == 2, "`x` must be a 2D array of shape [n, d]."

        # check that interv mask is binary
        if interv is not None:
            assert x.shape == interv.shape, "`x` and `interv` must have the same shape when provided."
            assert interv.ndim == 2, "`interv` must be a 2D array of shape [n, d] when provided."
            interv_is_binary = onp.all(onp.isclose(interv, 0) | onp.isclose(interv, 1))
            assert interv_is_binary, "Intervention mask `interv` has to be binary."
            interv = onp.around(interv, 0, out=interv)

        # check that x contains integers if model expects counts
        if self.expects_counts:
            x_is_int = onp.allclose(onp.mod(x, 1), 0)
            if not x_is_int:
                warnings.warn(f"Model expects count data but `x` contains non-integer values. ")

        # convert input to JAX and commit to device
        if type(devices) == str:
            devices = jax.local_devices(backend=devices)
        elif devices is None:
            devices = jax.local_devices()

        if shard_if_possible:
            device_count = len(devices)
            assert not x.shape[0] % device_count, ("observations `n` must be divisible by `device_count` if sharding "
                                                   "is enabled. ")
            mesh = mesh_utils.create_device_mesh((device_count,), devices)
            sharding = PositionalSharding(mesh)
            sh = sharding.reshape((device_count, 1))

            x = jax.device_put(x, sh)
            if interv is not None:
                interv = jax.device_put(interv, sh)
        else:
            x = jax.device_put(x, devices[0])
            if interv is not None:
                interv = jax.device_put(interv, devices[0])

        # main inference call
        out = self.__call_main__(x=x, interv=interv, experimental_chunk_size=experimental_chunk_size)

        # if desired, threshold output
        if not return_probs:
            out = (out > 0.5).astype(int)

        # convert to input type
        if x_type == onp.ndarray:
            out = onp.array(out)

        return out


def load_pretrained(download=None, force_download=False, checkpoint_dir=None, cache_path="./", expects_counts=None,
                    verbose=True):
    f"""
    Loads a pretrained AVICI model

    Args:
        download (str, optional): Specifier for existing pretrained model checkpoint to be downloaded (online)
        force_download (bool, optional): Whether to force re-download of model checkpoint specified via `download`
        checkpoint_dir (str, optional): Path to *folder* containing both the model checkpoint `<name>.pkl` and
            the model kwargs `{CHECKPOINT_KWARGS}` files (stored locally).
        cache_path (str, optional): Path used as cache directory for storing the
            downloaded model checkpoints (Default: `./cache`)
        expects_counts (bool, optional): Whether model expects count data, for data standardization purposes.
            Required when providing `checkpoint_dir`.
        verbose (bool, optional): Whether to print path information

    Returns:
        `avici.AVICIModel`
    """
    assert (download is None and checkpoint_dir is not None) \
        or (download is not None and checkpoint_dir is None), "Specify exactly one of `download` or `checkpoint_dir`"

    # get figshare source link and select standardization function (bind heldout_data as `None`)
    if download is not None:
        if download == "neurips-linear":
            paths = MODEL_NEURIPS_LINEAR
            expects_counts = False
        elif download == "neurips-rff":
            paths = MODEL_NEURIPS_RFF
            expects_counts = False
        elif download == "neurips-grn":
            paths = MODEL_NEURIPS_GRN
            expects_counts = True
        elif download == "scm-v0":
            paths = MODEL_NEURIPS_SCM_V0
            expects_counts = False
        else:
            raise ValueError(f"Unknown download specified: `{download}`")

        # download source if not yet downloaded
        if cache_path == "./":
            try:
                cache_path = Path(cache_path).resolve() / CACHE_SUBDIR
            except FileNotFoundError:
                raise FileNotFoundError("Could not resolve default cache_path `./` for downloads. "
                                        "Please specify the download location by specifying `cache_path`")
            if verbose:
                print(f"Using default cache_path: `{cache_path}`")

        # download from huggingface
        model_paths = []
        for path in paths:
            subfolder, filename = Path(path).parent, Path(path).name
            file_path = hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                subfolder=subfolder,
                filename=filename,
                cache_dir=cache_path,
                force_download=force_download,
            )
            model_paths.append(Path(file_path))

        assert all([p.parent == model_paths[0].parent for p in model_paths]), \
            f"Folder structure error. All files should be from the same model folder. "

        root_path = model_paths[0].parent

    else:
        assert expects_counts is not None, "When loading custom model checkpoint, need to specify `expects_counts` as "\
                                           "`True`/`False` to ensure proper data standardization. Specifying " \
                                           "`False` should be the default and implies the data is z-standardized."
        root_path = Path(checkpoint_dir)
        assert root_path.exists(), f"The provided checkpoint_dir `{checkpoint_dir}` does not exist. " \
                                    f"In case you provided a relative path, try the absolute path instead."
        assert root_path.is_dir(), "The provided checkpoint_dir is not a directory. You should specify the path to " \
                                    f"a folder that contains the model `.pkl` (and `{CHECKPOINT_KWARGS}`), " \
                                    "not the path to the `.pkl` alone."
        assert (root_path.parent / CHECKPOINT_KWARGS).exists(), f"The provided checkpoint_dir `{checkpoint_dir}` " \
                                    f"does not contain the file `{CHECKPOINT_KWARGS}`, which is required. "
        print(f"Loading local checkpoint from `{root_path}`")

    # select data standardizer
    if expects_counts:
        standardizer = jax_standardize_count_simple
    else:
        standardizer = jax_standardize_default_simple

    # load checkpoint
    state, loaded_config = load_checkpoint(root_path)
    inference_model_kwargs = loaded_config["inference_model_kwargs"]
    try:
        neural_net_kwargs = loaded_config["neural_net_kwargs"]["model_kwargs"] # legacy compatibility
    except KeyError:
        neural_net_kwargs = loaded_config["neural_net_kwargs"]

    # discard deprecated kwargs from checkpoint
    sig = inspect.signature(InferenceModel).parameters
    deprec = list(filter(lambda key: key not in sig, inference_model_kwargs.keys()))
    for k in deprec:
        del inference_model_kwargs[k]
        # print(f"Ignoring deprecated kwarg `{k}` loaded from `inference_model_kwargs` in checkpoint")

    # init model
    inference_model = InferenceModel(**inference_model_kwargs,
                                     model_class=BaseModel,
                                     model_kwargs=neural_net_kwargs)

    model = AVICIModel(
        params=state.params,
        model=inference_model,
        expects_counts=expects_counts,
        standardizer=standardizer,
    )

    return model




