import zipfile
import functools
import inspect
import shutil
import warnings
from tqdm.auto import tqdm
from pathlib import Path

import jax
import numpy as onp
import jax.numpy as jnp

from avici.model import BaseModel, InferenceModel
from avici.utils.load import load_checkpoint
from avici.utils.data_jax import jax_standardize_default_simple, jax_standardize_count_simple
from avici.definitions import CACHE_SUBDIR, CHECKPOINT_KWARGS, HUGGINGFACE_REPO, \
    MODEL_NEURIPS_LINEAR, MODEL_NEURIPS_RFF, MODEL_NEURIPS_GRN

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


    @functools.partial(jax.jit, static_argnums=(0,))
    def __call_main__(self, x, interv=None):
        # concatenate intervention indicators
        if interv is None:
            x = jnp.stack([x, jnp.zeros_like(x)], axis=-1)
        else:
            assert x.shape == interv.shape
            x = jnp.stack([x, interv.astype(x.dtype)], axis=-1)

        # standardize data
        x = self._standardizer(x)

        # forward pass through inference model
        g_edges_prob = self._model.infer_edge_probs(self.params, x)
        return g_edges_prob


    def __call__(self, x, interv=None, return_probs=True):
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

        Returns:
            `[d, d]` adjacency matrix of predicted edge probabilities
        """
        x_type = type(x)

        # check that interv mask is binary
        if interv is not None:
            assert x.shape == interv.shape, "`x` and `interv` must have the same shape when provided."
            interv_is_binary = onp.all(onp.isclose(interv, 0) | onp.isclose(interv, 1))
            assert interv_is_binary, "Intervention mask `interv` has to be binary."
            interv = onp.around(interv, 0, out=interv)

        # check that x contains integers if model expects counts
        if self.expects_counts:
            x_is_int = onp.allclose(onp.mod(x, 1), 0)
            if not x_is_int:
                warnings.warn(f"Model expects count data but `x` contains non-integer values. ")

        # main call
        out = self.__call_main__(x=x, interv=interv)

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




