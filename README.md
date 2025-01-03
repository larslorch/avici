# Amortized Inference for Causal Structure Learning

[![Downloads](https://static.pepy.tech/badge/avici)](https://pypi.org/project/avici/)
[![PyPi](https://img.shields.io/pypi/v/avici?logo=PyPI)](https://pypi.org/project/avici/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/larslorch/avici)

This is the code repository for the paper 
_Amortized Inference for Causal Structure Learning_
([Lorch et al., 2022](https://arxiv.org/abs/2205.12934), NeurIPS 2022).
Performing **a**mortized **v**ariational **i**nference for 
**c**ausal d**i**scovery (AVICI) allows inferring causal structure 
from data based on a  _simulator_ of the domain of interest.
By training a neural network to infer structure from the simulated 
data, it can acquire realistic inductive biases from prior knowledge
that is hard to cast as score functions or conditional 
independence tests.


To install the latest stable release, run:

```bash
pip install avici
````

The package allows training new models from scratch on custom data-generating processes 
and performing predictions with pretrained models from our side.
The codebase is written in Python and 
[JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).


## Quick Start: Pretrained Model

Using the `avici` package is as easy as running the following code
snippet:

```python
import avici
from avici import simulate_data

# g: [d, d] causal graph of `d` variables
# x: [n, d] data matrix containing `n` observations of the `d` variables
g, x, _ = simulate_data(d=50, n=200, domain="rff-gauss")

# load pretrained model
model = avici.load_pretrained(download="scm-v0")

# g_prob: [d, d] predicted edge probabilities of the causal graph
g_prob = model(x=x)
```
You can run a working example this snippet directly in the following Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/larslorch/avici/blob/master/example-pretrained/example.ipynb)

The above code automatically downloads and initializes 
a pretrained model checkpoint (~60MB) of the domain 
and predicts the causal structure underlying the simulated data.


We currently provide the following models checkpoints,
which can be specified by the `download` argument:

- `scm-v0`: linear and nonlinear SCM data, broad graph and noise distributions
- `neurips-linear`: SCM data with linear causal mechanisms
- `neurips-rff`: SCM data with nonlinear causal mechanisms drawn 
from GPs with squared-exponential kernel
(defined via random Fourier features)
- `neurips-grn`: Synthetic scRNA-seq gene expression data using the SERGIO
[simulator](https://github.com/PayamDiba/SERGIO) by 
[Dibaeinia and Sinha, (2020)](https://www.cell.com/cell-systems/pdf/S2405-4712(20)30287-8.pdf)

We recommend  the latest `scm-v0` for working with arbitrary real-valued data. 
This model was trained on SCM data simulated from a large variety of graph models with up to 100 nodes, 
both linear and nonlinear causal mechanisms, and homogeneous and heterogeneous additive noise from
Gaussian, Laplace, and Cauchy distributions.

The models `neurips-linear`, `neurips-rff`, `neurips-grn` studied in our original 
paper were purposely trained on narrower training distributions to assess the out-of-distribution 
capability of AVICI. Unless your prior domain knowledge is strong,
this may make the `neurips-*` models less suitable for benchmarking
or as general purpose/out-of-the-box tools in your application.
The training distribution of `scm-v0` essentially combines those of
`neurips-linear` and  `neurips-rff` as well as their out-of-distribution
settings in [Lorch et al., (2022)](https://arxiv.org/abs/2205.12934).


For details on the exact training distributions of these models,
please refer to the [model cards](https://huggingface.co/larslorch/avici) 
on HuggingFace. Appendix A of 
[Lorch et al., (2022)](https://arxiv.org/abs/2205.12934) also defines the training distributions
of the `neurips-*` models.
The YAML domain config file for each model is available in [`avici/config/train/`](avici/config/train/).


Calling `model` as obtained from `avici.load_pretrained`
predicts the `[d, d]` matrix of probabilities for each possible edge in the causal graph
and accepts the following arguments:

- **x** (ndarray) – Real-valued data matrix of shape `[n, d]`
- **interv** (ndarray, optional) – Binary matrix of the same shape as **x** 
        with **interv[i,j] = 1** iff node **j** was intervened upon in 
        observation **i**. (Default is `None`)  
- **return_probs** (bool, optional) –  Whether to return probability estimates 
        for each edge. `False` simply clips the predictions to 0 and 1 using 
        a decision threshold of 0.5. (Default is `True` as the computational 
        cost is the same.)

When sampling synthetic data via `avici.simulate_data`, 
the following domain specifiers (dataset distributions) 
are currently provided:
`lin-gauss`, 
`lin-gauss-heterosked`,
`lin-laplace-cauchy`, 
`rff-gauss`, 
`rff-gauss-heterosked`, 
`rff-laplace-cauchy`, 
`gene-ecoli`, 
but custom config files can be specified, too. 
All these domains are defined inside `avici.config.examples`.

## Quick Start: Custom Data-Generating Processes

In the **[example-custom](example-custom)** folder, 
we provide an extended README together with a corresponding implementation
that illustrates a detailed example of how to train an AVICI model
for a custom data-generating process.

In short, the following three components are needed for training a full model:

1. `func.py`: **(Optional) Python file defining custom data-generating processes**

    If you would like to train on data-generating processes not already provided by `avici.synthetic`,
    this file implements subclasses of `GraphModel` and `MechanismModel` doing so.  

2. `domain.yaml`: **YAML file defining the training data distribution**

    This configuration file specifies the full distribution over datasets used for training.
    Several graph models and data-generating mechanisms are available out-of-the-box, so providing
    additional modules via `func.py` is optional.
    This file can also be used to simulate data in `avici.simulate_data`.

4. `train.py`: **Python training script**

    Fully-fledged training script for multi-device training (if available) based on the above configurations. 

The checkpoints created using the training script can directly be loaded by the `avici.load_pretrained`
function from above:
```python
import avici
model = avici.load_pretrained(checkpoint_dir="path/to/checkpoint", expects_counts=False)
```


## Change Log

- **avici 1.1.0:** 
`experimental_chunk_size` flags in the forward pass more memory efficiency, which applies transformer blocks 
in chunks along the observations axis until the final max-pooling operation. This changes the output,
since attention is no longer applied jointly over all observation and is an experimental feature that 
was not properly evaluated. 
However, it can be useful for large datasets and performed better than staying at the memory limit 
in preliminary tests.

- **avici 1.0.7:** `devices` and `sharding_if_possible` flags in the forward pass of
[`BaseModel`](avici/model.py) 
for running the inference forward pass on a specific backend (e.g. CPU)
and automatically sharding computation on multiple devices.

- **avici 1.0.6:** Requirements and README updates. 

- **avici 1.0.3:** Published `scm-v0` model parameters.

- **avici 1.0.0:** Release

## Custom Installation and Branches

When using `avici` for your research and applications, we recommend using
the easy-to-use `main` branch and installing the latest stable
release using PyPI's `pip`
as explained above.

For custom installations, we recommend using `conda` and generating 
a new environment via
```
conda env create --file environment.yaml
```
You then need to install the `avici` package with
```
pip install -e .
```

#### Reproducibility branch
In addition to `main`, this repository also contains a `full` 
[branch](https://github.com/larslorch/avici/tree/full), 
which contains
comprehensive code for reproducing the the experimental results in 
[Lorch et al., (2022)](https://arxiv.org/abs/2205.12934). 
The purpose of `full` is reproducibility; the branch is not 
updated anymore and may contain outdated notation and documentation.


## Reference

```
@article{lorch2022amortized,
  title={Amortized Inference for Causal Structure Learning},
  author={Lorch, Lars and Sussex, Scott and Rothfuss, Jonas and Krause, Andreas and Sch{\"o}lkopf, Bernhard},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
