# Amortized Inference for Causal Structure Learning

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
[![PyPi](https://img.shields.io/pypi/v/avici?logo=PyPI)](https://pypi.org/project/avici/)

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
g, x = simulate_data(n=100, d=20, domain="linear-gaussian-scm")

# load pretrained model
model = avici.load_pretrained(download="linear")

# g_prob: [d, d] predicted edge probabilities of the causal graph
g_prob = model(x=x)
```
You can run a working example this snippet directly in the following Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/larslorch/avici/blob/master/example-pretrained/example.ipynb)

The above code automatically downloads and initializes 
a pretrained model checkpoint
(~60MB) of the requested domain and predicts
the causal structure of the simulated data.
Based on our paper, we provide the pretrained weights for models trained on
three data-generating processes, which are specified by the `download` argument:

- `linear`: SCM data with linear causal mechanisms
- `nonlinear`: SCM data with nonlinear causal mechanisms drawn 
from GPs with squared-exponential kernel
(defined via random Fourier features)
- `sergio`: Synthetic scRNA-seq gene expression data using the SERGIO
[simulator](https://github.com/PayamDiba/SERGIO) by 
[Dibaeinia and Sinha, (2020)](https://www.cell.com/cell-systems/pdf/S2405-4712(20)30287-8.pdf)

For details on the exact training distributions of these models,
please refer to Appendix A of 
[Lorch et al., (2022)](https://arxiv.org/abs/2205.12934). 
The corresponding YAML domain configuration files are available [here](avici/config/train/).


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



## Quick Start: Custom Data-Generating Processes

In the **[example-custom](example-custom)** folder, 
we provide an extended README together with a corresponding implementation
that illustrates a detailed example of how to train an AVICI model
for a custom data-generating process.

In short, we provide the following three components for training a full model:

1. `func.py`: **(Optional) Python file defining custom data-generating processes**

    If you would like to train on data-generating processes not already provided by `avici.synthetic`,
    this file implements subclasses of `GraphModel` and `MechanismModel` doing so.  

2. `domain.yaml`: **YAML file defining the training data distribution**

    This configuration file specifies the full distribution over datasets used for training.
    Several graph models and data-generating mechanisms are available out-of-the-box, so providing
    additional modules via `func.py` is optional.

4. `train.py`: **Python training script**

    Fully-fledged training script for multi-device training (if available) based on the above configurations. 

The checkpoints created using the training script can directly be loaded by the `avici.load_pretrained`
function from above:
```python
import avici
model = avici.load_pretrained(checkpoint_dir="path/to/checkpoint", expects_counts=False)
```


## Custom Installation and Branches (Apple Silicon)

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
#### Note to Apple Silicon/M1 chip users:
Installing the package by first setting up a conda environment
using our conda `environment.yaml` config and then installing
`pip install -r requirements.txt` before finally running
`pip install -e .` works on Apple M1 MacBooks.
Directly installing `avici` via PyPI may install incompatible versions 
or builds of package requirements, which may cause unexpected, low-level errors.

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