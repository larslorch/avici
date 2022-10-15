import functools
import numpy as onp

from avici.synthetic.recursive import sample_recursive

# import all sampling functions into mechanism.py for parser to find them
# do not remove even if PyCharm greys out!
from avici.sergio.sergio_sampler import (
    grn_sergio,
    sergio_clean, sergio_clean_count, sergio_noisy, sergio_noisy_count,
    kosergio_clean, kosergio_clean_count, kosergio_noisy, kosergio_noisy_count,
)


def draw_rff(*, rng, d, length_scale, output_scale, n_rff):
    """Draws random instantiation of rffs"""
    # draw parameters
    ls = length_scale(rng, shape=(1,)).item() if callable(length_scale) else length_scale
    c = output_scale(rng, shape=(1,)).item() if callable(output_scale) else output_scale

    # draw rffs
    # [d, n_rff]
    omega_j = rng.normal(loc=0, scale=1.0 / ls, size=(d, n_rff))

    # [n_rff, ]
    b_j = rng.uniform(0, 2 * onp.pi, size=(n_rff,))

    # [n_rff, ]
    w_j = rng.normal(loc=0, scale=1.0, size=(n_rff,))

    return dict(
        c=c,
        omega=omega_j,
        b=b_j,
        w=w_j,
        n_rff=n_rff,
    )


"""
Noise functions
"""

def _const_scale(*, rng, x, is_parent, dist, scale):
    n_obs = x.shape[0]
    return scale * dist(rng, shape=(n_obs,))

def _heteroscedastic_rff_scale(*, rng, x, is_parent, dist, c, omega, b, w, n_rff):
    # compute rff function f(x) = w'phi(x)
    x_parents = x[:, onp.where(is_parent)[0]]
    phi = onp.cos(onp.einsum('db,...d->...b', omega, x_parents) + b)
    f_x = onp.sqrt(2.0) * c * onp.einsum('b,...b->...', w, phi) / onp.sqrt(n_rff)

    # sample noise with scale^2 = log(1+exp(f(x))
    scale = onp.sqrt(onp.log(1.0 + onp.exp(f_x)))
    return scale * dist(rng, shape=scale.shape)


"""
Mechansim functions
"""

def _linear_additive_mechanism(*, rng, x, z_j, is_parent, is_intervened, interv_dist, theta_j, bias_j):
    """
    Args:
        x:              [n_obs, n_vars] currently sampled data (filled columns for ancestors of j)
        z_j:            [n_obs,] endogenous noise of j for each observation
        is_parent:      [n_vars,] indicator for parents of node
        is_intervened:  bool whether node is intervened upon

        theta_j         [n_vars]
        bias_j          [1]

    Returns:
        x_j:            [n_obs,] observations for node j
    """
    if is_intervened:
        x_j = interv_dist(rng, shape=z_j.shape)
    else:
        x_j = (x @ (theta_j * is_parent)) + bias_j + z_j

    return x_j


def linear_additive(*, spec, rng, g, effect_sgn, toporder, n_vars,
                    # specific inputs required in config
                    param,
                    bias,
                    noise,
                    noise_scale=None,
                    noise_scale_heteroscedastic=None,
                    # interventions
                    n_interv_vars=0,
                    interv_dist=None,
                    ):
    """
    Linear mechanism with additive noise
    """

    # construct mechanism for each node
    f = []
    for j in range(n_vars):
        # sample parameters
        # each call may sample a random noise scale, so call once per node
        theta_j = param(rng, shape=(n_vars,))
        bias_j = bias(rng, shape=(1,))

        # force sign of effect if we are given that information
        if effect_sgn is not None:
            assert set(onp.unique(effect_sgn)).issubset({-1.0, 0.0, 1.0})
            theta_j = onp.abs(theta_j) * effect_sgn[:, j].astype(onp.float32)

        # bind parameters to mechanism function
        f.append(functools.partial(_linear_additive_mechanism, theta_j=theta_j, bias_j=bias_j, interv_dist=interv_dist))

    # construct noise distribution for each node
    assert noise_scale is None or noise_scale_heteroscedastic is None
    assert noise_scale is not None or noise_scale_heteroscedastic is not None
    nse = []
    for j in range(n_vars):
        if noise_scale is not None:
            # bind constant noise scale to noise function
            nse.append(functools.partial(_const_scale, dist=noise, scale=noise_scale(rng)))

        elif noise_scale_heteroscedastic is not None:
            # draw rff function parameters that model heteroscedastic noise scale
            assert "rff" in  noise_scale_heteroscedastic
            d = int(g[:, j].sum().item())
            nse_params = draw_rff(rng=rng, d=d, length_scale=noise_scale_heteroscedastic["length_scale"],
                                  output_scale=noise_scale_heteroscedastic["output_scale"], n_rff=100)

            # bind these parameters to noise function
            nse.append(functools.partial(_heteroscedastic_rff_scale, dist=noise, **nse_params))

        else:
            raise KeyError("neither `noise_scale` nor `noise_scale_heterosc` are given")

    # sample recursively over g given functionals and endogenous noise distribution
    data = sample_recursive(
        spec=spec,
        rng=rng,
        g=g,
        toporder=toporder,
        n_vars=n_vars,
        f=f,
        nse=nse,
        is_count_data=False,
        n_interv_vars=n_interv_vars,
        interv_dist=interv_dist
    )
    return data


def _rff_mechanism(*, rng, x, z_j, is_parent, is_intervened, interv_dist, bias_j, omega, b, w, c, n_rff):
    """
    Args:
        rng
        x:              [n_obs, n_vars] currently sampled data (filled columns for ancestors of j)
        z_j:            [n_obs,] endogenous noise of j for each observation
        is_parent:      [n_vars,] indicator for parents of node
        is_intervened:  bool whether node is intervened upon

        bias_j
        omega
        b
        w
        c
        n_rff

    Returns:
        x_j:            [n_obs,] observations for node j
    """

    if is_intervened:
        x_j = interv_dist(rng, shape=z_j.shape)
    else:
        # [..., n_parents]
        x_parents = x[:, onp.where(is_parent)[0]]

        # feature map phi = cos(omega'x + b)
        # [..., n_rff]
        phi = onp.cos(onp.einsum('db,...d->...b', omega, x_parents) + b)

        # f(x) = w'phi(x)
        # [...]
        f_j = onp.sqrt(2.0) * c * onp.einsum('b,...b->...', w, phi) / onp.sqrt(n_rff)

        # additive noise
        x_j = f_j + bias_j + z_j

    return x_j


def rff_additive(*, spec, rng, g, effect_sgn, toporder, n_vars,
                 # specific inputs required in config
                 length_scale,
                 output_scale,
                 bias,
                 noise,
                 noise_scale=None,
                 noise_scale_heteroscedastic=None,
                 # interventions
                 n_interv_vars=0,
                 interv_dist=None,
                 ):
    """
    Random fourier feature functions with additive noise
    Corresponds to samples from GP prior with RBF kernel

    https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

    As n_rff goes to infinity, function returned corresponds to a sample from GP prior with
    RBF kernel

            k(x, y) = c^2 * exp(- |x-y|^2 / (2 * ls^2)

    """

    # construct mechanism for each node
    f = []
    for j in range(n_vars):
        # draw rff mechanism parameters
        d = int(g[:, j].sum().item())
        f_params = draw_rff(rng=rng, d=d, length_scale=length_scale, output_scale=output_scale, n_rff=100)
        bias_j = bias(rng, shape=(1,))

        # bind these parameters to mechanism function
        f.append(functools.partial(_rff_mechanism, **f_params, bias_j=bias_j, interv_dist=interv_dist))

    # construct noise distribution for each node
    assert noise_scale is None or noise_scale_heteroscedastic is None
    assert noise_scale is not None or noise_scale_heteroscedastic is not None
    nse = []
    for j in range(n_vars):
        if noise_scale is not None:
            # bind constant noise scale to noise function
            nse.append(functools.partial(_const_scale, dist=noise, scale=noise_scale(rng)))

        elif noise_scale_heteroscedastic is not None:
            # draw rff function parameters that model heteroscedastic noise scale
            assert "rff" in noise_scale_heteroscedastic
            d = int(g[:, j].sum().item())
            nse_params = draw_rff(rng=rng, d=d, length_scale=noise_scale_heteroscedastic["length_scale"],
                                  output_scale=noise_scale_heteroscedastic["output_scale"], n_rff=100)

            # bind these parameters to noise function
            nse.append(functools.partial(_heteroscedastic_rff_scale, dist=noise, **nse_params))

        else:
            raise KeyError("neither `noise_scale` nor `noise_scale_heterosc` are given")

    # sample recursively over g given functionals and endogenous noise distribution
    data = sample_recursive(
        spec=spec,
        rng=rng,
        g=g,
        toporder=toporder,
        n_vars=n_vars,
        f=f,
        nse=nse,
        is_count_data=False,
        n_interv_vars=n_interv_vars,
        interv_dist=interv_dist
    )
    return data


if __name__ == "__main__":

    from avici.utils.parse import load_data_config

    # test_spec = load_data_config("config/linear_additive-0.yaml")["data"]["train"][0]
    # test_spec = load_data_config("config/linear_heteroscedastic-0.yaml")["data"]["train"][0]
    # test_spec = load_data_config("config/rff_additive-0.yaml")["data"]["train"][0]
    # test_spec = load_data_config("config/rff_heteroscedastic-0.yaml")["data"]["train"][0]
    # test_spec = load_data_config("config/sergio-0.yaml")["data"]["train"][0]

    # test_spec = load_data_config("experiments/linear/train.yaml")["data"]["train"][0]
    # test_spec = load_data_config("experiments/rff/train.yaml")["data"]["train"][0]

    test_spec = load_data_config("experiments/linear-base/train.yaml")["data"]["train"][0]
    # test_spec = load_data_config("experiments/rff-base/train.yaml")["data"]["train"][0]


    testnvars = 10

    testrng = onp.random.default_rng(0)
    testg, testeffect_sgn, testtoporder = test_spec.g(testrng, testnvars)
    testdata = test_spec.mechanism(spec=test_spec, rng=testrng, g=testg, effect_sgn=testeffect_sgn, toporder=testtoporder, n_vars=testnvars)

    print(testdata)
