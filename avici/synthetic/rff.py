import functools
import numpy as onp

from avici.synthetic.noise_scale import init_noise_dist
from avici.synthetic.utils import draw_rff_params, sample_recursive_scm
from avici.synthetic import MechanismModel


class RFFAdditive(MechanismModel):
    """
    Random fourier feature functions with additive noise

    As n_rff goes to infinity, function returned corresponds to a sample from GP prior with RBF kernel
        k(x, y) = c^2 * exp(- |x-y|^2 / (2 * ls^2)

    Rahimi and Recht, 2007
    https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

    Args:
        length_scale (Distribution): distribution for sampling kernel length scale used to sample a GP function.
            Example: `avici.synthetic.Uniform`
        output_scale (Distribution): distribution for sampling kernel output scale used to sample a GP function.
            Example: `avici.synthetic.Uniform`
        bias (Distribution): distribution for sampling GP function biases. Example: `avici.synthetic.Uniform`
        noise (Distribution): distribution for sampling additive noise. Example: `avici.synthetic.Gaussian`
        noise_scale (Distribution): if specified, distribution used to sample noise scale
            for `avici.synthetic.SimpleNoise`. Example: `avici.synthetic.Uniform`.
            Only one of `noise_scale` and `noise_scale_heteroscedastic` must be specified.
        noise_scale_heteroscedastic (Distribution): if specified, kwargs for
            initalizing `avici.synthetic.HeteroscedasticRFFNoise`.
            Only one of `noise_scale` and `noise_scale_heteroscedastic` must be specified.
        n_interv_vars (int): no. unique variables intervened upoin in all of data collected;
            -1 indicates all nodes
        interv_dist (Distribution): distribution for sampling intervention values.
            Example: `avici.synthetic.SignedUniform`

    """
    def __init__(self,
                length_scale,
                output_scale,
                bias,
                noise,
                noise_scale=None,
                noise_scale_heteroscedastic=None,
                n_interv_vars=0,
                interv_dist=None):

       assert interv_dist is not None or n_interv_vars == 0

       self.length_scale = length_scale
       self.output_scale = output_scale
       self.bias = bias
       self.noise = noise
       self.noise_scale = noise_scale
       self.noise_scale_heteroscedastic = noise_scale_heteroscedastic
       self.n_interv_vars = n_interv_vars
       self.interv_dist = interv_dist

    @staticmethod
    def _rff_mechanism(*, x, z, is_parent, bias, omega, b, w, c, n_rff):
        # [..., n_parents]
        x_parents = x[:, onp.where(is_parent)[0]]

        # feature map phi = cos(omega @ x + b)
        # [..., n_parents, n_rff], [..., n_parents] -> [..., n_rff]
        phi = onp.cos(onp.einsum('db,...d->...b', omega, x_parents) + b)

        # f(x) = w @ phi(x)
        # [..., n_rff], [..., n_rff] -> [...]
        f_j = onp.sqrt(2.0) * c * onp.einsum('b,...b->...', w, phi) / onp.sqrt(n_rff)

        # additive noise
        x_j = f_j + bias + z
        return x_j


    def __call__(self, rng, g, n_observations_obs, n_observations_int):

        # construct mechanism for each node
        n_vars = g.shape[-1]
        f = []
        for j in range(n_vars):
            # draw rff mechanism parameters for node j
            n_parents = int(g[:, j].sum().item())
            f_params = draw_rff_params(rng=rng, d=n_parents, length_scale=self.length_scale,
                                       output_scale=self.output_scale, n_rff=100)
            b = self.bias(rng, shape=(1,))

            # bind these parameters to mechanism function
            f.append(functools.partial(RFFAdditive._rff_mechanism, **f_params, bias=b))

        # construct noise distribution for each node
        nse = []
        for j in range(n_vars):
            # sample parameters and bind to sampling function
            # each call may sample a random noise scale, so call once per node
            n_parents = int(g[:, j].sum().item())
            nse.append(init_noise_dist(rng=rng,
                                       dim=n_parents,
                                       dist=self.noise,
                                       noise_scale=self.noise_scale,
                                       noise_scale_heteroscedastic=self.noise_scale_heteroscedastic))

        # sample recursively over g given functionals and endogenous noise distribution
        data = sample_recursive_scm(
            rng=rng,
            n_observations_obs=n_observations_obs,
            n_observations_int=n_observations_int,
            g=g,
            f=f,
            nse=nse,
            interv_dist=self.interv_dist,
            n_interv_vars=self.n_interv_vars,
        )
        return data