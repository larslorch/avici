from avici.synthetic.noise_scale import init_noise_dist
from avici.synthetic.utils import sample_recursive_scm
from avici.synthetic import MechanismModel


class LinearAdditive(MechanismModel):
    """
    Linear mechanism with additive noise

    Args:
        param (Distribution): distribution for sampling linear function weights.
            Example: `avici.synthetic.SignedUniform` (for bounding away from zero)
        bias (Distribution): distribution for sampling linear function biases. Example: `avici.synthetic.Uniform`
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
                 param,
                 bias,
                 noise,
                 noise_scale=None,
                 noise_scale_heteroscedastic=None,
                 n_interv_vars=0,
                 interv_dist=None):

        assert interv_dist is not None or n_interv_vars == 0

        self.param = param
        self.bias = bias
        self.noise = noise
        self.noise_scale = noise_scale
        self.noise_scale_heteroscedastic = noise_scale_heteroscedastic
        self.n_interv_vars = n_interv_vars
        self.interv_dist = interv_dist

    def __call__(self, rng, g, n_observations_obs, n_observations_int):

        assert self.interv_dist is not None or self.n_interv_vars == 0

        # construct mechanism for each node
        n_vars = g.shape[-1]
        f = []
        for j in range(n_vars):
            # sample parameters
            # each call may use random hyperparameters, so call once per node
            w = self.param(rng, shape=(n_vars,))
            b = self.bias(rng, shape=(1,))

            # bind parameters to mechanism function
            f.append(lambda x, is_parent, z, theta=w, bias=b: (x @ (theta * is_parent)) + bias + z)

        # construct noise distribution for each node
        nse = []
        for j in range(n_vars):
            # sample parameters and bind to sampling function
            # each call may sample a random noise scale, so call once per node
            nse.append(init_noise_dist(rng=rng,
                                       dim=int(g[:, j].sum().item()),
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