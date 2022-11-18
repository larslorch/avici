import numpy as onp
from avici.synthetic.utils import draw_rff_params
from avici.synthetic import NoiseModel


class SimpleNoise(NoiseModel):
    """
    Simple noise with constant scale (homogeneous noise)

    Args:
        dist (Distribution): distribution from which noise is sampled
        scale (float): scale by which noise is scaled (similar to standard deviation in Gaussian)

    """
    def __init__(self, dist, scale):
        self.dist = dist
        self.scale = scale

    def __call__(self, rng, x, is_parent):
        n_obs = x.shape[0]
        return self.scale * self.dist(rng, shape=(n_obs,))


class HeteroscedasticRFFNoise(NoiseModel):
    """
    Heterogenous noise based on a random Fourier feature scaling function h(.)
    The noise scale of `dist` is input-dependent an given by scale^2 = log(1 + exp(h(x))

    Args:
        dist (Distribution): distribution from which noise is sampled
        rng (np.random.Generator): numpy pseudorandom number generator
        d (int): input dimension (number of parents)
        length_scale (Distribution): distribution from which length scale of GP is sampled
        output_scale (Distribution): distribution from which length scale of GP is sampled
        n_rff (int, optional): number of features used to approximate GP function draw
    """

    def __init__(self, dist, *, rng, d, length_scale, output_scale, n_rff=100):
        self.dist = dist
        self.param = draw_rff_params(rng=rng, d=d, length_scale=length_scale,
                                     output_scale=output_scale, n_rff=n_rff)
        self.n_rff = n_rff

    def __call__(self, rng, x, is_parent):
        # compute rff function f(x) = w'phi(x)
        x_parents = x[:, onp.where(is_parent)[0]]
        phi = onp.cos(onp.einsum('db,...d->...b', self.param["omega"], x_parents) + self.param["b"])
        f_x = onp.sqrt(2.0) * self.param["c"] * onp.einsum('b,...b->...', self.param["w"], phi) / onp.sqrt(self.n_rff)

        # sample noise with scale^2 = log(1+exp(f(x))
        scale = onp.sqrt(onp.log(1.0 + onp.exp(f_x)))
        return scale * self.dist(rng, shape=scale.shape)


def init_noise_dist(*, rng, dim, dist, noise_scale, noise_scale_heteroscedastic):

    if noise_scale is not None:
        assert noise_scale_heteroscedastic is None

        scale = noise_scale(rng)
        return SimpleNoise(dist, scale)

    elif noise_scale_heteroscedastic is not None:
        assert noise_scale is None
        assert "rff" in noise_scale_heteroscedastic

        return HeteroscedasticRFFNoise(dist, rng=rng, d=int(dim),
                                       length_scale=noise_scale_heteroscedastic["length_scale"],
                                       output_scale=noise_scale_heteroscedastic["output_scale"],
                                       n_rff=100)
    else:
        raise KeyError("neither `noise_scale` nor `noise_scale_heteroscedastic` are given")