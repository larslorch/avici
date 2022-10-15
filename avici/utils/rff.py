import functools
import numpy as onp

def make_rff_rbf_function(*, rng, d, c, ls, loc=0.0, n_rff=100):
    """
    Random fourier features
    https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

    As n_rff goes to infinity, function returned corresponds to a sample from GP prior with
    RBF kernel

            k(x, y) = c^2 * exp(- |x-y|^2 / (2 * ls^2)

    Args:
        rng
        d: int dimensionality of input
        c: output scale
        ls: length scale
        loc: constant shift
        n_rff: number of random fourier features

    Returns:
        callable function mapping [..., d] -> [...]
    """

    # draw rffs
    # [d, n_rff]
    omega = rng.normal(loc=0, scale=1.0 / ls, size=(d, n_rff))

    # [n_rff]
    b = rng.uniform(0, 2 * onp.pi, size=(n_rff,))

    # [n_rff, ]
    w = rng.normal(loc=0, scale=1.0, size=(n_rff,))

    def f(x):
        # [..., d] -> [...]

        # feature map phi = cos(omega'x + b)
        # [..., n_rff]
        phi = onp.cos(onp.einsum('db,...d->...b', omega, x) + b)

        # f(x) = w'phi(x)
        # [...]
        return loc + onp.sqrt(2.0) * c * onp.einsum('b,...b->...', w, phi) / onp.sqrt(n_rff)

    return f