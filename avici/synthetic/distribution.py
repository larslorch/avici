import numpy as onp


def gaussian(rng, *, shape=None, scale=1):
    """N(0, scale^2)"""
    sample = scale * rng.normal(size=shape)
    return sample



def laplace(rng, *, shape=None, scale=1):
    """Laplace(0, scale)"""
    sample = scale * rng.laplace(size=shape)
    return sample



def cauchy(rng, *, shape=None, scale=1):
    """Cauchy(0, scale)"""
    sample = scale * rng.standard_cauchy(size=shape)
    return sample


"""
Uniform
"""


def uniform(rng, *, shape=None, low, high):
    """Uniform"""
    sample = rng.uniform(size=shape, low=low, high=high)
    return sample


def signed_uniform(rng, *, shape=None, low, high):
    """Uniform with sign"""
    sgn = rng.choice([-1, 1], size=shape)
    sample = sgn * rng.uniform(size=shape, low=low, high=high)
    return sample


def randint(rng, *, low, high, shape=None, endpoint=True):
    """Uniform random integer"""
    sample = rng.integers(size=shape, low=low, high=high, endpoint=endpoint)
    return sample



"""
Beta
"""

def beta(rng, *, a, b, shape=None):
    """Beta(a, b)"""
    sample = rng.beta(a, b, size=shape)
    return sample


