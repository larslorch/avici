from avici.synthetic import Distribution


class Gaussian(Distribution):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, rng, shape=None):
        return self.scale * rng.normal(size=shape)


class Laplace(Distribution):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, rng, shape=None):
        return self.scale * rng.laplace(size=shape)


class Cauchy(Distribution):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, rng, shape=None):
        return self.scale * rng.standard_cauchy(size=shape)


class Uniform(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, rng, shape=None):
        return rng.uniform(size=shape, low=self.low, high=self.high)


class SignedUniform(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, rng, shape=None):
        sgn = rng.choice([-1, 1], size=shape)
        return sgn * rng.uniform(size=shape, low=self.low, high=self.high)


class RandInt(Distribution):
    def __init__(self, low, high, endpoint=True):
        self.low = low
        self.high = high
        self.endpoint = endpoint

    def __call__(self, rng, shape=None):
        return rng.integers(size=shape, low=self.low, high=self.high, endpoint=self.endpoint)


class Beta(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, rng, shape=None):
        return rng.beta(self.a, self.b, size=shape)
