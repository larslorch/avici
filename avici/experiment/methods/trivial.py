import numpy as onp

def trivial_zero(_, data):
    g = onp.zeros_like(data["g"], dtype=onp.int32)
    return dict(g_edges=g)

def trivial_rand(seed, data):
    rng = onp.random.default_rng(seed)
    n_vars = data["g"].shape[-1]
    perm = rng.permutation(onp.eye(n_vars, dtype=onp.int32))
    g = perm.T @ onp.tril(rng.binomial(1, p=0.5, size=(n_vars, n_vars)), k=-1) @ perm
    return dict(g_edges=g)

def trivial_rand_edges(seed, data):
    rng = onp.random.default_rng(seed)
    n_vars = data["g"].shape[-1]
    perm = rng.permutation(onp.eye(n_vars, dtype=onp.int32))
    p_expected = 2 * data["g"].sum() / (n_vars * (n_vars - 1))
    g = perm.T @ onp.tril(rng.binomial(1, p=p_expected, size=(n_vars, n_vars)), k=-1) @ perm
    return dict(g_edges=g)




