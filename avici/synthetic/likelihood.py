import jax.numpy as jnp
import jax.scipy.stats as stats

def linear_additive_standard_gaussian(g, theta, x):
    x = x.squeeze(-1)
    # (g * theta.T) because theta[j] indexes params of x_j and g[:, j] indexes parents of x_j (ignoring batch dim)
    theta_masked = g * jnp.swapaxes(theta, -2, -1)
    eltwise_logliks = stats.norm.logpdf(x=x, loc=x @ theta_masked, scale=1)  # [n_observations, n_vars]
    return jnp.sum(eltwise_logliks, axis=(-1, -2))