import jax.numpy as jnp
from dibs.utils.func import squared_norm_pytree

class AdditiveFrobeniusSEKernel:
    """
    Squared exponential kernel defined as

    :math:`k(Z, Z') = \\text{scale} \\cdot \\exp(- \\frac{1}{h} ||Z - Z'||^2_F )`

    Args:
        h (float): bandwidth parameter
        scale (float): scale parameter

    """

    def __init__(self, *, h=20.0, scale=1.0):
        super(AdditiveFrobeniusSEKernel, self).__init__()

        self.h = h
        self.scale = scale

    def eval(self, *, x, y):
        """Evaluates kernel function

        Args:
            x (ndarray): any shape ``[...]``
            y (ndarray): any shape ``[...]``, but same as ``x``

        Returns:
            kernel value of shape ``[1,]``
        """
        return self.scale * jnp.exp(- jnp.sum((x - y) ** 2.0) / self.h)


class JointAdditiveFrobeniusSEKernel:
    """
    Squared exponential kernel defined as

    :math:`k((Z, \\Theta), (Z', \\Theta')) = \\text{scale}_z \\cdot \\exp(- \\frac{1}{h_z} ||Z - Z'||^2_F ) + \\text{scale}_{\\theta} \\cdot \\exp(- \\frac{1}{h_{\\theta}} ||\\Theta - \\Theta'||^2_F )`

    Args:
        h_latent (float): bandwidth parameter for :math:`Z` term
        h_theta (float): bandwidth parameter for :math:`\\Theta` term
        scale_latent (float): scale parameter for :math:`Z` term
        scale_theta (float): scale parameter  for :math:`\\Theta` term
    """

    def __init__(self, *, h_latent=5.0, h_theta=500.0, scale_latent=1.0, scale_theta=1.0):
        super(JointAdditiveFrobeniusSEKernel, self).__init__()

        self.h_latent = h_latent
        self.h_theta = h_theta
        self.scale_latent = scale_latent
        self.scale_theta = scale_theta

    def eval(self, *, x_latent, x_theta, y_latent, y_theta):
        """Evaluates kernel function k(x, y)

        Args:
            x_latent (ndarray): any shape ``[...]``
            x_theta (Any): any PyTree of ``jnp.array`` tensors
            y_latent (ndarray): any shape ``[...]``, but same as ``x_latent``
            y_theta (Any): any PyTree of ``jnp.array`` tensors, but same as ``x_theta``

        Returns:
            kernel value of shape ``[1,]``
        """

        # compute norm
        latent_squared_norm = jnp.sum((x_latent - y_latent) ** 2.0)
        theta_squared_norm = squared_norm_pytree(x_theta, y_theta)

        # compute kernel
        return (self.scale_latent * jnp.exp(- latent_squared_norm / self.h_latent)
                + self.scale_theta * jnp.exp(- theta_squared_norm / self.h_theta))

