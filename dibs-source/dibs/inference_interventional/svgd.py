import functools
import numpy as onp

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, grad
from jax.tree_util import tree_map
from jax.scipy.special import logsumexp
try:
    from jax.example_libraries import optimizers
except ImportError:
    # for jax <= 2.24
    from jax.experimental import optimizers


from dibs.inference_interventional.dibs import IntervDiBS
from dibs.kernel import AdditiveFrobeniusSEKernel, JointAdditiveFrobeniusSEKernel
from dibs.metrics import ParticleDistribution
from dibs.utils.func import expand_by


class IntervMarginalDiBS(IntervDiBS):
    """
    This class implements Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016)
    for DiBS inference (Lorch et al., 2021) of the marginal DAG posterior :math:`p(G | D)`.
    For joint inference of :math:`p(G, \\Theta | D)`, use the analogous class
    :class:`~dibs.inference.JointDiBS`.

    An SVGD update of tensor :math:`v` is defined as

    :math:`\\phi(v) \\propto \\sum_{u} k(v, u) \\nabla_u \\log p(u) + \\nabla_u k(u, v)`

    Args:
        x (ndarray): observations of shape ``[n_observations, n_vars]``
        interv_mask (ndarray): binary matrix of shape ``[n_observations, n_vars]`` indicating
            whether a given variable was intervened upon in a given sample (intervention = 1, no intervention = 0)
        inference_model: Bayes net inference model defining prior :math:`\\log p(G)`
            and marginal likelihood :math:`\\log p(D | G)`` underlying the inferred posterior.
            Object *has to implement two methods*:
            ``log_graph_prior`` and ``observational_log_marginal_prob``.
            Example: :class:`~dibs.models.BGe`
        kernel: Class of kernel. *Has to implement the method* ``eval(u, v)``.
            Example: :class:`~dibs.kernel.AdditiveFrobeniusSEKernel`
        kernel_param (dict): kwargs to instantiate ``kernel``
        optimizer (str): optimizer identifier
        optimizer_param (dict): kwargs to instantiate ``optimizer``
        alpha_linear (float): slope of of linear schedule for inverse temperature :math:`\\alpha`
            of sigmoid in latent graph model :math:`p(G | Z)`
        beta_linear (float):  slope of of linear schedule for inverse temperature :math:`\\beta`
            of constraint penalty in latent prior :math:`p(Z)`
        tau (float):  constant Gumbel-softmax temperature parameter
        n_grad_mc_samples (int): number of Monte Carlo samples in gradient estimator
            for likelihood term :math:`p(\Theta, D | G)`
        n_acyclicity_mc_samples (int):  number of Monte Carlo samples in gradient estimator
            for acyclicity constraint
        grad_estimator_z (str): gradient estimator :math:`\\nabla_Z` of expectation over :math:`p(G | Z)`;
            choices: ``score`` or ``reparam``
        score_function_baseline (float): scale of additive baseline in score function (REINFORCE) estimator;
            ``score_function_baseline == 0.0`` corresponds to not using a baseline
        latent_prior_std (float): standard deviation of Gaussian prior over :math:`Z`; defaults to ``1/sqrt(k)``
    """

    def __init__(self, *,
                 x,
                 inference_model,
                 interv_mask=None,
                 kernel=AdditiveFrobeniusSEKernel,
                 kernel_param=None,
                 optimizer="rmsprop",
                 optimizer_param=None,
                 alpha_linear=1.0,
                 beta_linear=1.0,
                 tau=1.0,
                 n_grad_mc_samples=128,
                 n_acyclicity_mc_samples=32,
                 grad_estimator_z="score",
                 score_function_baseline=0.0,
                 latent_prior_std=None,
                 verbose=False):

        # handle mutable default args
        if kernel_param is None:
            kernel_param = {"h": 5.0}
        if optimizer_param is None:
            optimizer_param = {"stepsize": 0.005}

        # handle interv mask in observational case
        if interv_mask is None:
            interv_mask = jnp.zeros_like(x, dtype=jnp.int32)

        # init DiBS superclass methods
        super(IntervMarginalDiBS, self).__init__(
            x=x,
            interv_mask=interv_mask,
            log_graph_prior=inference_model.log_graph_prior,
            log_joint_prob=inference_model.interventional_log_marginal_prob,
            alpha_linear=alpha_linear,
            beta_linear=beta_linear,
            tau=tau,
            n_grad_mc_samples=n_grad_mc_samples,
            n_acyclicity_mc_samples=n_acyclicity_mc_samples,
            grad_estimator_z=grad_estimator_z,
            score_function_baseline=score_function_baseline,
            latent_prior_std=latent_prior_std,
            verbose=verbose,
        )

        self.inference_model = inference_model
        self.eltwise_log_marginal_likelihood = vmap(lambda g, x_ho, interv_msk_ho:
            inference_model.interventional_log_marginal_prob(g, None, x_ho, interv_msk_ho, None), (0, None, None), 0)

        self.kernel = kernel(**kernel_param)

        if optimizer == 'gd':
            self.opt = optimizers.sgd(optimizer_param['stepsize'])
        elif optimizer == 'rmsprop':
            self.opt = optimizers.rmsprop(optimizer_param['stepsize'])
        else:
            raise ValueError()


    def _sample_initial_random_particles(self, *, key, n_particles, n_dim=None):
        """
        Samples random particles to initialize SVGD

        Args:
            key (ndarray): rng key
            n_particles (int): number of particles inferred
            n_dim (int): size of latent dimension :math:`k`. Defaults to ``n_vars``, s.t. :math:`k = d`

        Returns:
            batch of latent tensors ``[n_particles, d, k, 2]``
        """
        # default full rank
        if n_dim is None:
            n_dim = self.n_vars

        # like prior
        std = self.latent_prior_std or (1.0 / jnp.sqrt(n_dim))

        # sample
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, self.n_vars, n_dim, 2)) * std

        return z


    def _f_kernel(self, x_latent, y_latent):
        """
        Evaluates kernel

        Args:
            x_latent (ndarray): latent tensor of shape ``[d, k, 2]``
            y_latent (ndarray): latent tensor of shape ``[d, k, 2]``

        Returns:
            kernel value of shape ``[1, ]``
        """
        return self.kernel.eval(x=x_latent, y=y_latent)


    def _f_kernel_mat(self, x_latents, y_latents):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents (ndarray): latent tensor of shape ``[A, d, k, 2]``
            y_latents (ndarray): latent tensor of shape ``[B, d, k, 2]``

        Returns:
            kernel values of shape ``[A, B]``
        """
        return vmap(vmap(self._f_kernel, (None, 0), 0), (0, None), 0)(x_latents, y_latents)


    def _eltwise_grad_kernel_z(self, x_latents, y_latent):
        """
        Computes gradient :math:`\\nabla_Z k(Z, Z')` elementwise for each provided particle :math:`Z`
        in batch ``x_latents`

        Args:
            x_latents (ndarray): batch of latent particles for :math:`Z` of shape ``[n_particles, d, k, 2]``
            y_latent (ndarray): single latent particle :math:`Z'` ``[d, k, 2]``

        Returns:
            batch of gradients of shape ``[n_particles, d, k, 2]``
        """
        grad_kernel_z = grad(self._f_kernel, 0)
        return vmap(grad_kernel_z, (0, None), 0)(x_latents, y_latent)

    def _z_update(self, single_z, kxx, z, grad_log_prob_z):
        """
        Computes SVGD update for ``single_z`` particle given the kernel values
        ``kxx`` and the :math:`d/dZ` gradients of the target density for each of the available particles

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``, which is the Z particle being updated
            kxx (ndarray): pairwise kernel values for all particles ``[n_particles, n_particles]``
            z (ndarray):  all latent particles ``[n_particles, d, k, 2]``
            grad_log_prob_z (ndarray): gradients of all Z particles w.r.t
                target density of shape ``[n_particles, d, k, 2]``

        Returns
            transform vector of shape ``[d, k, 2]`` for the particle ``single_z``

        """

        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None, None] * grad_log_prob_z
        repulsion = self._eltwise_grad_kernel_z(z, single_z)

        # average and negate (for optimizer)
        return - (weighted_gradient_ascent + repulsion).mean(axis=0)

    def _parallel_update_z(self, *args):
        """
        Vectorizes :func:`~dibs.inference.MarginalDiBS._z_update`
        for all available particles in batched first input dim (``single_z``)
        Otherwise, same inputs as :func:`~dibs.inference.MarginalDiBS._z_update`.
        """
        return vmap(self._z_update, (0, 1, None, None), 0)(*args)

    def _svgd_step(self, t, opt_state_z, key, sf_baseline):
        """
        Performs a single SVGD step in the DiBS framework, updating all :math:`Z` particles jointly.

        Args:
            t (int): step
            opt_state_z: optimizer state for latent :math:`Z` particles; contains ``[n_particles, d, k, 2]``
            key (ndarray): prng key
            sf_baseline (ndarray): batch of baseline values of shape ``[n_particles, ]``
                in case score function gradient is used

        Returns:
            the updated inputs ``opt_state_z``, ``key``, ``sf_baseline``
        """

        z = self.get_params(opt_state_z)  # [n_particles, d, k, 2]
        n_particles = z.shape[0]

        # d/dz log p(D | z)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(z, None, sf_baseline, t, jnp.array(batch_subk))
        # here `None` is a placeholder for theta (in the joint inference case)
        # since this is an inherited function from the general `DiBS` class

        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(z, jnp.array(batch_subk), t)

        # d/dz log p(z, D) = d/dz log p(z)  + log p(D | z)
        dz_log_prob = dz_log_prior + dz_log_likelihood

        # k(z, z) for all particles
        kxx = self._f_kernel_mat(z, z)

        # transformation phi() applied in batch to each particle individually
        phi_z = self._parallel_update_z(z, kxx, z, dz_log_prob)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)

        return opt_state_z, key, sf_baseline

    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0, 2))
    def _svgd_loop(self, start, n_steps, init):
        return jax.lax.fori_loop(start, start + n_steps, lambda i, args: self._svgd_step(i, *args), init)

    def sample(self, *, key, n_particles, steps, n_dim_particles=None, callback=None, callback_every=None):
        """
        Use SVGD with DiBS to sample ``n_particles`` particles :math:`G` from the marginal posterior
        :math:`p(G | D)` as defined by the BN model ``self.inference_model``

        Arguments:
            key (ndarray): prng key
            n_particles (int): number of particles to sample
            steps (int): number of SVGD steps performed
            n_dim_particles (int): latent dimensionality :math:`k` of particles :math:`Z = \{ U, V \}`
                with :math:`U, V \\in \\mathbb{R}^{k \\times d}`. Default is ``n_vars``
            callback: function to be called every ``callback_every`` steps of SVGD.
            callback_every: if ``None``, ``callback`` is only called after particle updates have finished

        Returns:
            batch of samples :math:`G \\sim p(G | D)` of shape ``[n_particles, n_vars, n_vars]``

        """

        # randomly sample initial particles
        key, subk = random.split(key)
        init_z = self._sample_initial_random_particles(key=subk, n_particles=n_particles, n_dim=n_dim_particles)

        # initialize score function baseline (one for each particle)
        n_particles, _, n_dim, _ = init_z.shape
        sf_baseline = jnp.zeros(n_particles)

        if self.latent_prior_std is None:
            self.latent_prior_std = 1.0 / jnp.sqrt(n_dim)

        # maintain updated particles with optimizer state
        opt_init, self.opt_update, get_params = self.opt
        self.get_params = jit(get_params)
        opt_state_z = opt_init(init_z)

        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        # faster if for-loop is functionally pure and compiled, so only interrupt for callback
        callback_every = callback_every or steps
        for t in (range(0, steps, callback_every) if steps else range(0)):

            # perform sequence of SVGD steps
            opt_state_z, key, sf_baseline = self._svgd_loop(t, callback_every, (opt_state_z, key, sf_baseline))

            # callback
            if callback:
                z = self.get_params(opt_state_z)
                callback(
                    dibs=self,
                    t=t + callback_every,
                    zs=z,
                )

        # retrieve transported particles
        z_final = jax.device_get(self.get_params(opt_state_z))

        # as alpha is large, we can convert the latents Z to their corresponding graphs G
        g_final = self.particle_to_g_lim(z_final)
        return g_final

    def get_empirical(self, g):
        """
        Converts batch of binary (adjacency) matrices into *empirical* particle distribution
        where mixture weights correspond to counts/occurrences

        Args:
            g (ndarray): batch of graph samples ``[n_particles, d, d]`` with binary values

        Returns:
            :class:`~dibs.metrics.ParticleDistribution`:
            particle distribution of graph samples and associated log probabilities
        """
        N, _, _ = g.shape
        unique, counts = onp.unique(g, axis=0, return_counts=True)

        # empirical distribution using counts
        logp = jnp.log(counts) - jnp.log(N)

        return ParticleDistribution(logp=logp, g=g)

    def get_mixture(self, g):
        """
        Converts batch of binary (adjacency) matrices into *mixture* particle distribution,
        where mixture weights correspond to unnormalized target (i.e. posterior) probabilities

        Args:
            g (ndarray): batch of graph samples ``[n_particles, d, d]`` with binary values

        Returns:
            :class:`~dibs.metrics.ParticleDistribution`:
            particle distribution of graph samples and associated log probabilities

        """

        N, _, _ = g.shape

        # mixture weighted by respective marginal probabilities
        eltwise_log_marginal_target = vmap(lambda single_g:
           self.log_joint_prob(single_g, None, self.x, self.interv_mask, None), 0, 0)
        logp = eltwise_log_marginal_target(g)
        logp -= logsumexp(logp)

        return ParticleDistribution(logp=logp, g=g)




class IntervJointDiBS(IntervDiBS):
    """
    This class implements Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016)
    for DiBS inference (Lorch et al., 2021) of the marginal DAG posterior :math:`p(G | D)`.
    For marginal inference of :math:`p(G | D)`, use the analogous class
    :class:`~dibs.inference.MarginalDiBS`.

    An SVGD update of tensor :math:`v` is defined as

    :math:`\\phi(v) \\propto \\sum_{u} k(v, u) \\nabla_u \\log p(u) + \\nabla_u k(u, v)`

    Args:
        x (ndarray): observations of shape ``[n_observations, n_vars]``
        interv_mask (ndarray): binary matrix of shape ``[n_observations, n_vars]`` indicating
            whether a given variable was intervened upon in a given sample (intervention = 1, no intervention = 0)
        inference_model: Bayes net inference model defining prior :math:`\\log p(G)`
            and joint likelihood :math:`\\log p(\\Theta, D | G) = \\log p(\\Theta | G) + \\log p(D | G, \\Theta``
            underlying the inferred posterior. Object *has to implement two methods*:
            ``log_graph_prior`` and ``observational_log_joint_prob``.
            Example: :class:`~dibs.models.LinearGaussian`
        kernel: Class of kernel. *Has to implement the method* ``eval(u, v)``.
            Example: :class:`~dibs.kernel.JointAdditiveFrobeniusSEKernel`
        kernel_param (dict): kwargs to instantiate ``kernel``
        optimizer (str): optimizer identifier
        optimizer_param (dict): kwargs to instantiate ``optimizer``
        alpha_linear (float): slope of of linear schedule for inverse temperature :math:`\\alpha`
            of sigmoid in latent graph model :math:`p(G | Z)`
        beta_linear (float):  slope of of linear schedule for inverse temperature :math:`\\beta`
            of constraint penalty in latent prior :math:`p(Z)`
        tau (float):  constant Gumbel-softmax temperature parameter
        n_grad_mc_samples (int): number of Monte Carlo samples in gradient estimator
            for likelihood term :math:`p(\Theta, D | G)`
        n_acyclicity_mc_samples (int):  number of Monte Carlo samples in gradient estimator
            for acyclicity constraint
        grad_estimator_z (str): gradient estimator :math:`\\nabla_Z` of expectation over :math:`p(G | Z)`;
            choices: ``score`` or ``reparam``
        score_function_baseline (float): scale of additive baseline in score function (REINFORCE) estimator;
            ``score_function_baseline == 0.0`` corresponds to not using a baseline
        latent_prior_std (float): standard deviation of Gaussian prior over :math:`Z`; defaults to ``1/sqrt(k)``

    """

    def __init__(self, *,
                 x,
                 inference_model,
                 interv_mask=None,
                 kernel=JointAdditiveFrobeniusSEKernel,
                 kernel_param=None,
                 optimizer="rmsprop",
                 optimizer_param=None,
                 alpha_linear=0.05,
                 beta_linear=1.0,
                 tau=1.0,
                 n_grad_mc_samples=128,
                 n_acyclicity_mc_samples=32,
                 grad_estimator_z="reparam",
                 score_function_baseline=0.0,
                 latent_prior_std=None,
                 verbose=False):

        # handle mutable default args
        if kernel_param is None:
            kernel_param = {"h_latent": 5.0, "h_theta": 500.0}
        if optimizer_param is None:
            optimizer_param = {"stepsize": 0.005}

        # handle interv mask in observational case
        if interv_mask is None:
            interv_mask = jnp.zeros_like(x, dtype=jnp.int32)

        # init DiBS superclass methods
        super(IntervJointDiBS, self).__init__(
            x=x,
            interv_mask=interv_mask,
            log_graph_prior=inference_model.log_graph_prior,
            log_joint_prob=inference_model.interventional_log_joint_prob,
            alpha_linear=alpha_linear,
            beta_linear=beta_linear,
            tau=tau,
            n_grad_mc_samples=n_grad_mc_samples,
            n_acyclicity_mc_samples=n_acyclicity_mc_samples,
            grad_estimator_z=grad_estimator_z,
            score_function_baseline=score_function_baseline,
            latent_prior_std=latent_prior_std,
            verbose=verbose,
        )

        self.inference_model = inference_model
        self.eltwise_log_likelihood = vmap(lambda g, theta, x_ho, interv_msk_ho:
            inference_model.interventional_log_joint_prob(g, theta, x_ho, interv_msk_ho, None), (0, 0, None, None), 0)

        self.kernel = kernel(**kernel_param)

        if optimizer == 'gd':
            self.opt = optimizers.sgd(optimizer_param['stepsize'])
        elif optimizer == 'rmsprop':
            self.opt = optimizers.rmsprop(optimizer_param['stepsize'])
        else:
            raise ValueError()
       

    def _sample_initial_random_particles(self, *, key, n_particles, n_dim=None):
        """
        Samples random particles to initialize SVGD

        Args:
            key (ndarray): rng key
            n_particles (int): number of particles inferred
            n_dim (int): size of latent dimension :math:`k`. Defaults to ``n_vars``, s.t. :math:`k = d`

        Returns:
            batch of latent tensors ``[n_particles, d, k, 2]``
        """
        # default full rank
        if n_dim is None:
            n_dim = self.n_vars
        
        # std like Gaussian prior over Z           
        std = self.latent_prior_std  or (1.0 / jnp.sqrt(n_dim))

        # sample from parameter prior
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, self.n_vars, n_dim, 2)) * std

        key, subk = random.split(key)
        theta = self.inference_model.sample_parameters(key=subk, n_particles=n_particles, n_vars=self.n_vars)

        return z, theta


    def _f_kernel(self, x_latent, x_theta, y_latent, y_theta):
        """
        Evaluates kernel

        Args:
            x_latent (ndarray): latent tensor of shape ``[d, k, 2]``
            x_theta (Any): parameter PyTree
            y_latent (ndarray): latent tensor of shape ``[d, k, 2]``
            y_theta (Any): parameter PyTree

        Returns:
            kernel value of shape ``[1, ]``

        """
        return self.kernel.eval(
            x_latent=x_latent, x_theta=x_theta,
            y_latent=y_latent, y_theta=y_theta)
    

    def _f_kernel_mat(self, x_latents, x_thetas, y_latents, y_thetas):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents (ndarray): latent tensor of shape ``[A, d, k, 2]``
            x_thetas (Any): parameter PyTree with batch size ``A`` as leading dim
            y_latents (ndarray): latent tensor of shape ``[B, d, k, 2]``
            y_thetas (Any): parameter PyTree with batch size ``B`` as leading dim

        Returns:
            kernel values of shape ``[A, B]``
        """
        return vmap(vmap(self._f_kernel, (None, None, 0, 0), 0),
                    (0, 0, None, None), 0)(x_latents, x_thetas, y_latents, y_thetas)

    
    def _eltwise_grad_kernel_z(self, x_latents, x_thetas, y_latent, y_theta):
        """
        Computes gradient :math:`\\nabla_Z k((Z, \\Theta), (Z', \\Theta'))` elementwise
        for each provided particle :math:`(Z, \\Theta)` in batch (``x_latents`, ``x_thetas``)

        Args:
            x_latents (ndarray): batch of latent particles for :math:`Z` of shape ``[n_particles, d, k, 2]``
            x_thetas (Any): batch of parameter PyTrees for :math:`\\Theta` with leading dim ``n_particles``
            y_latent (ndarray): single latent particle :math:`Z'` ``[d, k, 2]``
            y_theta (Any): single parameter PyTree for :math:`\\Theta'`

        Returns:
            batch of gradients of shape ``[n_particles, d, k, 2]``
        
        """        
        grad_kernel_z = grad(self._f_kernel, 0)
        return vmap(grad_kernel_z, (0, 0, None, None), 0)(x_latents, x_thetas, y_latent, y_theta)


    def _eltwise_grad_kernel_theta(self, x_latents, x_thetas, y_latent, y_theta):
        """
        Computes gradient :math:`\\nabla_{\\Theta} k((Z, \\Theta), (Z', \\Theta'))` elementwise
        for each provided particle :math:`(Z, \\Theta)` in batch (``x_latents`, ``x_thetas``)

        Args:
            x_latents (ndarray): batch of latent particles for :math:`Z` of shape ``[n_particles, d, k, 2]``
            x_thetas (Any): batch of parameter PyTrees for :math:`\\Theta` with leading dim ``n_particles``
            y_latent (ndarray): single latent particle :math:`Z'` ``[d, k, 2]``
            y_theta (Any): single parameter PyTree for :math:`\\Theta'`

        Returns:
            batch of gradient PyTrees with leading dim ``n_particles``
        """
        grad_kernel_theta = grad(self._f_kernel, 1)
        return vmap(grad_kernel_theta, (0, 0, None, None), 0)(x_latents, x_thetas, y_latent, y_theta)


    def _z_update(self, single_z, single_theta, kxx, z, theta, grad_log_prob_z):
        """
        Computes SVGD update for ``single_z`` of a particle tuple (``single_z``, ``single_theta``)
        particle given the kernel values ``kxx`` and the :math:`d/dZ` gradients of the target density
        for each of the available particles

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``, which is the :math:`\\Z` particle being updated
            single_theta (Any): single parameter PyTree, the :math:`\\Theta` particle of the :math:`\\Z` particle being updated
            kxx (ndarray): pairwise kernel values for all particles, of shape ``[n_particles, n_particles]``
            z (ndarray):  all latent particles ``[n_particles, d, k, 2]``
            theta (Any): all theta particles as PyTree with leading dim `n_particles`
            grad_log_prob_z (ndarray): gradients of all Z particles w.r.t
                target density of shape ``[n_particles, d, k, 2]``

        Returns
            transform vector of shape ``[d, k, 2]`` for the particle ``single_z``
        """
    
        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None, None] * grad_log_prob_z
        repulsion = self._eltwise_grad_kernel_z(z, theta, single_z, single_theta)

        # average and negate (for optimizer)
        return - (weighted_gradient_ascent + repulsion).mean(axis=0)

    def _parallel_update_z(self, *args):
        """
        Vectorizes :func:`~dibs.inference.JointDiBS._z_update`
        for all available particles in batched first and second input
        dim (``single_z``, ``single_theta``)
        Otherwise, same inputs as :func:`~dibs.inference.JointDiBS._z_update`.
        """
        return vmap(self._z_update, (0, 0, 1, None, None, None), 0)(*args)


    def _theta_update(self, single_z, single_theta, kxx, z, theta, grad_log_prob_theta):
        """
        Computes SVGD update for ``single_theta`` of a particle tuple (``single_z``, ``single_theta``)
        particle given the kernel values ``kxx`` and the :math:`d/d\\Theta` gradients of the target density
        for each of the available particles.

        Analogous to :func:`dibs.inference.JointDiBS._z_update` but for updating :math:`\Theta`.

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``, which is the particle
                particle of the :math:`\\Theta` particle being updated
            single_theta (Any): single parameter PyTree, the :math:`\\Theta`, which is the
                :math:`\\Theta` particle being updated
            kxx (ndarray): pairwise kernel values for all particles, of shape ``[n_particles, n_particles]``
            z (ndarray):  all latent particles ``[n_particles, d, k, 2]``
            theta (Any): all theta particles as PyTree with leading dim `n_particles`
            grad_log_prob_theta (ndarray): gradients of all :math:`\\Theta` particles w.r.t
                target density of shape ``[n_particles, d, k, 2]``

        Returns:
            transform vector PyTree with leading dim ``n_particles`` for the particle ``single_theta``
        """
    
        # compute terms in sum
        weighted_gradient_ascent = tree_map(
            lambda leaf_theta_grad: 
                expand_by(kxx, leaf_theta_grad.ndim - 1) * leaf_theta_grad, 
            grad_log_prob_theta)
        
        repulsion = self._eltwise_grad_kernel_theta(z, theta, single_z, single_theta)

        # average and negate (for optimizer)
        return  tree_map(
            lambda grad_asc_leaf, repuls_leaf: 
                - (grad_asc_leaf + repuls_leaf).mean(axis=0), 
            weighted_gradient_ascent, 
            repulsion)


    def _parallel_update_theta(self, *args):
        """
        Vectorizes :func:`~dibs.inference.JointDiBS._theta_update`
        for all available particles in batched first and second input
        dim (``single_z``, ``single_theta``).
        Otherwise, same inputs as :func:`~dibs.inference.JointDiBS._theta_update`.
        """
        return vmap(self._theta_update, (0, 0, 1, None, None, None), 0)(*args)


    def _svgd_step(self, t, opt_state_z, opt_state_theta, key, sf_baseline):
        """
        Performs a single SVGD step in the DiBS framework, updating all :math:`(Z, \\Theta)` particles jointly.

        Args:
            t (int): step
            opt_state_z: optimizer state for latent :math:`Z` particles; contains ``[n_particles, d, k, 2]``
            opt_state_theta: optimizer state for parameter :math:`\\Theta` particles;
                contains PyTree with ``n_particles`` leading dim
            key (ndarray): prng key
            sf_baseline (ndarray): batch of baseline values of shape ``[n_particles, ]``
                in case score function gradient is used

        Returns:
            the updated inputs ``opt_state_z``, ``opt_state_theta``, ``key``, ``sf_baseline``
        """

        z = self.get_params(opt_state_z) # [n_particles, d, k, 2]
        theta = self.get_params(opt_state_theta) # PyTree with `n_particles` leading dim
        n_particles = z.shape[0]

        # d/dtheta log p(theta, D | z)
        key, *batch_subk = random.split(key, n_particles + 1)
        dtheta_log_prob = self.eltwise_grad_theta_likelihood(z, theta, t, jnp.array(batch_subk))

        # d/dz log p(theta, D | z)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(z, theta, sf_baseline, t, jnp.array(batch_subk))

        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(z, jnp.array(batch_subk), t)

        # d/dz log p(z, theta, D) = d/dz log p(z)  + log p(theta, D | z) 
        dz_log_prob = dz_log_prior + dz_log_likelihood
        
        # k((z, theta), (z, theta)) for all particles
        kxx = self._f_kernel_mat(z, theta, z, theta)

        # transformation phi() applied in batch to each particle individually
        phi_z = self._parallel_update_z(z, theta, kxx, z, theta, dz_log_prob)
        phi_theta = self._parallel_update_theta(z, theta, kxx, z, theta, dtheta_log_prob)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)
        opt_state_theta = self.opt_update(t, phi_theta, opt_state_theta)

        return opt_state_z, opt_state_theta, key, sf_baseline
    
    
    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0, 2))
    def _svgd_loop(self, start, n_steps, init):
        return jax.lax.fori_loop(start, start + n_steps, lambda i, args: self._svgd_step(i, *args), init)


    def sample(self, *, key, n_particles, steps, n_dim_particles=None, callback=None, callback_every=None):
        """
        Use SVGD with DiBS to sample ``n_particles`` particles :math:`(G, \\Theta)` from the joint posterior
        :math:`p(G, \\Theta | D)` as defined by the BN model ``self.inference_model``

        Arguments:
            key (ndarray): prng key
            n_particles (int): number of particles to sample
            steps (int): number of SVGD steps performed
            n_dim_particles (int): latent dimensionality :math:`k` of particles :math:`Z = \{ U, V \}`
                with :math:`U, V \\in \\mathbb{R}^{k \\times d}`. Default is ``n_vars``
            callback: function to be called every ``callback_every`` steps of SVGD.
            callback_every: if ``None``, ``callback`` is only called after particle updates have finished

        Returns:
            tuple of shape (``[n_particles, n_vars, n_vars]``, ``PyTree``) where ``PyTree`` has leading dimension ``n_particles``:
            batch of samples :math:`G, \\Theta \\sim p(G, \\Theta | D)`

        """

        # randomly sample initial particles
        key, subk = random.split(key)
        init_z, init_theta = self._sample_initial_random_particles(key=subk, n_particles=n_particles,
                                                                   n_dim=n_dim_particles)

        # initialize score function baseline (one for each particle)
        n_particles, _, n_dim, _ = init_z.shape
        sf_baseline = jnp.zeros(n_particles)

        if self.latent_prior_std is None:
            self.latent_prior_std = 1.0 / jnp.sqrt(n_dim)

        # maintain updated particles with optimizer state
        opt_init, self.opt_update, get_params = self.opt
        self.get_params = jit(get_params)
        opt_state_z = opt_init(init_z)
        opt_state_theta = opt_init(init_theta)

        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        # faster if for-loop is functionally pure and compiled, so only interrupt for callback
        callback_every = callback_every or steps
        for t in (range(0, steps, callback_every) if steps else range(0)):

            # perform sequence of SVGD steps
            opt_state_z, opt_state_theta, key, sf_baseline = self._svgd_loop(t, callback_every,
                                                                             (opt_state_z, opt_state_theta, key, sf_baseline))

            # callback
            if callback:
                z = self.get_params(opt_state_z)
                theta = self.get_params(opt_state_theta)
                callback(
                    dibs=self,
                    t=t + callback_every,
                    zs=z,
                    thetas=theta,
                )

        # retrieve transported particles
        z_final = jax.device_get(self.get_params(opt_state_z))
        theta_final = jax.device_get(self.get_params(opt_state_theta))

        # as alpha is large, we can convert the latents Z to their corresponding graphs G
        g_final = self.particle_to_g_lim(z_final)
        return g_final, theta_final


    def get_empirical(self, g, theta):
        """
        Converts batch of binary (adjacency) matrices and parameters into *empirical* particle distribution
        where mixture weights correspond to counts/occurrences

        Args:
            g (ndarray): batch of graph samples ``[n_particles, d, d]`` with binary values
            theta (Any): PyTree with leading dim ``n_particles``

        Returns:
            :class:`~dibs.metrics.ParticleDistribution`:
            particle distribution of graph and parameter samples and associated log probabilities
        """

        N, _, _ = g.shape

        # since theta continuous, each particle (G, theta) is unique always
        logp = - jnp.log(N) * jnp.ones(N)

        return ParticleDistribution(logp=logp, g=g, theta=theta)


    def get_mixture(self, g, theta):
        """
        Converts batch of binary (adjacency) matrices and particles into *mixture* particle distribution,
        where mixture weights correspond to unnormalized target (i.e. posterior) probabilities

        Args:
            g (ndarray): batch of graph samples ``[n_particles, d, d]`` with binary values
            theta (Any): PyTree with leading dim ``n_particles``

        Returns:
            :class:`~dibs.metrics.ParticleDistribution`:
            particle distribution of graph and parameter samples and associated log probabilities

        """

        N, _, _ = g.shape

        # mixture weighted by respective joint probabilities
        eltwise_log_joint_target = vmap(lambda single_g, single_theta:
            self.log_joint_prob(single_g, single_theta, self.x, self.interv_mask, None), (0, 0), 0)
        logp = eltwise_log_joint_target(g, theta)
        logp -= logsumexp(logp)

        return ParticleDistribution(logp=logp, g=g, theta=theta)
