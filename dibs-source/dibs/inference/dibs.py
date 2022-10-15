import jax.numpy as jnp
from jax import vmap, random, grad
from jax.scipy.special import logsumexp
from jax.nn import sigmoid, log_sigmoid
import jax.lax as lax
from jax.tree_util import tree_map

from dibs.graph_utils import acyclic_constr_nograd
from dibs.utils.func import expand_by


class DiBS:
    """
    This class implements the backbone for DiBS, i.e. all gradient estimators and sampling
    components. Any inference method in the DiBS framework should inherit from this class.

    Args:
        x (ndarray): matrix of shape ``[n_observations, n_vars]`` of i.i.d. observations of the variables
        log_graph_prior (callable):
            function implementing prior :math:`\\log p(G)` of soft adjacency matrix of
            edge probabilities.
            For example: :func:`~dibs.graph.ErdosReniDAGDistribution.unnormalized_log_prob_soft`
            or usually bound in e.g. :func:`~dibs.graph.LinearGaussian.log_graph_prior`
        log_joint_prob (callable):
            function implementing joint likelihood :math:`\\log p(\Theta, D | G)`
            of parameters and observations given the discrete graph adjacency matrix
            For example: :func:`dibs.models.LinearGaussian.observational_log_joint_prob`.
            When inferring the marginal posterior :math:`p(G | D)` via a closed-form
            marginal likelihood :math:`\\log p(D | G)`, the same function signature has to be
            satisfied (simply ignoring :math:`\\Theta`)
        alpha_linear (float): slope of of linear schedule for inverse temperature :math:`\\alpha`
            of sigmoid in latent graph model :math:`p(G | Z)`
        beta_linear (float):  slope of of linear schedule for inverse temperature :math:`\\beta`
            of constraint penalty in latent prio :math:`p(Z)`
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
                 log_graph_prior,
                 log_joint_prob,
                 alpha_linear=0.05,
                 beta_linear=1.0,
                 tau=1.0,
                 n_grad_mc_samples=128,
                 n_acyclicity_mc_samples=32,
                 grad_estimator_z='reparam',
                 score_function_baseline=0.0,
                 latent_prior_std=None,
                 verbose=False):
        super(DiBS, self).__init__()

        self.x = x
        self.n_vars = x.shape[-1]
        self.log_graph_prior = log_graph_prior
        self.log_joint_prob = log_joint_prob
        self.alpha = lambda t: (alpha_linear * t)
        self.beta = lambda t: (beta_linear * t)
        self.tau = tau
        self.n_grad_mc_samples = n_grad_mc_samples
        self.n_acyclicity_mc_samples = n_acyclicity_mc_samples
        self.grad_estimator_z = grad_estimator_z
        self.score_function_baseline = score_function_baseline
        self.latent_prior_std = latent_prior_std
        self.verbose = verbose

    """
    Backbone functionality
    """

    def vec_to_mat(self, z, n_vars):
        """
        Reshapes particle to latent adjacency matrix form. Last dim gets shaped into matrix
        
        Args:
            z (ndarray): flattened matrix of shape ``[..., n_vars * n_vars]``

        Returns:
            matrix of shape ``[..., d, d]``
        """
        return z.reshape(*z.shape[:-1], n_vars, n_vars)


    def mat_to_vec(self, w):
        """
        Reshapes latent adjacency matrix form to particle. Last two dims get flattened into vector
        
        Args:
            w (ndarray): matrix of shape ``[..., d, d]``
        
        Returns:
            flattened matrix of shape ``[..., d * d]``
        """
        n_vars = w.shape[-1]
        return w.reshape(*w.shape[:-2], n_vars * n_vars)


    def particle_to_g_lim(self, z):
        """
        Returns :math:`G` corresponding to :math:`\\alpha = \\infty` for particles `z`

        Args:
            z (ndarray): latent variables ``[..., d, k, 2]``

        Returns:
            graph adjacency matrices of shape ``[..., d, d]``
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        g_samples = (scores > 0).astype(jnp.int32)

        # zero diagonal
        g_samples = g_samples.at[..., jnp.arange(scores.shape[-1]), jnp.arange(scores.shape[-1])].multiply(0)
        return g_samples


    def sample_g(self, p, subk, n_samples):
        """
        Sample Bernoulli matrix according to matrix of probabilities

        Args:
            p (ndarray): matrix of probabilities ``[d, d]``
            n_samples (int): number of samples
            subk (ndarray): rng key
        
        Returns:
            an array of matrices sampled according to ``p`` of shape ``[n_samples, d, d]``
        """
        n_vars = p.shape[-1]
        g_samples = self.vec_to_mat(random.bernoulli(
            subk, p=self.mat_to_vec(p), shape=(n_samples, n_vars * n_vars)), n_vars).astype(jnp.int32)

        # mask diagonal since it is explicitly not modeled
        g_samples = g_samples.at[..., jnp.arange(p.shape[-1]), jnp.arange(p.shape[-1])].multiply(0)

        return g_samples

    def particle_to_soft_graph(self, z, eps, t):
        """ 
        Gumbel-softmax / concrete distribution using Logistic(0,1) samples ``eps``

        Args:
            z (ndarray): a single latent tensor :math:`Z` of shape ``[d, k, 2]```
            eps (ndarray): random i.i.d. Logistic(0,1) noise  of shape ``[d, d]``
            t (int): step
        
        Returns:
            Gumbel-softmax sample of adjacency matrix [d, d]
        """
        scores = jnp.einsum('...ik,...jk->...ij', z[..., 0], z[..., 1])

        # soft reparameterization using gumbel-softmax/concrete distribution
        # eps ~ Logistic(0,1)
        soft_graph = sigmoid(self.tau * (eps + self.alpha(t) * scores))

        # mask diagonal since it is explicitly not modeled
        n_vars = soft_graph.shape[-1]
        soft_graph = soft_graph.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].multiply(0.0)
        return soft_graph


    def particle_to_hard_graph(self, z, eps, t):
        """ 
        Bernoulli sample of :math:`G` using probabilities implied by latent ``z``

        Args:
            z (ndarray): a single latent tensor :math:`Z` of shape ``[d, k, 2]``
            eps (ndarray): random i.i.d. Logistic(0,1) noise  of shape ``[d, d]``
            t (int): step
        
        Returns:
            Gumbel-max (hard) sample of adjacency matrix ``[d, d]``
        """
        scores = jnp.einsum('...ik,...jk->...ij', z[..., 0], z[..., 1])

        # simply take hard limit of sigmoid in gumbel-softmax/concrete distribution
        hard_graph = ((self.tau * (eps + self.alpha(t) * scores)) > 0.0).astype(jnp.float32)

        # mask diagonal since it is explicitly not modeled
        n_vars = hard_graph.shape[-1]
        hard_graph = hard_graph.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].multiply(0.0)
        return hard_graph


    """
    Generative graph model p(G | Z)
    """

    def edge_probs(self, z, t):
        """
        Edge probabilities encoded by latent representation 

        Args:
            z (ndarray): latent tensors :math:`Z`  ``[..., d, k, 2]``
            t (int): step
        
        Returns:
            edge probabilities of shape ``[..., d, d]``
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        probs =  sigmoid(self.alpha(t) * scores)

        # mask diagonal since it is explicitly not modeled
        probs = probs.at[..., jnp.arange(probs.shape[-1]), jnp.arange(probs.shape[-1])].multiply(0.0)
        return probs

    
    def edge_log_probs(self, z, t):
        """
        Edge log probabilities encoded by latent representation

        Args:
            z (ndarray): latent tensors :math:`Z` ``[..., d, k, 2]``
            t (int): step

        Returns:
            tuple of tensors ``[..., d, d], [..., d, d]`` corresponding to ``log(p)`` and ``log(1-p)``
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        log_probs, log_probs_neg =  log_sigmoid(self.alpha(t) * scores), log_sigmoid(self.alpha(t) * -scores)

        # mask diagonal since it is explicitly not modeled
        # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
        log_probs = log_probs.at[..., jnp.arange(log_probs.shape[-1]), jnp.arange(log_probs.shape[-1])].multiply(0.0)
        log_probs_neg = log_probs_neg.at[..., jnp.arange(log_probs_neg.shape[-1]), jnp.arange(log_probs_neg.shape[-1])].multiply(0.0)
        return log_probs, log_probs_neg



    def latent_log_prob(self, single_g, single_z, t):
        """
        Log likelihood of generative graph model

        Args:
            single_g (ndarray): single graph adjacency matrix ``[d, d]``
            single_z (ndarray): single latent tensor ``[d, k, 2]``
            t (int): step
        
        Returns:
            log likelihood :math:`log p(G | Z)` of shape ``[1,]``
        """
        # [d, d], [d, d]
        log_p, log_1_p = self.edge_log_probs(single_z, t)

        # [d, d]
        log_prob_g_ij = single_g * log_p + (1 - single_g) * log_1_p

        # [1,] # diagonal is masked inside `edge_log_probs`
        log_prob_g = jnp.sum(log_prob_g_ij)

        return log_prob_g


    def eltwise_grad_latent_log_prob(self, gs, single_z, t):
        """
        Gradient of log likelihood of generative graph model w.r.t. :math:`Z`
        i.e. :math:`\\nabla_Z \\log p(G | Z)`
        Batched over samples of :math:`G` given a single :math:`Z`.

        Args:
            gs (ndarray): batch of graph matrices ``[n_graphs, d, d]``
            single_z (ndarray): latent variable ``[d, k, 2]``
            t (int): step

        Returns:
            batch of gradients of shape ``[n_graphs, d, k, 2]``
        """
        dz_latent_log_prob = grad(self.latent_log_prob, 1)
        return vmap(dz_latent_log_prob, (0, None, None), 0)(gs, single_z, t)



    """
    Estimators for scores of log p(theta, D | Z) 
    """

    def eltwise_log_joint_prob(self, gs, single_theta, rng):
        """
        Joint likelihood :math:`\\log p(\\Theta, D | G)` batched over samples of :math:`G`

        Args:
            gs (ndarray): batch of graphs ``[n_graphs, d, d]``
            single_theta (Any): single parameter PyTree
            rng (ndarray): for mini-batching ``x`` potentially

        Returns:
            batch of logprobs of shape ``[n_graphs, ]``
        """

        return vmap(self.log_joint_prob, (0, None, None, None), 0)(gs, single_theta, self.x, rng)

    

    def log_joint_prob_soft(self, single_z, single_theta, eps, t, subk):
        """
        This is the composition of :math:`\\log p(\\Theta, D | G) `
        and :math:`G(Z, U)`  (Gumbel-softmax graph sample given :math:`Z`)

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``
            single_theta (Any): single parameter PyTree
            eps (ndarray): i.i.d Logistic noise of shape ``[d, d]``
            t (int): step
            subk (ndarray): rng key

        Returns:
            logprob of shape ``[1, ]``

        """
        soft_g_sample = self.particle_to_soft_graph(single_z, eps, t)
        return self.log_joint_prob(soft_g_sample, single_theta, self.x, subk)
    

    #
    # Estimators for score d/dZ log p(theta, D | Z)   
    # (i.e. w.r.t the latent embeddings Z for graph G)
    #

    def eltwise_grad_z_likelihood(self,  zs, thetas, baselines, t, subkeys):
        """
        Computes batch of estimators for score :math:`\\nabla_Z \\log p(\\Theta, D | Z)`
        Selects corresponding estimator used for the term :math:`\\nabla_Z E_{p(G|Z)}[ p(\\Theta, D | G) ]`
        and executes it in batch.

        Args:
            zs (ndarray): batch of latent tensors :math:`Z` ``[n_particles, d, k, 2]``
            thetas (Any): batch of parameters PyTree with ``n_particles`` as leading dim
            baselines (ndarray): array of score function baseline values of shape ``[n_particles, ]``

        Returns:
            tuple batch of (gradient estimates, baselines) of shapes ``[n_particles, d, k, 2], [n_particles, ]``
        """

        # select the chosen gradient estimator
        if self.grad_estimator_z == 'score':
            grad_z_likelihood = self.grad_z_likelihood_score_function

        elif self.grad_estimator_z == 'reparam':
            grad_z_likelihood = self.grad_z_likelihood_gumbel

        else:
            raise ValueError(f'Unknown gradient estimator `{self.grad_estimator_z}`')

        # vmap
        return vmap(grad_z_likelihood, (0, 0, 0, None, 0), (0, 0))(zs, thetas, baselines, t, subkeys)



    def grad_z_likelihood_score_function(self, single_z, single_theta, single_sf_baseline, t, subk):
        """
        Score function estimator (aka REINFORCE) for the score :math:`\\nabla_Z \\log p(\\Theta, D | Z)`
        Uses the same :math:`G \\sim p(G | Z)` samples for expectations in numerator and denominator.

        This does not use :math:`\\nabla_G \\log p(\\Theta, D | G)` and is hence applicable when
        the gradient w.r.t. the adjacency matrix is not defined (as e.g. for the BGe score).

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``
            single_theta (Any): single parameter PyTree
            single_sf_baseline (ndarray): ``[1, ]``
            t (int): step
            subk (ndarray): rng key
        
        Returns:
            tuple of gradient, baseline  ``[d, k, 2], [1, ]``

        """

        # [d, d]
        p = self.edge_probs(single_z, t)
        n_vars, n_dim = single_z.shape[0:2]

        # [n_grad_mc_samples, d, d]
        subk, subk_ = random.split(subk)
        g_samples = self.sample_g(p, subk_, self.n_grad_mc_samples)

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # [n_grad_mc_samples, ] 
        subk, subk_ = random.split(subk)
        logprobs_numerator = self.eltwise_log_joint_prob(g_samples, single_theta, subk_)
        logprobs_denominator = logprobs_numerator

        # variance_reduction
        logprobs_numerator_adjusted = lax.cond(
            self.score_function_baseline <= 0.0,
            lambda _: logprobs_numerator,
            lambda _: logprobs_numerator - single_sf_baseline,
            operand=None)

        # [d * k * 2, n_grad_mc_samples]
        grad_z = self.eltwise_grad_latent_log_prob(g_samples, single_z, t)\
            .reshape(self.n_grad_mc_samples, n_vars * n_dim * 2)\
            .transpose((1, 0))

        # stable computation of exp/log/divide
        # [d * k * 2, ]  [d * k * 2, ]
        log_numerator, sign = logsumexp(a=logprobs_numerator_adjusted, b=grad_z, axis=1, return_sign=True)

        # []
        log_denominator = logsumexp(logprobs_denominator, axis=0)

        # [d * k * 2, ]
        stable_sf_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))

        # [d, k, 2]
        stable_sf_grad_shaped = stable_sf_grad.reshape(n_vars, n_dim, 2)

        # update baseline
        single_sf_baseline = (self.score_function_baseline * logprobs_numerator.mean(0) +
                            (1 - self.score_function_baseline) * single_sf_baseline)

        return stable_sf_grad_shaped, single_sf_baseline
        


    def grad_z_likelihood_gumbel(self, single_z, single_theta, single_sf_baseline, t, subk):
        """
        Reparameterization estimator for the score  :math:`\\nabla_Z \\log p(\\Theta, D | Z)`
        sing the Gumbel-softmax / concrete distribution reparameterization trick.
        Uses the same :math:`G \\sim p(G | Z)` samples for expectations in numerator and denominator.

        This **does** require a well-defined gradient
        :math:`\\nabla_G \\log p(\\Theta, D | G)` and is hence not applicable when
        the gradient w.r.t. the adjacency matrix is not defined for Gumbel-relaxations
        of the discrete adjacency matrix.
        Any (marginal) likelihood expressible as a function of ``g[:, j]`` and ``theta`` ,
        e.g. using the vector of (possibly soft) parent indicators as a mask, satisfies this.

        Examples are: ``dibs.models.LinearGaussian`` and ``dibs.models.DenseNonlinearGaussian``
        See also e.g. http://proceedings.mlr.press/v108/zheng20a/zheng20a.pdf

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``
            single_theta (Any): single parameter PyTree
            single_sf_baseline (ndarray): ``[1, ]``
            t (int): step
            subk (ndarray): rng key

        Returns:
            tuple of gradient, baseline  ``[d, k, 2], [1, ]``


        """   
        n_vars = single_z.shape[0]

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # sample Logistic(0,1) as randomness in reparameterization
        subk, subk_ = random.split(subk)
        eps = random.logistic(subk_, shape=(self.n_grad_mc_samples, n_vars, n_vars))                

        # [n_grad_mc_samples, ]
        # since we don't backprop per se, it leaves us with the option of having
        # `soft` and `hard` versions for evaluating the non-grad p(.))
        subk, subk_ = random.split(subk)
       
        # [d, k, 2], [d, d], [n_grad_mc_samples, d, d], [1,], [1,] -> [n_grad_mc_samples]
        logprobs_numerator = vmap(self.log_joint_prob_soft, (None, None, 0, None, None), 0)(single_z, single_theta, eps, t, subk_) 
        logprobs_denominator = logprobs_numerator

        # [n_grad_mc_samples, d, k, 2]
        # d/dx log p(theta, D | G(x, eps)) for a batch of `eps` samples
        # use the same minibatch of data as for other log prob evaluation (if using minibatching)
        
        # [d, k, 2], [d, d], [n_grad_mc_samples, d, d], [1,], [1,] -> [n_grad_mc_samples, d, k, 2]
        grad_z = vmap(grad(self.log_joint_prob_soft, 0), (None, None, 0, None, None), 0)(single_z, single_theta, eps, t, subk_)

        # stable computation of exp/log/divide
        # [d, k, 2], [d, k, 2]
        log_numerator, sign = logsumexp(a=logprobs_numerator[:, None, None, None], b=grad_z, axis=0, return_sign=True)

        # []
        log_denominator = logsumexp(logprobs_denominator, axis=0)

        # [d, k, 2]
        stable_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))

        return stable_grad, single_sf_baseline


    #
    # Estimators for score d/dtheta log p(theta, D | Z) 
    # (i.e. w.r.t the conditional distribution parameters)
    #

    def eltwise_grad_theta_likelihood(self, zs, thetas, t, subkeys):
        """
        Computes batch of estimators for the score   :math:`\\nabla_{\\Theta} \\log p(\\Theta, D | Z)`,
        i.e. w.r.t the conditional distribution parameters.
        Uses the same :math:`G \\sim p(G | Z)` samples for expectations in numerator and denominator.

        This does not use :math:`\\nabla_G \\log p(\\Theta, D | G)` and is hence applicable when
        the gradient w.r.t. the adjacency matrix is not defined (as e.g. for the BGe score).
        Analogous to ``eltwise_grad_z_likelihood`` but gradient w.r.t :math:`\\Theta` instead of :math:`Z`

        Args:
            zs (ndarray): batch of latent tensors Z of shape ``[n_particles, d, k, 2]``
            thetas (Any): batch of parameter PyTree with ``n_mc_samples`` as leading dim

        Returns:
            batch of gradients in form of ``thetas`` PyTree with ``n_particles`` as leading dim

        """
        return vmap(self.grad_theta_likelihood, (0, 0, None, 0), 0)(zs, thetas, t, subkeys)


    def grad_theta_likelihood(self, single_z, single_theta, t, subk):
        """
        Computes Monte Carlo estimator for the score  :math:`\\nabla_{\\Theta} \\log p(\\Theta, D | Z)`

        Uses hard samples of :math:`G`, but a soft reparameterization like for :math:`\\nabla_Z` is also possible.
        Uses the same :math:`G \\sim p(G | Z)` samples for expectations in numerator and denominator.

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``
            single_theta (Any): single parameter PyTree
            t (int): step
            subk (ndarray): rng key

        Returns:
            parameter gradient PyTree

        """

        # [d, d]
        p = self.edge_probs(single_z, t)

        # [n_grad_mc_samples, d, d]
        g_samples = self.sample_g(p, subk, self.n_grad_mc_samples)

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # [n_mc_numerator, ] 
        subk, subk_ = random.split(subk)
        logprobs_numerator = self.eltwise_log_joint_prob(g_samples, single_theta, subk_)
        logprobs_denominator = logprobs_numerator

        # PyTree  shape of `single_theta` with additional leading dimension [n_mc_numerator, ...]
        # d/dtheta log p(theta, D | G) for a batch of G samples
        # use the same minibatch of data as for other log prob evaluation (if using minibatching)
        grad_theta_log_joint_prob = grad(self.log_joint_prob, 1)
        grad_theta = vmap(grad_theta_log_joint_prob, (0, None, None, None), 0)(g_samples, single_theta, self.x, subk_)

        # stable computation of exp/log/divide and PyTree compatible
        # sums over MC graph samples dimension to get MC gradient estimate of theta
        # original PyTree shape of `single_theta`
        log_numerator = tree_map(
            lambda leaf_theta: 
                logsumexp(a=expand_by(logprobs_numerator, leaf_theta.ndim - 1), b=leaf_theta, axis=0, return_sign=True)[0], 
            grad_theta)

        # original PyTree shape of `single_theta`
        sign = tree_map(
            lambda leaf_theta:
                logsumexp(a=expand_by(logprobs_numerator, leaf_theta.ndim - 1), b=leaf_theta, axis=0, return_sign=True)[1], 
            grad_theta)

        # []
        log_denominator = logsumexp(logprobs_denominator, axis=0)

        # original PyTree shape of `single_theta`
        stable_grad = tree_map(
            lambda sign_leaf_theta, log_leaf_theta: 
                (sign_leaf_theta * jnp.exp(log_leaf_theta - jnp.log(n_mc_numerator) - log_denominator + jnp.log(n_mc_denominator))), 
            sign, log_numerator)

        return stable_grad


    """
    Estimators for score d/dZ log p(Z) 
    """
    def constraint_gumbel(self, single_z, single_eps, t):
        """ 
        Evaluates continuous acyclicity constraint using 
        Gumbel-softmax instead of Bernoulli samples

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``
            single_eps (ndarray): i.i.d. Logistic noise of shape ``[d, d``] for Gumbel-softmax
            t (int): step
        
        Returns:
            constraint value of shape ``[1,]``
        """
        n_vars = single_z.shape[0]
        G = self.particle_to_soft_graph(single_z, single_eps, t)
        h = acyclic_constr_nograd(G, n_vars)
        return h


    def grad_constraint_gumbel(self, single_z, key, t):
        """
        Reparameterization estimator for the gradient :math:`\\nabla_Z E_{p(G|Z)} [ h(G) ]`
        where :math:`h` is the acyclicity constraint penalty function.

        Since :math:`h` is differentiable w.r.t. :math:`G`, always uses
        the Gumbel-softmax / concrete distribution reparameterization trick.

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``
            key (ndarray): rng
            t (int): step

        Returns:
            gradient of shape ``[d, k, 2]``
        """
        n_vars = single_z.shape[0]
        
        # [n_mc_samples, d, d]
        eps = random.logistic(key, shape=(self.n_acyclicity_mc_samples, n_vars, n_vars))

        # [n_mc_samples, d, k, 2]
        mc_gradient_samples = vmap(grad(self.constraint_gumbel, 0), (None, 0, None), 0)(single_z, eps, t)

        # [d, k, 2]
        return mc_gradient_samples.mean(0)


    def log_graph_prior_particle(self, single_z, t):
        """
        Computes :math:`\\log p(G)` component of :math:`\\log p(Z)`,
        i.e. not the contraint or Gaussian prior term, but the DAG belief.

        The log prior :math:`\\log p(G)` is evaluated with
        edge probabilities :math:`G_{\\alpha}(Z)` given :math:`Z`.

        Args:
            single_z (ndarray): single latent tensor ``[d, k, 2]``
            t (int): step

        Returns:
            log prior graph probability`\\log p(G_{\\alpha}(Z))`  of shape ``[1,]``
        """
        # [d, d] # masking is done inside `edge_probs`
        single_soft_g = self.edge_probs(single_z, t)

        # [1, ]
        return self.log_graph_prior(single_soft_g)


    def eltwise_grad_latent_prior(self, zs, subkeys, t):
        """
        Computes batch of estimators for the score :math:`\\nabla_Z \\log p(Z)`
        with

        :math:`\\log p(Z) = - \\beta(t) E_{p(G|Z)} [h(G)] + \\log \\mathcal{N}(Z) + \\log f(Z)`

        where :math:`h` is the acyclicity constraint and `f(Z)` is additional DAG prior factor
        computed inside ``dibs.inference.DiBS.log_graph_prior_particle``.

        Args:
            zs (ndarray): single latent tensor  ``[n_particles, d, k, 2]``
            subkeys (ndarray): batch of rng keys ``[n_particles, ...]``

        Returns:
            batch of gradients of shape ``[n_particles, d, k, 2]``

        """

        # log f(Z) term
        # [d, k, 2], [1,] -> [d, k, 2]
        grad_log_graph_prior_particle = grad(self.log_graph_prior_particle, 0)

        # [n_particles, d, k, 2], [1,] -> [n_particles, d, k, 2]
        grad_prior_z = vmap(grad_log_graph_prior_particle, (0, None), 0)(zs, t)

        # constraint term
        # [n_particles, d, k, 2], [n_particles,], [1,] -> [n_particles, d, k, 2]
        eltwise_grad_constraint = vmap(self.grad_constraint_gumbel, (0, 0, None), 0)(zs, subkeys, t)

        return - self.beta(t) * eltwise_grad_constraint \
               - zs / (self.latent_prior_std ** 2.0) \
               + grad_prior_z 
            

    def visualize_callback(self, ipython=True, save_path=None):
        """Returns callback function for visualization of particles during inference updates

        Args:
            ipython (bool): set to ``True`` when running in a jupyter notebook
            save_path (str): path to save plotted images to

        Returns:
            callback
        """

        from dibs.utils.visualize import visualize
        from dibs.graph_utils import elwise_acyclic_constr_nograd as constraint
        if ipython:
            from IPython import display

        def callback(**kwargs):
            zs = kwargs["zs"]
            gs = kwargs["dibs"].particle_to_g_lim(zs)
            probs = kwargs["dibs"].edge_probs(zs, kwargs["t"])
            if ipython:
                display.clear_output(wait=True)
            visualize(probs,  save_path=save_path, t=kwargs["t"], show=True)
            print(
                f'iteration {kwargs["t"]:6d}'
                f' | alpha {self.alpha(kwargs["t"]):6.1f}'
                f' | beta {self.beta(kwargs["t"]):6.1f}'
                f' | #cyclic {(constraint(gs, self.n_vars) > 0).sum().item():3d}'
            )
            return

        return callback