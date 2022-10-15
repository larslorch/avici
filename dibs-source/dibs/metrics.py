import jax.numpy as jnp
from jax.scipy.special import logsumexp

from dibs.utils.tree import tree_mul, tree_select
from dibs.graph_utils import elwise_acyclic_constr_nograd

from sklearn import metrics as sklearn_metrics

from typing import Any, NamedTuple


class ParticleDistribution(NamedTuple):
    """ NamedTuple for structuring sampled particles :math:`(G, \\Theta)` (or :math:`G`)
    and their assigned log probabilities

    Args:
        logp (ndarray): vector of log probabilities or weights of shape ``[M, ]``
        g (ndarray): batch of graph adjacency matrix of shape ``[M, d, d]``
        theta (ndarray): batch of parameter PyTrees with leading dimension ``M``

    """

    logp: Any
    g: Any
    theta: Any = None


def pairwise_structural_hamming_distance(*, x, y):
    """
    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
    This means, edge reversals do not double count, and that getting an undirected edge wrong only counts 1

    Args:
        x (ndarray): batch of adjacency matrices  [N, d, d]
        y (ndarray): batch of adjacency matrices  [M, d, d]

    Returns:
        matrix of shape ``[N, M]``  where elt ``i,j`` is  SHD(``x[i]``, ``y[j]``)
    """

    # all but first axis is usually used for the norm, assuming that first dim is batch dim
    assert(x.ndim == 3 and y.ndim == 3)

    # via computing pairwise differences
    pw_diff = jnp.abs(jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0))
    pw_diff = pw_diff + pw_diff.transpose((0, 1, 3, 2))

    # ignore double edges
    pw_diff = jnp.where(pw_diff > 1, 1, pw_diff)
    shd = jnp.sum(pw_diff, axis=(2, 3)) / 2

    return shd


def expected_shd(*, dist, g):
    """
    Computes expected structural hamming distance metric, defined as

    :math:`\\text{expected SHD}(p, G^*) := \\sum_G p(G | D)  \\text{SHD}(G, G^*)`

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): particle distribution
        g (ndarray): ground truth adjacency matrix of shape ``[d, d]``

    Returns: 
        expected SHD ``[1, ]``
    """
    n_vars = g.shape[0]

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # score as "wrong on every edge"
        return n_vars * (n_vars - 1) / 2
    
    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])
    
    # compute shd for each graph
    shds = pairwise_structural_hamming_distance(x=particles, y=g[None]).squeeze(1)

    # expected SHD = sum_G p(G) SHD(G)
    log_expected_shd, log_expected_shd_sgn = logsumexp(
        log_weights, b=shds.astype(log_weights.dtype), axis=0, return_sign=True)

    eshd = log_expected_shd_sgn * jnp.exp(log_expected_shd)
    return eshd


def expected_edges(*, dist):
    """
    Computes expected number of edges, defined as

    :math:`\\text{expected edges}(p) := \\sum_G p(G | D)  |\\text{edges}(G)|`

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): particle distribution

    Returns:
        expected number of edges ``[1, ]``
    """

    n_vars = dist.g.shape[-1]

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # if no acyclic graphs, count the edges of the cyclic graphs; more consistent 
        n_edges_cyc = dist.g.sum(axis=(-1, -2))
        log_expected_edges_cyc, log_expected_edges_cyc_sgn = logsumexp(
            dist.logp, b=n_edges_cyc.astype(dist.logp.dtype), axis=0, return_sign=True)

        expected_edges_cyc = log_expected_edges_cyc_sgn * jnp.exp(log_expected_edges_cyc)
        return expected_edges_cyc
    
    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])
    
    # count edges for each graph
    n_edges = particles.sum(axis=(-1, -2))

    # expected edges = sum_G p(G) edges(G)
    log_expected_edges, log_expected_edges_sgn = logsumexp(
        log_weights, b=n_edges.astype(log_weights.dtype), axis=0, return_sign=True)

    edges = log_expected_edges_sgn * jnp.exp(log_expected_edges)
    return edges


def threshold_metrics(*, dist, g):
    """
    Computes various threshold metrics (e.g. ROC, precision-recall, ...)

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): sampled particle distribution
        g (ndarray): ground truth adjacency matrix of shape ``[d, d]``

    Returns:
        dict of metrics
    """
    n_vars = g.shape[0]
    g_flat = g.reshape(-1)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # score as random/junk classifier
        # for AUROC: 0.5
        # for precision-recall: no. true edges/ no. possible edges
        return {
            'roc_auc': 0.5,
            'prc_auc': (g.sum() / (n_vars * (n_vars - 1))).item(),
            'ave_prec': (g.sum() / (n_vars * (n_vars - 1))).item(),
        }

    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])

    # P(G_ij = 1) = sum_G w_G 1[G = G] in log space
    log_edge_belief, log_edge_belief_sgn = logsumexp(
        log_weights[..., jnp.newaxis, jnp.newaxis], 
        b=particles.astype(log_weights.dtype), 
        axis=0, return_sign=True)

    # L1 edge error
    p_edge = log_edge_belief_sgn * jnp.exp(log_edge_belief)
    p_edge_flat = p_edge.reshape(-1)

    # threshold metrics 
    fpr_, tpr_, _ = sklearn_metrics.roc_curve(g_flat, p_edge_flat)
    roc_auc_ = sklearn_metrics.auc(fpr_, tpr_)
    precision_, recall_, _ = sklearn_metrics.precision_recall_curve(g_flat, p_edge_flat)
    prc_auc_ = sklearn_metrics.auc(recall_, precision_)
    ave_prec_ = sklearn_metrics.average_precision_score(g_flat, p_edge_flat)
    
    return {
        'fpr': fpr_.tolist(),
        'tpr': tpr_.tolist(),
        'roc_auc': roc_auc_,
        'precision': precision_.tolist(),
        'recall': recall_.tolist(),
        'prc_auc': prc_auc_,
        'ave_prec': ave_prec_,
    }


def neg_ave_log_marginal_likelihood(*, dist, eltwise_log_marginal_likelihood, x):
    """
    Computes neg. ave log marginal likelihood for a marginal posterior over :math:`G`, defined as

    :math:`\\text{neg. MLL}(p, G^*) := - \\sum_G p(G | D)  p(D^{\\text{test}} | G)`

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): particle distribution
        eltwise_log_marginal_likelihood (callable):
            function evaluting the marginal log likelihood :math:`p(D | G)` for a batch of graph samples given
            a data set of held-out observations;
            must satisfy the signature
            ``[:, d, d], [N, d] -> [:,]``
        x (ndarray): held-out observations of shape ``[N, d]``

    Returns:
        neg. ave log marginal likelihood metric of shape ``[1,]``
    """
    n_ho_observations, n_vars = x.shape

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        g = jnp.zeros((1, n_vars, n_vars), dtype=dist.g.dtype)
        log_weights = jnp.array([0.0], dtype=dist.logp.dtype)

    else:
        g = dist.g[is_dag, :, :]
        log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])
        
    log_likelihood = eltwise_log_marginal_likelihood(g, x)

     # - sum_G p(G | D) log(p(x | G))
    log_score, log_score_sgn = logsumexp(
        log_weights, b=log_likelihood, axis=0, return_sign=True)
    score = - log_score_sgn * jnp.exp(log_score)
    return score


def neg_ave_log_likelihood(*, dist, eltwise_log_likelihood, x):
    """
    Computes neg. ave log likelihood for a joint posterior over :math:`(G, \\Theta)`, defined as

    :math:`\\text{neg. LL}(p, G^*) := - \\sum_G \\int_{\\Theta} p(G, \\Theta | D)  p(D^{\\text{test}} | G, \\Theta)`

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): particle distribution
        eltwise_log_likelihood (callable):
            function evaluting the log likelihood :math:`p(D | G, \\Theta)` for a batch of graph samples given
            a data set of held-out observations;
            must satisfy the signature
            ``[:, d, d], PyTree(leading dim :), [N, d] -> [:,]``
        x (ndarray): held-out observations of shape ``[N, d]``

    Returns:
        neg. ave log likelihood metric of shape ``[1,]``
    """
    assert dist.theta is not None
    n_ho_observations, n_vars = x.shape

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        g = tree_mul(dist.g, 0.0)
        theta = tree_mul(dist.theta, 0.0)
        log_weights = tree_mul(dist.logp, 0.0)

    else:
        g = dist.g[is_dag, :, :]
        theta = tree_select(dist.theta, is_dag)
        log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])
        
    log_likelihood = eltwise_log_likelihood(g, theta, x)

    # - sum_G p(G, theta | D) log(p(x | G, theta))
    log_score, log_score_sgn = logsumexp(
        log_weights, b=log_likelihood, axis=0, return_sign=True)
    score = - log_score_sgn * jnp.exp(log_score)
    return score


