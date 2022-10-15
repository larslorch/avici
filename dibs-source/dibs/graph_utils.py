import functools 

import igraph as ig
import jax.numpy as jnp
from jax import jit, vmap


@functools.partial(jit, static_argnums=(1,))
def acyclic_constr_nograd(mat, n_vars):
    """
    Differentiable acyclicity constraint from Yu et al. (2019)
    http://proceedings.mlr.press/v97/yu19a/yu19a.pdf

    Args:
        mat (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
        n_vars (int): number of variables, to allow for ``jax.jit``-compilation

    Returns:
        constraint value ``[1, ]``
    """

    alpha = 1.0 / n_vars
    # M = jnp.eye(n_vars) + alpha * mat * mat # [original version]
    M = jnp.eye(n_vars) + alpha * mat

    M_mult = jnp.linalg.matrix_power(M, n_vars)
    h = jnp.trace(M_mult) - n_vars
    return h

elwise_acyclic_constr_nograd = jit(vmap(acyclic_constr_nograd, (0, None), 0), static_argnums=(1,))


def graph_to_mat(g):
    """Returns adjacency matrix of ``ig.Graph`` object

    Args:
        g (igraph.Graph): graph

    Returns:
        ndarray:
        adjacency matrix

    """
    return jnp.array(g.get_adjacency().data)

def mat_to_graph(mat):
    """Returns ``ig.Graph`` object for adjacency matrix

    Args:
        mat (ndarray): adjacency matrix

    Returns:
        igraph.Graph:
        graph
    """
    return ig.Graph.Weighted_Adjacency(mat.tolist())

def mat_is_dag(mat):
    """Returns ``True`` iff adjacency matrix represents a DAG

    Args:
        mat (ndarray): graph adjacency matrix

    Returns:
        bool:
        ``True`` iff ``mat`` represents a DAG
    """
    G = ig.Graph.Weighted_Adjacency(mat.tolist())
    return G.is_dag()


def adjmat_to_str(mat, max_len=40):
    """
    Converts binary adjacency matrix to human-readable string

    Args:
        mat (ndarray): graph adjacency matrix
        max_len (int): maximum length of string

    Returns:
        str:
        human readable description of edges in adjacency matrix
    """
    edges_mat = jnp.where(mat == 1)
    undir_ignore = set() # undirected edges, already printed

    def get_edges():
        for e in zip(*edges_mat):
            u, v = e
            # undirected?
            if mat[v, u] == 1:
                # check not printed yet
                if e not in undir_ignore:
                    undir_ignore.add((v, u))
                    yield (u, v, True) 
            else:
                yield (u, v, False)

    strg = '  '.join([(f'{e[0]}--{e[1]}' if e[2] else
                       f'{e[0]}->{e[1]}') for e in get_edges()])
    if len(strg) > max_len:
        return strg[:max_len] + ' ... '
    elif strg == '':
        return '<empty graph>'
    else:
        return strg
