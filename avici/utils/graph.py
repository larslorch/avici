import igraph as ig
import networkx as nx
import numpy as onp


class InvalidCPDAGError(Exception):
    # raised when a "CPDAG" returned by a learning alg does not admit a random extension
    pass


def is_acyclic(mat):
    return ig.Graph.Weighted_Adjacency(mat.tolist()).is_dag()


def adjmat_to_str(mat, max_len=40):
    """
    Converts {0,1}-adjacency matrix to human-readable string
    """

    edges_mat = onp.where(mat == 1)
    undir_ignore = set() # undirected edges, already printed

    def get_edges():
        for e in zip(*edges_mat):
            u, v = e
            # undirected?
            if mat[v, u] == 1:
                # check not printed yet
                if e not in undir_ignore:
                    undir_ignore.add((v, u))
                    yield u, v, True
            else:
                yield u, v, False

    strg = '  '.join([(f'{e[0]}--{e[1]}' if e[2] else f'{e[0]}->{e[1]}') for e in get_edges()])
    if len(strg) > max_len:
        return strg[:max_len] + ' ... '
    elif strg == '':
        return '<empty graph>'
    else:
        return strg


def nx_adjacency(g):
    return onp.array(nx.adj_matrix(g).toarray())


def random_consistent_expansion(*, rng, cpdag):
    '''
    Generates a "consistent extension" DAG of a CPDAG as defined by
    https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf
    i.e. a graph where DAG and CPDAG have the same skeleton and v-structures
    and every directed edge in the CPDAG has the same direction in the DAG
    This is achieved using the algorithm of
    http://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf
    Every DAG in the MEC is a consistent extension of the corresponding CPDAG.
    Arguments:
        rng
        cpdag:  [n_vars, n_vars]
                adjacency matrix of a CPDAG;
                breaks if it is not a valid CPDAG (merely a PDAG)
                (i.e. if cannot be extended to a DAG, e.g. undirected ring graph)

    Returns:
        [n_vars, n_vars] : adjacency matrix of a DAG
    '''

    # check whether there are any undirected edges at all
    if onp.sum((cpdag == cpdag.T) & (cpdag == 1)) == 0:
        return cpdag

    G = cpdag.copy()
    A = cpdag.copy()

    N = A.shape[0]
    n_left = A.shape[0]
    node_exists = onp.ones(A.shape[0])

    ordering = rng.permutation(N)

    while n_left > 0:

        # find i satisfying:
        #   1) no directed edge leaving i (i.e. sink)
        #   2) undirected edge (i, j) must have j adjacent to all adjacent nodes of i
        #      (to avoid forming new v-structures when directing j->i)
        # If a valid CPDAG is input, then such an i must always exist, as every DAG in the MEC of a CPDAG is a consistent extension

        found_any_valid_candidate = False
        for i in ordering:

            if node_exists[i] == 0:
                continue

            # no outgoing _directed_ edges: (i,j) doesn't exist, or, (j,i) also does
            directed_i_out = A[i, :] == 1
            directed_i_in = A[:, i] == 1

            is_sink = onp.all((1 - directed_i_out) | directed_i_in)
            if not is_sink:
                continue

            # for each undirected neighbor j of sink i
            i_valid_candidate = True
            undirected_neighbors_i = (directed_i_in == 1) & (directed_i_out == 1)
            for j in onp.where(undirected_neighbors_i)[0]:

                # check that adjacents of i are a subset of adjacents j
                # i.e., check that there is no adjacent of i (ingoring j) that is not adjacent to j
                adjacents_j = (A[j, :] == 1) | (A[:, j] == 1)
                is_not_j = onp.arange(N) != j
                if onp.any(directed_i_in & (1 - adjacents_j) & is_not_j):
                    i_valid_candidate = False
                    break

            # i is valid, orient all edges towards i in consistent extension
            # and delete i and all adjacent egdes
            if i_valid_candidate:
                found_any_valid_candidate = True

                # to orient G towards i, delete (oppositely directed) i,j edges from adjacency
                # G = index_update(G, index[i, jnp.where(undirected_neighbors_i)], 0)
                G[i, onp.where(undirected_neighbors_i)] = 0

                # remove i in A
                # A = index_update(A, index[i, :], 0)
                # A = index_update(A, index[:, i], 0)
                A[i, :] = 0
                A[:, i] = 0

                # node_exists = index_update(node_exists, index[i], 0)
                node_exists[i] = 0

                n_left -= 1

                break

        if not found_any_valid_candidate:
            err_msg = (
                'Unable to create random consistent extension of CPDAG because non-chordal: ' + adjmat_to_str(cpdag) +
                ' | G: ' + adjmat_to_str(G) +
                ' | A: ' + adjmat_to_str(A) +
                ' | ordering : ' + str(ordering.tolist())
            )
            raise InvalidCPDAGError(err_msg)

    return G