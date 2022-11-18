import math
import igraph as ig
import numpy as onp
import random as pyrandom
import pandas as pd

from avici.definitions import ROOT_DIR, GRAPH_YEAST, GRAPH_ECOLI
from avici.utils.graph import mat_to_graph, graph_to_mat
from avici.synthetic import GraphModel


def _module_extraction(rng, source, d, topk=0.2, at_least_n_regulators=1, make_acyclic=True, drop_self_edges=True,
                       return_node_indices=False):
    """
    Submodule extraction method by Marbach et al. (2009)
    https://www.liebertpub.com/doi/pdfplus/10.1089/cmb.2008.09TT

    Chooses a random node from the source network as seed and then adds
    a neighboring node (and all edges connecting the nodes in the source network) to the subnetwork randomly among the
    top k-percent "modular" neighbors.

    Important points:
      - The returned network contains all edges from the directed source network covered by the selected nodes
      - We consider the *undirected* skeleton of the source network for the modularity and neighborhood computation
        and ultimately selecting our subset of nodes
        (There is no definition of neighborhood/degree in the paper by Marbach et al (2009), which would require
        further explanation for a directed graph, and the original source code in GNW appears to only
        considers an undirected graph)

    Reference implementation:
    https://github.com/tschaffter/genenetweaver/blob/c5310349f5d5723306585c2bb62aedbdeb70db46/src/ch/epfl/lis/gnw/SubnetExtractor.java#L261
    https://github.com/tschaffter/jmod/blob/e2b230781ea3c3ad00833c58bd31b3b7d820317f/src/main/java/ch/epfl/lis/jmod/modularity/ModularityDetector.java#L204
    https://github.com/tschaffter/jmod/blob/e2b230781ea3c3ad00833c58bd31b3b7d820317f/src/main/java/ch/epfl/lis/jmod/JmodNetwork.java#L118
    (jmod is not the exact library used in genenetweaver, but is by the same author and satisfies almost the same api;
    I am using this as a reference since the actual imod library code used by genenetweaver is not public)

    Args:
        rng
        source: source network of shape (n_vars, n_vars)
        d: size of network we want to extract; d < n_vars
        topk: in [0.0, 1.0] percentage of neighbor nodes sorted by "modularity" from which we randomly choose the next node
        at_least_n_regulators: number of regulators/transcription factors to include with prob 1.
            A master regulator is a node with at least one outgoing edge. For the yeast graph used here, 
            157 of the 4284 nodes are master regulators/transcription factors
        make_acyclic
        drop_self_edges
        return_node_indices
    """

    dmax = source.shape[0]
    assert d < dmax
    assert d > at_least_n_regulators
    assert 0.0 <= topk <= 1.0
    assert source.shape[0] == source.shape[1]
    assert onp.all(onp.isclose(source, 0) | onp.isclose(source, 1))

    if drop_self_edges:
        source[onp.diag_indices(dmax)] = 0

    regulators = set(onp.where(source.sum(1) > 0)[0])

    # intialize modularity matrix B; do this based the undirected version of the source network (see docstring)
    a = source | source.T
    upper_tria_a = onp.triu(a, k=1)
    k = upper_tria_a.sum(0) + upper_tria_a.sum(1)
    m = upper_tria_a.sum()
    b = a - onp.outer(k, k) / (2 * m)
    assert m == 0.5 * k.sum()

    # randomly select seed
    s = -1 * onp.ones(dmax)
    if at_least_n_regulators:
        # if at_least_n_regulators > 0, then use a regulator here
        seed = rng.choice(list(regulators))
        assert source[seed, :].sum() > 0
    else:
        seed = rng.choice(dmax)

    s[seed] = 1
    subnet = {seed}

    # add d - 1 more nodes
    for _ in range(d - 1):

        # find neighbors of s in undirected source network
        neighbors = set()
        for i in subnet:
            for j in onp.where(a[i] == 1)[0]:
                if j not in subnet:
                    # if not enough regulators added yet, only add regulators to neighborhood
                    if at_least_n_regulators > 0 and len(regulators.intersection(subnet)) < at_least_n_regulators:
                        if j in regulators:
                            neighbors.add(j)
                    else:
                        neighbors.add(j)

        # if no neighbors (subnet is an island or we require a non-neighboring regulator), set all nodes as neighbors
        # in case we still require a regulator, this ensures we will add one this or next iteration
        if len(neighbors) == 0:
            neighbors = set(range(dmax)) - subnet

        assert len(neighbors.intersection(subnet)) == 0
        assert len(neighbors) > 0

        # compute modularity resulting from adding a neighbor
        qs, qs_nodes = [], []
        bterm_left = s @ b
        for neighbor in neighbors:
            assert s[neighbor] == -1
            assert neighbor not in subnet
            bterm_left_adj = bterm_left + 2 * b[neighbor, :]
            q_prop = (bterm_left_adj @ s) + 2 * bterm_left_adj[neighbor]
            qs.append(q_prop / (4 * m))
            qs_nodes.append(neighbor)

        # add random node among topk% modular graph options
        topk_q_indices = onp.argsort(qs)[::-1][:math.floor(topk * len(neighbors)) + 1]
        topk_q_nodes = onp.array(qs_nodes)[topk_q_indices]

        u = rng.choice(topk_q_nodes)
        assert s[u] == -1
        assert u not in subnet
        s[u] = 1
        subnet.add(u)

        assert set(onp.where(s == 1)[0]) == subnet

    # sanity checks once network extracted
    assert len(subnet) == d
    assert at_least_n_regulators == 0 or len(regulators.intersection(subnet)) >= at_least_n_regulators

    # extract directed subnetwork from source network based on selected node set
    subgraph = _extract_subnetwork(source, subnet)

    if make_acyclic:
        subgraph = _break_cycles_randomly(rng, subgraph)

    if return_node_indices:
        return subgraph, subnet
    else:
        return subgraph


def _extract_subnetwork(source, nodes):
    """
    Extracts subnetwork defindes by nodes from the source network

    Args:
        source: [n, n] adjacency matrix
        nodes: set of node indices in `source` defining the subnetwork
    """
    d = len(nodes)
    subgraph = onp.zeros((d, d), dtype=source.dtype)
    idx = {k: idx for idx, k in enumerate(nodes)}
    for node_i in nodes:
        for node_j in nodes:
            subgraph[idx[node_i], idx[node_j]] = source[node_i, node_j]

    return subgraph


def _break_cycles_randomly(rng, mat):
    """
    DFS that breaks cycles at random position through a random starting point
    """
    color = [0] * mat.shape[0]

    def dfs(u):
        color[u] = 1
        for v in onp.where(mat[u, :] == 1)[0]:
            if color[v] == 1:
                # back edge, which implies a cycle; remove edge that closes the cycle
                mat[u, v] = 0
            elif color[v] == 0:
                dfs(v)
        color[u] = 2

    for s in rng.permutation(mat.shape[0]):
        if color[s] == 0:
            dfs(s)

    assert mat_to_graph(mat).is_dag()
    return mat


def orient_pdag_randomly(rng, mat):
    """
    Orient PDAG randomly as a DAG by consistently orienting undirected edges with the partial ordering
    Done by viewing undirected edges as 2-cycles that are broken randomly
    """
    orig_mat = mat.copy()
    dag = _break_cycles_randomly(rng, mat)
    assert onp.all(~(((orig_mat == 1) & (orig_mat.T == 1)) & ((dag == 0) & (dag.T == 0)))), \
        "Some undirected edges were deleted completely"
    return dag


class ErdosRenyi(GraphModel):
    """
    Erdos-Renyi random graph
    
    Args:
        edges_per_var (float): expected number of edges per node, scaled to account for number of nodes
    """
    def __init__(self, edges_per_var):
        self.edges_per_var = edges_per_var

    def __call__(self, rng, n_vars):
        # select p s.t. we get requested edges_per_var in expectation
        n_edges = self.edges_per_var * n_vars
        p = min(n_edges / ((n_vars * (n_vars - 1)) / 2), 0.99)

        # sample
        mat = rng.binomial(n=1, p=p, size=(n_vars, n_vars)).astype(int) # bernoulli

        # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
        dag = onp.tril(mat, k=-1)

        # randomly permute
        p = rng.permutation(onp.eye(n_vars).astype(int))
        dag = p.T @ dag @ p
        return dag


class ScaleFree(GraphModel):
    """
    Barabasi-Albert (scale-free)
    Power-law in-degree
    
    Args:
        edges_per_var (int): number of edges per node
        power (float): power in preferential attachment process. 
            Higher values make few nodes have high in-degree.

    """
    def __init__(self, edges_per_var, power=1.0):
        self.edges_per_var = edges_per_var
        self.power = power

    def __call__(self, rng, n_vars):
        pyrandom.seed(rng.bit_generator.state["state"]["state"]) # seed pyrandom based on state of numpy rng
        _ = rng.normal() # advance rng state by 1
        perm = rng.permutation(n_vars).tolist()
        g = ig.Graph.Barabasi(n=n_vars, m=self.edges_per_var, directed=True, power=self.power).permute_vertices(perm)
        mat = graph_to_mat(g)
        return mat


class ScaleFreeTranspose(ScaleFree):
    """
    Barabasi-Albert (scale-free)
    Power-law out-degree
    
    Args:
        edges_per_var (int): number of edges per node
        power (float): power in preferential attachment process. 
            Higher values make few nodes have high out-degree.

    """
    def __init__(self, edges_per_var, power=1):
        super().__init__(edges_per_var=edges_per_var, power=power)

    def __call__(self, rng, n_vars):
        mat = super().__call__(rng, n_vars)
        return mat.T


class WattsStrogatz(GraphModel):
    """
    Watts-Strogatz (small-world)

    Args:
        dim (int): the dimension of the lattice
        nei (int): value giving the distance (number of steps) within which two vertices will be connected.
        p (float): rewiring probability
    """
    def __init__(self, dim=2, nei=2, p=0.1):
        self.dim = dim
        self.nei = nei
        self.p = p

    def __call__(self, rng, n_vars):
        # choose size s.t. we get at smallest possible n_vars greater than requested n_vars given the dimension of lattice
        dim_size = math.ceil(n_vars ** (1.0 / self.dim))

        pyrandom.seed(rng.bit_generator.state["state"]["state"]) # seed pyrandom based on state of numpy rng
        _ = rng.normal() # advance rng state by 1
        g = ig.Graph.Watts_Strogatz(dim=self.dim, size=dim_size, nei=self.nei, p=self.p, multiple=False, loops=False)

        # drop excessive vertices s.t. we get exactly n_vars
        n_excessive = len(g.vs) - n_vars
        assert n_excessive >= 0
        if n_excessive:
            g.delete_vertices(rng.choice(g.vs, size=n_excessive, replace=False))
        assert len(g.vs) == n_vars, f"Didn't get requested graph; g.vs: {len(g.vs)}, n_vars {n_vars}"

        # make directed
        mat = onp.triu(graph_to_mat(g), k=1)

        # randomly permute
        p = rng.permutation(onp.eye(n_vars).astype(int))
        mat = p.T @ mat @ p

        assert ig.Graph.Weighted_Adjacency(mat.tolist()).is_dag()
        return mat


class SBM(GraphModel):
    """
    Stochastic Block Model

    Args:
        edges_per_var (int): expected number of edges per node
        n_blocks (int): number of blocks in model
        damp (float): if p is probability of intra block edges, damp * p is probability of inter block edges
            p is determined based on `edges_per_var`. For damp = 1.0, this is equivalent to erdos renyi
    """
    def __init__(self, edges_per_var, n_blocks, damp=0.2):
        self.edges_per_var = edges_per_var
        self.n_blocks = n_blocks
        self.damp = damp

    def __call__(self, rng, n_vars):
        assert n_vars >= self.n_blocks >= 1, \
            f"Invalid `n_blocks` ({self.n_blocks}) given `n_vars` ({n_vars})"

        # sample blocks
        splits = onp.sort(rng.choice(n_vars, size=self.n_blocks - 1, replace=False))
        blocks = onp.split(rng.permutation(n_vars), splits)
        block_sizes = onp.array([b.shape[0] for b in blocks])

        # select p s.t. we get requested edges_per_var in expectation
        block_edges_sampled = (onp.outer(block_sizes, block_sizes) - onp.diag(block_sizes)) / 2
        relative_block_probs = onp.eye(self.n_blocks) + self.damp * (1 - onp.eye(self.n_blocks))
        n_edges = self.edges_per_var * n_vars
        p = min(0.99, n_edges / onp.sum(block_edges_sampled * relative_block_probs))

        # sample graph
        mat_intra = rng.binomial(n=1, p=p, size=(n_vars, n_vars)).astype(int) # bernoulli
        mat_inter = rng.binomial(n=1, p=self.damp * p, size=(n_vars, n_vars)).astype(int) # bernoulli

        mat = onp.zeros((n_vars, n_vars))
        for i, bi in enumerate(blocks):
            for j, bj in enumerate(blocks):
                mat[onp.ix_(bi, bj)] = (mat_intra if i == j else mat_inter)[onp.ix_(bi, bj)]

        # make directed
        mat = onp.triu(mat, k=1)

        # randomly permute
        p = rng.permutation(onp.eye(n_vars).astype(int))
        mat = p.T @ mat @ p

        assert ig.Graph.Weighted_Adjacency(mat.tolist()).is_dag()
        return mat


class GRG(GraphModel):
    """
    Geometric random graph
    
    Args:
        radius (float): radius inside which random dots in a unit square are connected
    """
    def __init__(self, radius=0.1):
        self.radius = radius

    def __call__(self, rng, n_vars):
        pyrandom.seed(rng.bit_generator.state["state"]["state"])  # seed pyrandom based on state of numpy rng
        _ = rng.normal()  # advance rng state by 1
        g = ig.Graph._GRG(n=n_vars, radius=self.radius)[0]

        # make directed
        mat = onp.triu(graph_to_mat(g), k=1)

        # randomly permute
        p = rng.permutation(onp.eye(n_vars).astype(int))
        mat = p.T @ mat @ p

        assert ig.Graph.Weighted_Adjacency(mat.tolist()).is_dag()
        return mat


class Yeast(GraphModel):
    """
    Random subgraph extracted from real yeast graph
    using the extraction method https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.6733&rep=rep1&type=pdf
    
    Args:
        topk (float): heuristic in modularity extraction method introducing additional randomness.
            Percentage in [0.0, 1.0] of neighbor nodes sorted by "modularity" from which we randomly choose next node.
        at_least_n_regulators (int): number of regulators/transcription factors to include with prob 1.
             A master regulator is a node with at least one outgoing edge.
        make_acyclic (bool, optional): whether to make acyclic
    """
    def __init__(self, topk=0.2, at_least_n_regulators=1, make_acyclic=True):
        self.topk = topk
        self.at_least_n_regulators = at_least_n_regulators
        self.make_acyclic = make_acyclic

    def __call__(self, rng, n_vars):
        # load edges
        # (12873, 2)
        raw = pd.read_csv(ROOT_DIR / GRAPH_YEAST, sep="\t", index_col=False, header=None)

        # convert to numerical
        unique_genes = sorted(list(set(raw[0].unique().tolist() + raw[1].unique().tolist())))
        mapping = {k: v for v, k in enumerate(unique_genes)}
        edges = raw.applymap(lambda x: mapping[x]).to_numpy()

        g_raw = onp.zeros((len(unique_genes), len(unique_genes)), dtype=onp.int32) # shape (4441, 4441)
        g_raw[edges[:, 0], edges[:, 1]] = 1

        # extract random subnetwork
        g = _module_extraction(rng, g_raw, n_vars, topk=self.topk, at_least_n_regulators=self.at_least_n_regulators,
                               make_acyclic=self.make_acyclic, return_node_indices=False)

        # randomly permute
        p = rng.permutation(onp.eye(n_vars).astype(int))
        g_permuted = p.T @ g @ p
        return g_permuted


class Ecoli(GraphModel):
    """
    Random subgraph extracted from real e.coli graph
    using the extraction method https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.6733&rep=rep1&type=pdf
    
    Args:
        topk (float): heuristic in modularity extraction method introducing additional randomness.
            Percentage in [0.0, 1.0] of neighbor nodes sorted by "modularity" from which we randomly choose next node.
        at_least_n_regulators (int): number of regulators/transcription factors to include with prob 1.
             A master regulator is a node with at least one outgoing edge.
        make_acyclic (bool, optional): whether to make acyclic
    """
    def __init__(self, topk=0.2, at_least_n_regulators=1, make_acyclic=True):
        self.topk = topk
        self.at_least_n_regulators = at_least_n_regulators
        self.make_acyclic = make_acyclic

    def __call__(self, rng, n_vars):
        # load edges
        # (3758, 2)
        raw = pd.read_csv(ROOT_DIR / GRAPH_ECOLI, sep="\t", index_col=False, header=None)
        raw.loc[raw[2] == "+-", 2] = onp.nan
        raw.loc[raw[2] == "?", 2] = onp.nan

        # impute + or - randomly for +- and ?
        prob_pos = (raw[2] == "+").sum() / ((raw[2] == "+").sum() + (raw[2] == "-").sum())
        raw.loc[raw[2].isnull(), 2] = rng.choice(["+", "-"], p=[prob_pos, 1-prob_pos], size=raw[2].isnull().sum())
        assert set(raw[2].unique()) == {'+', '-'}

        # convert to numerical
        unique_genes = sorted(list(set(raw[0].unique().tolist() + raw[1].unique().tolist())))
        mapping = {k: v for v, k in enumerate(unique_genes)}
        edges = raw.loc[:, [0, 1]].applymap(lambda x: mapping[x]).to_numpy()
        effect_signs = raw.loc[:, [2]].applymap(lambda x: {"+": 1.0, "-": -1.0}[x]).to_numpy().squeeze(-1)
        assert onp.all(onp.isclose(effect_signs, -1.0) | onp.isclose(effect_signs, 1.0))

        g_raw = onp.zeros((len(unique_genes), len(unique_genes)), dtype=onp.int32) # shape (1565, 1565)
        effect_raw_sgn = onp.zeros((len(unique_genes), len(unique_genes)), dtype=onp.float32)
        g_raw[edges[:, 0], edges[:, 1]] = 1
        effect_raw_sgn[edges[:, 0], edges[:, 1]] = effect_signs

        # extract random subnetwork
        g, subnet = _module_extraction(rng, g_raw, n_vars, topk=self.topk, at_least_n_regulators=self.at_least_n_regulators,
                                       make_acyclic=self.make_acyclic, return_node_indices=True)

        effect_sgn = _extract_subnetwork(effect_raw_sgn, subnet)
        effect_sgn *= g.astype(onp.float32) # mask in case a cycle was broken

        # randomly permute
        p = rng.permutation(onp.eye(n_vars).astype(onp.int32))
        g_permuted = p.T @ g @ p
        # effect_sgn_permuted = p.T @ effect_sgn @ p
        return g_permuted
