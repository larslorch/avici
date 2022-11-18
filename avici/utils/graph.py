import igraph as ig
import numpy as onp

def graph_to_mat(g):
    """Returns adjacency matrix of ig.Graph object """
    return onp.array(g.get_adjacency().data).astype(int)

def mat_to_graph(mat):
    """Returns ig.Graph object for adjacency matrix """
    return ig.Graph.Weighted_Adjacency(mat.tolist())

def graph_to_toporder(g):
    """Returns adjacency matrix of ig.Graph object """
    return onp.array(g.topological_sorting()).astype(int)

def mat_to_toporder(mat):
    """Returns adjacency matrix of ig.Graph object """
    return onp.array(mat_to_graph(mat).topological_sorting()).astype(int)