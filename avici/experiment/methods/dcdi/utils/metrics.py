import networkx as nx
from cdt.metrics import retrieve_adjacency_matrix


def edge_errors(pred, target):
    """
    Counts all types of edge errors (false negatives, false positives, reversed edges)

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    fn, fp, rev

    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

    diff = true_labels - predictions

    rev = (((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn, fp, rev


def edge_accurate(pred, target):
    """
    Counts the number of edge in ground truth DAG, true positives and the true
    negatives

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    total_edges, tp, tn

    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

    total_edges = (true_labels).sum()

    tp = ((predictions == 1) & (predictions == true_labels)).sum()
    tn = ((predictions == 0) & (predictions == true_labels)).sum()

    return total_edges, tp, tn


def shd(pred, target):
    """
    Calculates the structural hamming distance

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    shd

    """
    return sum(edge_errors(pred, target))
