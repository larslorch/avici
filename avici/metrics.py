import numpy as onp
from sklearn import metrics as sklearn_metrics


def shd(g, h):
    """
    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
        - this means, edge reversals do not double count
        - this means, getting an undirected edge wrong only counts 1

    Args:
        g:  [..., d, d]
        h:  [..., d, d]
    """
    assert g.ndim == h.ndim
    abs_diff =  onp.abs(g - h)
    mistakes = abs_diff + onp.swapaxes(abs_diff, -2, -1)  # mat + mat.T (transpose of last two dims)

    # ignore double edges
    mistakes_adj = onp.where(mistakes > 1, 1, mistakes)

    return onp.triu(mistakes_adj).sum((-1, -2))


def n_edges(g):
    """
    Args:
        g:  [..., d, d]
    """
    return g.sum((-1, -2))


def is_acyclic(g):
    """
       Args:
           g:  [d, d]
       """
    n_vars = g.shape[-1]
    mat = onp.eye(n_vars) + g / n_vars
    mat_pow = onp.linalg.matrix_power(mat, n_vars)
    acyclic_constr = onp.trace(mat_pow) - n_vars
    return onp.isclose(acyclic_constr, 0.0)


def is_cyclic(g):
    """
    Args:
        g:  [d, d]
    """
    return not is_acyclic(g)


def classification_metrics(true, pred):
    """
    Args:
        true:  [...]
        pred:  [...]
    """
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)

    if onp.sum(pred_flat) > 0 and onp.sum(true_flat) > 0:
        precision, recall, f1, _ = sklearn_metrics.precision_recall_fscore_support(
            true_flat, pred_flat, average="binary")
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    elif onp.sum(pred_flat) == 0 and onp.sum(true_flat) == 0:
        # no true positives, and no positives were predicted
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    else:
        # no true positives, but we predicted some positives
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }


def threshold_metrics(true, pred):
    """
    Args:
        true:  [...]
        pred:  [...]
    """
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)

    if onp.sum(pred_flat) > 0 and onp.sum(true_flat) > 0:
        fpr, tpr, _ = sklearn_metrics.roc_curve(true_flat, pred_flat)
        precision, recall, _ = sklearn_metrics.precision_recall_curve(true_flat, pred_flat)
        ave_prec = sklearn_metrics.average_precision_score(true_flat, pred_flat)
        roc_auc = sklearn_metrics.auc(fpr, tpr)
        prc_auc = sklearn_metrics.auc(recall, precision)

        return {
            "auroc": roc_auc,
            "auprc": prc_auc,
            "ap": ave_prec,
        }

    elif onp.sum(pred_flat) == 0 and onp.sum(true_flat) == 0:
        # no true positives, and no positives were predicted
        return {
            "auroc": 1.0,
            "auprc": 1.0,
            "ap": 1.0,
        }

    else:
        # no true positives, but we predicted some positives
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "ap": 0.0,
        }


def _calibration_stats(*, y_pred, y_true, n_bins):
    assert y_true.ndim == 1 and y_pred.ndim == 1
    assert y_true.shape[0] == y_pred.shape[0]

    counts_per_bin = onp.zeros(n_bins)
    prob_pred = onp.zeros(n_bins)
    prob_true = onp.zeros(n_bins)

    bins = onp.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        # get predicted probs that fall into this bin
        mask = (lo <= y_pred) & (y_pred < hi)
        y_pred_in_bin = y_pred[mask]
        n_preds_in_bin = y_pred_in_bin.shape[0]
        if n_preds_in_bin:
            # get true labels for these predicted probs
            y_true_in_bin = y_true[mask]
            y_true_in_bin_0 = y_true_in_bin == 0
            y_true_in_bin_1 = y_true_in_bin == 1
            assert y_true_in_bin_0.sum() + y_true_in_bin_1.sum() == y_true_in_bin.shape[0]

            # compute coverage
            prob_pred[i] = y_pred_in_bin.sum() / n_preds_in_bin
            prob_true[i] = y_true_in_bin_1.sum() / n_preds_in_bin

        else:
            prob_pred[i] = onp.nan
            prob_true[i] = onp.nan

        counts_per_bin[i] = n_preds_in_bin

    nan_mask = onp.isnan(prob_pred) | onp.isnan(prob_true)
    prob_pred = prob_pred[~nan_mask]
    prob_true = prob_true[~nan_mask]
    counts_per_bin = counts_per_bin[~nan_mask]

    assert prob_pred.shape == prob_true.shape
    assert prob_pred.shape == counts_per_bin.shape

    return prob_pred, prob_true, counts_per_bin, bins


def make_calibration_stats(true_inp, pred_inp, n_bins=10):

    # convert list into one flattened array
    if type(true_inp) == list and type(pred_inp) == list:
        assert len(true_inp) == len(pred_inp)
    else:
        true_inp = [true_inp]
        pred_inp = [pred_inp]

    y_true_all = onp.concatenate(true_inp).reshape(-1)
    y_pred_all = onp.concatenate(pred_inp).reshape(-1)

    # compute calibration line using full aggregate
    prob_pred_all, prob_true_all, counts_per_bin_all, bins_all = \
        _calibration_stats(y_pred=y_pred_all, y_true=y_true_all, n_bins=n_bins)

    # compute expected calibration error (ece) and brier score, batches of datasets to compute error
    eces, briers = [], []
    for y_true, y_pred in zip(true_inp, pred_inp):
        y_pred_single = y_pred.reshape(-1)
        y_true_single = y_true.reshape(-1)
        prob_pred_single, prob_true_single, counts_per_bin_single, _ = \
            _calibration_stats(y_pred=y_pred_single, y_true=y_true_single, n_bins=n_bins)

        # ece https://arxiv.org/pdf/1706.04599.pdf
        ece = onp.sum(onp.abs(prob_true_single - prob_pred_single) * counts_per_bin_single / y_pred_single.shape[0])
        eces.append(ece)

        # brier
        brier = onp.mean((y_pred_single - y_true_single) ** 2.0)
        briers.append(brier)

    return dict(
        prob_pred=prob_pred_all,
        prob_true=prob_true_all,
        counts_per_bin=counts_per_bin_all,
        y_pred=y_pred_all,
        bins=bins_all,
        eces=eces,
        briers=briers,
    )