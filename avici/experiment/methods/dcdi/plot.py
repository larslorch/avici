# coding=utf-8
"""
GraN-DAG

Copyright © 2019 Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import os
import torch

# To avoid displaying the figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_weighted_adjacency(weighted_adjacency, gt_adjacency, exp_path, name="abs-weight-product", mus=None,
                            gammas=None, iter_=None, first_stop=0,
                            second_stop=0, plotting_callback=None):
    """iter is useful to deal with jacobian, it will interpolate."""
    num_vars = weighted_adjacency.shape[1]
    max_value = 0
    fig, ax1 = plt.subplots()

    # Plot weight of incorrect edges
    for i in range(num_vars):
        for j in range(num_vars):
            if gt_adjacency[i, j]:
                continue
            else:
                color = 'r'
            y = weighted_adjacency[:, i, j]
            num_iter = len(y)
            if iter_ is not None and len(y) > 1:
                num_iter = iter_ + 1
                y = np.interp(np.arange(iter_ + 1), np.linspace(0, iter_, num=len(y), dtype=int), y)
            ax1.plot(range(num_iter), y, color, linewidth=1)
            if len(y) > 0:
                max_value = max(max_value, np.max(y))

    # Plot weight of correct edges
    for i in range(num_vars):
        for j in range(num_vars):
            if gt_adjacency[i, j]:
                color = 'g'
            else:
                continue
            y = weighted_adjacency[:, i, j]
            num_iter = len(y)
            if iter_ is not None and len(y) > 1:
                num_iter = iter_ + 1
                y = np.interp(np.arange(iter_ + 1), np.linspace(0, iter_, num=len(y), dtype=int), y)
            ax1.plot(range(num_iter), y, color, linewidth=1)
            if len(y) > 0:
                max_value = max(max_value, np.max(y))

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(name)
    if first_stop > 0:
        ax1.axvline(x=first_stop, linestyle='--', color='black')
    if second_stop > 0:
        ax1.axvline(x=second_stop, linestyle='--', color='black')

    if mus is not None or gammas is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'$\frac{\mu}{2}$ and $\gamma$', color='blue')
        if mus is not None:
            ax2.plot(range(len(mus)), 0.5 * np.array(mus), color='blue', linestyle="dashed", linewidth=1,
                     label=r"$\frac{\mu}{2}$")
        if gammas is not None:
            ax2.plot(range(len(gammas)), gammas, color='blue', linestyle="dotted", linewidth=1, label=r"$\gamma$")
        ax2.legend()
        ax2.set_yscale("log")
        ax2.tick_params(axis='y', labelcolor='blue')

    fig.tight_layout()
    if plotting_callback is not None:
        plotting_callback("weighted_adjacency", fig)
    fig.savefig(os.path.join(exp_path, name + '.png'))
    fig.clf()


def plot_adjacency(adjacency, gt_adjacency, exp_path, name=''):
    """
    Plot side by side: 1)the learned adjacency matrix, 2)the ground truth adj
    matrix and 3)the difference of these matrices
    :param np.ndarray adjacency: learned adjacency matrix
    :param np.ndarray gt_adjacency: ground truth adjacency matrix
    :param str exp_path: path where to save the image
    :param str name: additional suffix to add to the image name
    """
    plt.clf()
    _, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)
    sns.heatmap(adjacency, ax=ax1, cbar=False, vmin=-1, vmax=1,
                cmap="Blues_r", xticklabels=False, yticklabels=False)
    sns.heatmap(gt_adjacency, ax=ax2, cbar=False, vmin=-1, vmax=1,
                cmap="Blues_r", xticklabels=False, yticklabels=False)
    sns.heatmap(adjacency - gt_adjacency, ax=ax3, cbar=False, vmin=-1, vmax=1,
                cmap="Blues_r", xticklabels=False, yticklabels=False)

    ax1.set_title("Learned")
    ax2.set_title("Ground truth")
    ax3.set_title("Learned - GT")

    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(exp_path, 'adjacency' + name + '.png'))


def plot_interv_w(interv_w, gt_interv, exp_path, name=''):
    """
    Plot side by side: 1)the learned intervention matrix I, 2)the ground truth
    interv matrix I* and 3)the difference of these matrices
    :param np.ndarray adjacency: learned intervention matrix
    :param np.ndarray gt_adjacency: ground truth intervention matrix
    :param str exp_path: path where to save the image
    :param str name: additional suffix to add to the image name
    """
    plt.clf()
    _, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)
    sns.heatmap(interv_w, ax=ax1, cbar=False, vmin=-1, vmax=1,
                cmap="Blues_r", xticklabels=False, yticklabels=False)
    sns.heatmap(1 - gt_interv, ax=ax2, cbar=False, vmin=-1, vmax=1,
                cmap="Blues_r", xticklabels=False, yticklabels=False)
    sns.heatmap(interv_w - (1 - gt_interv), ax=ax3, cbar=False, vmin=-1, vmax=1,
                cmap="Blues_r", xticklabels=False, yticklabels=False)

    ax1.set_title("Learned")
    ax2.set_title("Ground truth")
    ax3.set_title("Learned - GT")

    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(exp_path, 'interv_w' + name + '.png'))


def plot_learning_curves(not_nlls, aug_lagrangians, aug_lagrangians_ma,
                         aug_lagrangians_val, nlls, nlls_val, exp_path,
                         first_stop=0, second_stop=0):
    """
    Plot the learning curves (negative log-likelihood, augmented lagrangian,
    etc) for training and evaluation
    """
    fig, ax1 = plt.subplots()
    aug_lagrangians_val = np.array(aug_lagrangians_val)
    ax1.plot(range(len(nlls)), nlls, color="orange", linewidth=1, label="NLL")
    ax1.plot(aug_lagrangians_val[:, 0], nlls_val, color="blue",
             linewidth=1, label="NLL on validation set")
    ax1.plot(range(len(not_nlls)), not_nlls, color="m", linewidth=1, label="AL minus NLL")
    ax1.plot(range(len(aug_lagrangians)), aug_lagrangians, color="k", linewidth=1, alpha=0.5)
    ax1.plot(range(len(aug_lagrangians_ma)), aug_lagrangians_ma, color="k",
             linewidth=1, label="Augmented Lagrangian")
    ax1.plot(aug_lagrangians_val[:, 0], aug_lagrangians_val[:, 1], color="r", linewidth=1,
             label="Augmented Lagrangian on validation set")

    ax1.set_xlabel("Iterations")
    ax1.set_ylim(bottom=0, top=2)

    ax1.legend()
    if first_stop > 0:
        ax1.axvline(x=first_stop, linestyle='--', color='black')
    if second_stop > 0:
        ax1.axvline(x=second_stop, linestyle='--', color='black')

    fig.tight_layout()
    fig.savefig(os.path.join(exp_path, 'learning-curves.png'), bbox_inches="tight", padding=0)
    fig.clf()


def get_joint_density(model_, x, node, parent, mask, intervention, intervention_type,
                      intervention_knowledge, regime=0):
    """
    Given a model and data, return the density associated to each example.
    Several cases depending on the type/knowledge of intervention
    """
    weights, biases, extra_params = model_.get_parameters(mode="wbx")
    mask = mask.expand((x.size(0), -1))

    if intervention_type == "perfect" or not intervention:
        if "LearnableModel_NonLinGaussANM" in str(type(model_)):
            logp = model_.compute_log_likelihood(x, weights, biases, extra_params).detach().cpu().numpy()
        else:
            # Flow model
            density_params = model_.forward_given_params(x, weights, biases)
            model_.num_vars = 1
            logp_node = model_._log_likelihood(x[:, [node]], [density_params[node]]).detach().cpu().numpy()
            logp_parent = model_._log_likelihood(x[:, [parent]], [density_params[parent]]).detach().cpu().numpy()
    else:
        # mask = torch.ones_like(x)
        if "LearnableModel_NonLinGaussANM" in str(type(model_)):
            logp = model_.compute_log_likelihood(x, weights, biases, extra_params, mask=mask).detach().cpu().numpy()
        else:
            # Flow model
            density_params = model_.forward_given_params(x, weights, biases, mask=mask)
            model_.num_vars = 1
            logp_node = model_._log_likelihood(x[:, [node]], [density_params[node]]).detach().cpu().numpy()
            logp_parent = model_._log_likelihood(x[:, [parent]], [density_params[parent]]).detach().cpu().numpy()

    if "LearnableModel_NonLinGaussANM" in str(type(model_)):
        return logp
    else:
        return logp_node + logp_parent



def choose_node_to_plot(adj):
    """
    Given the adjacency matrix 'adj', find a node with only one parent.
    If cannot find a node with a single parent, return 0,0
    :param np.ndarray adj: adjacency matrix
    :return: node, parent
    """
    node_found = False
    for i in range(adj.shape[0]):
        if np.sum(adj[:, i]) == 1:
            node_found = True
            for j in range(adj.shape[0]):
                if adj[j, i] == 1:
                    node = i
                    parent = j
            break

    if not node_found:
        print("Could not find a node with a single parent. No density plot will be plotted.")
        return 0, 0

    return node, parent


def plot_density(model, opt, gt_adj, data, mask, node, parent, exp_path, step,
                 name, resolution=256, show_data=True, log_probas=True,
                 cmap=None, scatter_color="black", intervention=None,
                 intervention_type="perfect",
                 intervention_knowledge="known", window=5):
    node_values = np.linspace(-window, window, resolution)
    parent_values = np.linspace(-window, window, resolution)

    x = np.zeros((resolution * resolution, data.shape[1]))
    x = torch.DoubleTensor(x)
    x[:, node] = torch.DoubleTensor(np.hstack([node_values.tolist()] * resolution))
    x[:, parent] = torch.DoubleTensor(np.hstack([[pv] * resolution for i, pv in enumerate(parent_values)]))

    from copy import deepcopy
    model_ = deepcopy(model)

    # Keep only relevant edge
    model_.adjacency[...] = 0
    model_.adjacency[parent, node] = 1
    assert model_.adjacency.sum() == 1

    # Get density from model
    logp = get_joint_density(model_, x, node, parent, mask, intervention, intervention_type, intervention_knowledge)

    if logp.shape[1] > 1:
        logp = logp[:, node]
    else:
        logp = logp[:, 0]

    if log_probas:
        plot_data = logp
    else:
        plot_data = np.exp(logp)

    plt.clf()
    ax = sns.heatmap(plot_data.reshape(resolution, resolution).T,
                     square=False, cmap=cmap, vmin=-5, vmax=0)
    ax.invert_yaxis()

    if show_data:
        new_x = np.array(data[:, parent])
        new_x = (new_x + window)/(2*window) * resolution
        new_y = np.array(data[:, node])
        new_y = (new_y + window)/(2*window) * resolution
        plt.scatter(new_x, new_y, c=scatter_color, s=1)

    plt.gcf().set_size_inches(w=6, h=6)
    plt.savefig(os.path.join(exp_path, "density_%s_x%d_given_x%d.step%d.log%s.png" % (name, node, parent, step, str(log_probas))), dpi=400)


def plot_learned_density(model, opt, gt_adj, data, mask, exp_path, step, resolution=256, show_data=True,
                         log_probas=True, cmap=None, scatter_color="black", intervention=None,
                         intervention_type="perfect", intervention_knowledge="known", window=5):
    """
    Choose a pair of nodes (where a node is the unique parent of the other) and
    call plot densities for all interventional and observational context
    """

    torch.set_default_tensor_type(torch.DoubleTensor)
    hps = opt.__dict__
    assert not hps["normalize_data"]

    # Choose nodes to plot
    node, parent = choose_node_to_plot(gt_adj)
    if node == parent:
        return

    # obs data
    plot_data = data[(mask[:, node] == 1)]
    plot_mask = torch.ones((1, plot_data.size(1)))
    plot_density(model, opt, gt_adj, plot_data, plot_mask, node, parent, exp_path,
                 step, "obs", resolution, show_data, log_probas, cmap,
                 scatter_color, intervention, intervention_type,
                 intervention_knowledge, window)


    if intervention is not None:
        plot_data = data[(mask[:, parent] == 1) & (mask[:, node] == 0)]
        plot_mask = torch.ones((1, plot_data.size(1)))
        plot_mask[0, node] = 0
        plot_density(model, opt, gt_adj, plot_data, plot_mask, node, parent, exp_path,
                     step, "interv_child", resolution, show_data, log_probas, cmap,
                     scatter_color, intervention, intervention_type,
                     intervention_knowledge, window)


def plot_learning_curves_retrain(losses, losses_val, nlls, nlls_val, exp_path):
    fig, ax1 = plt.subplots()
    losses_val = np.array(losses_val)
    ax1.plot(range(len(nlls)), nlls, color="orange", linewidth=1, label="NLL")
    ax1.plot(losses_val[:,0], nlls_val, color="blue", linewidth=1, label="NLL (val)")
    ax1.plot(range(len(losses)), losses, color="k", linewidth=1, label="NLL + REG")
    ax1.plot(losses_val[:,0], losses_val[:,1], color="r", linewidth=1, label="NLL + REG (val)")
    ax1.set_xlabel("Iterations")
    ax1.set_ylim(bottom=0, top=2)

    ax1.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(exp_path, 'learning-curves.png'), bbox_inches="tight", padding=0)
    fig.clf()
