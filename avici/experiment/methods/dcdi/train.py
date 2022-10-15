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
import cdt
import os
import time
import copy
import numpy as np
import torch
from cdt.utils.R import RPackages, launch_R_script

try:
    from .dag_optim import compute_dag_constraint, is_acyclic
    from .prox import monkey_patch_RMSprop
    from .utils.metrics import edge_errors, shd as shd_metric
    from .utils.penalty import compute_penalty
    from .utils.save import dump, load
except ImportError:
    from dag_optim import compute_dag_constraint, is_acyclic
    from prox import monkey_patch_RMSprop
    from utils.metrics import edge_errors, shd as shd_metric
    from utils.penalty import compute_penalty
    from utils.save import dump, load

# from .plot import plot_learned_density, plot_weighted_adjacency, plot_adjacency, plot_learning_curves, plot_interv_w, plot_learning_curves_retrain
np.set_printoptions(linewidth=200)
EPSILON = 1e-8


def compute_loss(x, mask, regime, model, weights, biases, extra_params, intervention,
                 intervention_type, intervention_knowledge, mean_std=False):
    # TODO: add param
    """
    Compute the loss. If intervention is perfect and known, remove
    the intervened targets from the loss with a mask.
    """
    if intervention and intervention_type == "perfect" and intervention_knowledge =="known":
        log_likelihood = model.compute_log_likelihood(x, weights, biases, extra_params)
        log_likelihood = torch.sum(log_likelihood * mask, dim=0) / mask.size(0)
    else:
        log_likelihood = model.compute_log_likelihood(x, weights, biases,
                                                  extra_params, mask=mask,
                                                  regime=regime)
        log_likelihood = torch.sum(log_likelihood, dim=0) / mask.size(0)
    loss = - torch.mean(log_likelihood)

    if not mean_std:
        return loss
    else:
        joint_log_likelihood = torch.mean(log_likelihood * mask, dim=1)
        return loss, torch.sqrt(torch.var(joint_log_likelihood) / joint_log_likelihood.size(0))


def train(model, gt_interv, train_data, test_data, opt, metrics_callback, plotting_callback):
    """
    Applying augmented Lagrangian to solve the continuous constrained problem.
    """
    first_stop = 0
    second_stop = 0
    thresholded = False

    patience = opt.train_patience
    patience_thresh = opt.train_patience_post
    best_nll_val = np.inf
    best_lagrangian_val = np.inf

    # # Prepare path for saving results
    # save_path = os.path.join(opt.exp_path, "train")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # # Check if already computed
    # if os.path.exists(os.path.join(save_path, "DAG.npy")):
    #     print("Train already computed. Loading result from disk.")
    #     return load(save_path, "model.pkl")

    time0 = time.time()

    # initialize stuff for learning loop
    aug_lagrangians = []
    aug_lagrangian_ma = [0.0] * (opt.num_train_iter + 1)
    aug_lagrangians_val = []
    grad_norms = []
    grad_norm_ma = [0.0] * (opt.num_train_iter + 1)
    if not opt.no_w_adjs_log:
        w_adjs = np.zeros((opt.num_train_iter, opt.num_vars, opt.num_vars), dtype=np.float32)

    constraint_violation_list = []
    not_nlls = []  # Augmented Lagrangrian minus (pseudo) NLL
    nlls = []  # NLL on train
    nlls_val = []  # NLL on validation
    delta_mu = np.inf
    w_adj_mode = "gumbel"

    # Augmented Lagrangian stuff
    mu = opt.mu_init
    gamma = opt.gamma_init
    mus = []
    gammas = []

    if opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    elif opt.optimizer == "rmsprop":
        # This allows the optimizer to return the learning rates for each parameters
        monkey_patch_RMSprop(torch.optim.RMSprop)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr)
    else:
        raise NotImplementedError("optimizer {} is not implemented".format(opt.optimizer))

    # compute constraint normalization
    with torch.no_grad():
        full_adjacency = torch.ones((model.num_vars, model.num_vars)) - torch.eye(model.num_vars)
        constraint_normalization = compute_dag_constraint(full_adjacency).item()

    # Learning loop:
    for iter in range(opt.num_train_iter):
        # compute loss
        model.train()
        x, mask, regime = train_data.sample(opt.train_batch_size)
        weights, biases, extra_params = model.get_parameters(mode="wbx")

        loss = compute_loss(x, mask, regime, model, weights, biases, extra_params,
                            opt.intervention, opt.intervention_type,
                            opt.intervention_knowledge)

        nlls.append(loss.item())
        model.eval()

        # constraint related
        w_adj = model.get_w_adj()
        h = compute_dag_constraint(w_adj) / constraint_normalization
        constraint_violation = h.item()

        # compute regularizer
        reg = opt.reg_coeff * compute_penalty([w_adj], p=1)
        reg /= w_adj.shape[0]**2

        if opt.coeff_interv_sparsity > 0 and opt.intervention_knowledge == "unknown" :
            interv_w = 1 - model.gumbel_interv_w.get_proba()
            group_norm = torch.norm(interv_w, p=1, dim=1, keepdim=True)
            reg_interv = opt.coeff_interv_sparsity * (group_norm).sum()
        else:
            reg_interv = torch.tensor(0)

        # compute augmented langrangian
        lagrangian = loss + reg + reg_interv + gamma * h
        augmentation = h ** 2

        aug_lagrangian = lagrangian + 0.5 * mu * augmentation

        # optimization step on augmented lagrangian
        optimizer.zero_grad()
        aug_lagrangian.backward()
        _, lr = optimizer.step() if opt.optimizer == "rmsprop" else optimizer.step(), opt.lr

        # logging
        if not opt.no_w_adjs_log:
            w_adjs[iter, :, :] = w_adj.detach().cpu().numpy().astype(np.float32)
        mus.append(mu)
        gammas.append(gamma)
        not_nlls.append(reg.item() + 0.5 * mu * h.item() ** 2 + gamma * h.item())

        # compute augmented lagrangian moving average
        aug_lagrangians.append(aug_lagrangian.item())
        aug_lagrangian_ma[iter + 1] = aug_lagrangian_ma[iter] + 0.01 * (aug_lagrangian.item() - aug_lagrangian_ma[iter])
        grad_norms.append(model.get_grad_norm("wbx").item())
        grad_norm_ma[iter + 1] = grad_norm_ma[iter] + 0.01 * (grad_norms[-1] - grad_norm_ma[iter])

        # compute loss on whole validation set
        if iter % opt.stop_crit_win == 0:
            with torch.no_grad():
                x, mask, regime = test_data.sample(test_data.num_samples)
                loss_val = compute_loss(x, mask, regime, model, weights, biases,
                                        extra_params, opt.intervention,
                                        opt.intervention_type,
                                        opt.intervention_knowledge).item()

                nlls_val.append(loss_val)
                aug_lagrangians_val.append([iter, loss_val + not_nlls[-1]])

        # compute delta for gamma
        if iter >= 2 * opt.stop_crit_win and iter % (2 * opt.stop_crit_win) == 0:
            t0, t_half, t1 = aug_lagrangians_val[-3][1], aug_lagrangians_val[-2][1], aug_lagrangians_val[-1][1]

            # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
            if not (min(t0, t1) < t_half < max(t0, t1)):
                delta_gamma = -np.inf
            else:
                delta_gamma = (t1 - t0) / opt.stop_crit_win
        else:
            delta_gamma = -np.inf  # do not update gamma nor mu

        # log metrics
        if iter % 5000 == 0:
            print("Iteration:", iter)
            # if opt.num_vars <= 5:
            #     print("    w_adj({}):\n".format(w_adj_mode), w_adj.detach().cpu().numpy())
            #     print("    current_adjacency:\n", model.adjacency.detach().cpu().numpy())

            with torch.no_grad():
                to_keep = (model.get_w_adj() > 0.5).type(torch.Tensor)
                current_adj = model.adjacency * to_keep
                current_adj = current_adj.cpu().numpy()
                acyclic = is_acyclic(current_adj)

            metrics_callback(stage="train", step=iter,
                             metrics={"aug-lagrangian": aug_lagrangian.item(),
                                      "aug-lagrangian-moving-avg": aug_lagrangian_ma[iter + 1],
                                      "aug-lagrangian-val": aug_lagrangians_val[-1][1],
                                      "nll": nlls[-1],
                                      "nll-val": nlls_val[-1],
                                      "nll-gap": nlls_val[-1] - nlls[-1],
                                      "grad-norm-moving-average": grad_norm_ma[iter + 1],
                                      "delta_gamma": delta_gamma,
                                      "omega_gamma": opt.omega_gamma,
                                      "delta_mu": delta_mu,
                                      "omega_mu": opt.omega_mu,
                                      "constraint_violation": constraint_violation,
                                      "acyclicity_violation": h.item(),
                                      "mu": mu,
                                      "gamma": gamma,
                                      "initial_lr": opt.lr,
                                      "current_lr": opt.lr,
                                      "is_acyclic": int(acyclic),
                                      })

            if iter % 10000 == 0:
                print("    current_adjacency:\n", model.adjacency.detach().cpu().numpy())

        # plot
        # if iter % opt.plot_freq == 0:
        #     if not opt.no_w_adjs_log:
        #         plot_weighted_adjacency(w_adjs[:iter + 1], gt_adjacency, opt.exp_path,
        #                                 name="w_adj_{}".format(w_adj_mode),
        #                                 mus=mus, gammas=gammas,
        #                                 first_stop=first_stop,
        #                                 second_stop=second_stop, plotting_callback=plotting_callback)
        #     plot_adjacency(model.adjacency.detach().cpu().numpy(), gt_adjacency, opt.exp_path)
        #     if opt.intervention_knowledge == "unknown":
        #         gumbel_interv = model.gumbel_interv_w.get_proba().detach().cpu().numpy()
        #         np.save(os.path.join(save_path, "gumbel_interv"), gumbel_interv)
        #         np.save(os.path.join(save_path, "gt_interv"), gt_interv)
        #         plot_interv_w(gumbel_interv, gt_interv, opt.exp_path)
        #         plot_interv_w(model.gumbel_interv_w.get_proba().detach().cpu().numpy(), gt_interv, opt.exp_path)
        #
        #     plot_learning_curves(not_nlls, aug_lagrangians, aug_lagrangian_ma[:iter], aug_lagrangians_val, nlls,
        #                          nlls_val, opt.exp_path, first_stop=first_stop,
        #                          second_stop=second_stop)
        #
        #     if opt.plot_density:
        #         # data: use only the observational data in the plot and use the train + test
        #         x, mask, regime = train_data.sample(3000)
        #         plot_learned_density(model, opt, gt_adjacency, x, mask,
        #                              opt.exp_path, step=iter, resolution=256,
        #                              show_data=True, log_probas=True,
        #                              cmap=None, scatter_color="white",
        #                              intervention=opt.intervention,
        #                              intervention_type=opt.intervention_type,
        #                              intervention_knowledge=opt.intervention_knowledge)


        # Does the augmented lagrangian converged?
        if constraint_violation <= opt.h_threshold and acyclic:
            if first_stop == 0:
                print(f"First stop at {iter}")
                first_stop = iter

        if constraint_violation > opt.h_threshold or not acyclic:
            # if we have found a stationary point of the augmented loss
            if abs(delta_gamma) < opt.omega_gamma or delta_gamma > 0:
                gamma += mu * h.item()
                print("Updated gamma to {}".format(gamma))

                # Did the constraint improve sufficiently?
                constraint_violation_list.append(constraint_violation)
                if len(constraint_violation_list) >= 2:
                    if constraint_violation_list[-1] > constraint_violation_list[-2] * opt.omega_mu:
                        mu *= opt.mu_mult_factor
                        print("Updated mu to {}".format(mu))

                # little hack to make sure the moving average is going down.
                with torch.no_grad():
                    gap_in_not_nll = reg.item() + 0.5 * mu * h.item() ** 2 + gamma * h.item() - not_nlls[-1]
                    aug_lagrangian_ma[iter + 1] += gap_in_not_nll
                    aug_lagrangians_val[-1][1] += gap_in_not_nll

                if opt.optimizer == "rmsprop":
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr_reinit)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr_reinit)

        else:
            if patience > 0:
                if iter % 1000 == 0:
                    # compute loss on whole validation set
                    # and then aug lagrangian
                    with torch.no_grad():
                        x, mask, regime = test_data.sample(test_data.num_samples)
                        loss_val = compute_loss(x, mask, regime, model, weights, biases,
                                                extra_params, opt.intervention,
                                                opt.intervention_type,
                                                opt.intervention_knowledge).item()
                    aug_lagrangian_val = loss_val + not_nlls[-1]

                    if aug_lagrangian_val < best_lagrangian_val:
                        best_lagrangian_val = aug_lagrangian_val
                        patience = opt.train_patience
                        # best_model = copy.deepcopy(model)
                    else:
                        patience -= 1
                    print(f"aug_lagrangian_val: {aug_lagrangian_val}, \
                          best_lagrangian_val:{best_lagrangian_val}")

            elif not thresholded:
                # Final thresholding of all edges <= 0.5
                # and edges > 0.5 are set to 1
                with torch.no_grad():
                    higher = (w_adj > 0.5).type(torch.Tensor)
                    lower = (w_adj <= 0.5).type(torch.Tensor)
                    model.gumbel_adjacency.log_alpha.copy_(higher * 100 + lower * -100)
                    model.gumbel_adjacency.log_alpha.requires_grad = False
                    model.adjacency.copy_(higher)
                best_nll_val = np.inf
                thresholded = True
                if second_stop == 0:
                    print(f"Second stop at {iter}")
                    second_stop = iter

            elif patience_thresh > 0:
                if iter % 1000 == 0:
                    # compute loss on whole validation set
                    with torch.no_grad():
                        x, mask, regime = test_data.sample(test_data.num_samples)
                        loss_val = compute_loss(x, mask, regime, model, weights, biases,
                                                extra_params, opt.intervention,
                                                opt.intervention_type,
                                                opt.intervention_knowledge).item()

                    # nll_val the best?
                    if loss_val < best_nll_val:
                        best_nll_val = loss_val
                        patience_thresh = opt.train_patience_post
                    else:
                        patience_thresh -= 1
            else:
                # End of training
                # timing = time.time() - time0

                # if second_stop == 0:
                #     print(f"Second stop at {iter}")
                #     second_stop = iter

                # # compute nll on train and validation set
                # weights, biases, extra_params = model.get_parameters(mode="wbx")
                # x, mask, regime = train_data.sample(train_data.num_samples)
                # # Since we do not have a DAG yet, this is not really a negative log likelihood.
                # nll_train = compute_loss(x, mask, regime, model, weights, biases,
                #                          extra_params, opt.intervention,
                #                          opt.intervention_type,
                #                          opt.intervention_knowledge)
                #
                # x, mask, regime = test_data.sample(test_data.num_samples)
                # nll_val = compute_loss(x, mask, regime, model, weights, biases,
                #                        extra_params, opt.intervention,
                #                        opt.intervention_type,
                #                        opt.intervention_knowledge)
                #
                # if opt.intervention_knowledge == "unknown":
                #     with torch.no_grad():
                #         gumbel_interv = model.gumbel_interv_w.get_proba().detach().cpu().numpy()
                #         gb_binary = (gumbel_interv < 0.5) * 1.
                #         diff = np.sum(np.abs(gt_interv - gb_binary))
                #         tp = np.sum((gt_interv == gb_binary) & (gb_binary == 1))
                #         fp = np.sum((gt_interv != gb_binary) & (gb_binary == 1))
                #         dump(f"diff: {diff}, tp: {tp}, fp:{fp}", save_path, 'regime_learned', True)

                # # Save
                # if not opt.no_w_adjs_log:
                #     w_adjs = w_adjs[:iter]
                # dump(model, save_path, 'model')
                # dump(opt.__dict__, save_path, 'opt')
                # if opt.num_vars <= 50 and not opt.no_w_adjs_log:
                #     dump(w_adjs, save_path, 'w_adjs')
                # dump(nll_train, save_path, 'pseudo-nll-train')
                # dump(nll_val, save_path, 'pseudo-nll-val')
                # dump(nlls, save_path, 'nlls')
                # dump(nlls_val, save_path, 'nlls-val')
                # dump(not_nlls, save_path, 'not-nlls')
                # dump(aug_lagrangians, save_path, 'aug-lagrangians')
                # dump(aug_lagrangian_ma[:iter], save_path, 'aug-lagrangian-ma')
                # dump(aug_lagrangians_val, save_path, 'aug-lagrangians-val')
                # dump(grad_norms, save_path, 'grad-norms')
                # dump(grad_norm_ma[:iter], save_path, 'grad-norm-ma')
                # dump(timing, save_path, 'timing')
                # np.save(os.path.join(save_path, "DAG"), model.adjacency.detach().cpu().numpy())
                #
                # # plot
                # if not opt.no_w_adjs_log:
                #     plot_weighted_adjacency(w_adjs, gt_adjacency, save_path,
                #                             name="w_adj_{}".format(w_adj_mode),
                #                             mus=mus, gammas=gammas,
                #                             first_stop=first_stop)
                # plot_adjacency(model.adjacency.detach().cpu().numpy(), gt_adjacency, save_path)
                # if opt.intervention_knowledge == "unknown":
                #     gumbel_interv = model.gumbel_interv_w.get_proba().detach().cpu().numpy()
                #     np.save(os.path.join(save_path, "gumbel_interv"), gumbel_interv)
                #     np.save(os.path.join(save_path, "gt_interv"), gt_interv)
                #     plot_interv_w(gumbel_interv, gt_interv, opt.exp_path)
                #
                # plot_learning_curves(not_nlls, aug_lagrangians, aug_lagrangian_ma[:iter], aug_lagrangians_val, nlls,
                #                      nlls_val, save_path, first_stop=first_stop)
                #
                # if opt.plot_density:
                #     x, mask, regime = train_data.sample(3000)
                #     plot_data = x[mask.sum(dim=-1) == mask.shape[1]]
                #     plot_learned_density(model, opt, gt_adjacency, plot_data,
                #                          opt.exp_path, step=iter, resolution=256,
                #                          show_data=True, log_probas=True, cmap=None, scatter_color="white",
                #                          intervention=opt.intervention,
                #                          intervention_type=opt.intervention_type,
                #                          intervention_knowledge=opt.intervention_knowledge)
                #     del plot_data
                #

                # # save results
                # model.eval()
                # _, mask, regime = train_data.sample(train_data.num_samples)
                #
                # # evaluate on validation set
                # x, mask, regime = test_data.sample(test_data.num_samples)
                # weights, biases, extra_params = model.get_parameters(mode="wbx")
                # nll_val = compute_loss(x, mask, regime, model, weights, biases, extra_params,
                #                        opt.intervention, opt.intervention_type,
                #                        opt.intervention_knowledge).item()
                #
                # # Compute SHD and SID metrics
                # pred_adj_ = model.adjacency.detach().cpu().numpy()
                # train_adj_ = train_data.adjacency.detach().cpu().numpy()
                # sid = float(cdt.metrics.SID(target=train_adj_, pred=pred_adj_))
                # shd = float(shd_metric(pred_adj_, train_adj_))
                # shd_cpdag = float(cdt.metrics.SHD_CPDAG(target=train_adj_, pred=pred_adj_))
                # fn, fp, rev = edge_errors(pred_adj_, train_adj_)
                # del train_adj_, pred_adj_
                #
                # # Save results
                # results = f"shd: {shd},\nsid: {sid},\nfn: {fn},\nfp: {fp},\nrev: {rev},\nnll_val:{best_nll_val}"
                # dump(results, save_path, 'results', True)
                # metrics_callback(stage="final", step=iter,
                #                  metrics={"shd": shd,
                #                           "sid": sid,
                #                           "fn": fn,
                #                           "fp": fp,
                #                           "rev": rev,
                #                           "nll_val": best_nll_val
                #                           })

                return model

# def retrain(model, train_data, test_data, dag_folder, opt, metrics_callback, plotting_callback):
#     """
#     Retrain a model which is already DAG
#     """
#     # Prepare path for saving results
#     stage_name = "retrain_{}".format(dag_folder)
#     save_path = os.path.join(opt.exp_path, stage_name)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     # Check if already computed
#     if os.path.exists(os.path.join(save_path, "best-model.pkl")):
#         print(stage_name, "already computed. Loading result from disk.")
#         return load(save_path, "best-model.pkl")
#
#     time0 = time.time()
#
#     # initialize stuff for learning loop
#     nlls = []
#     nlls_val = []
#     losses = []
#     losses_val = []
#     grad_norms = []
#     grad_norm_ma = [0.0] * (opt.num_train_iter + 1)
#
#     # early stopping stuff
#     best_model = copy.deepcopy(model)
#     best_nll_val = np.inf
#     patience = opt.patience
#
#     if opt.optimizer == "sgd":
#         optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
#     elif opt.optimizer == "rmsprop":
#         # This allows the optimizer to return the learning rates for each parameters
#         monkey_patch_RMSprop(torch.optim.RMSprop)
#         optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr)
#     else:
#         raise NotImplementedError("optimizer {} is not implemented".format(opt.optimizer))
#
#     # Learning loop:
#     for iter in range(opt.num_train_iter):
#         # compute loss
#         model.train()
#         x, mask, regime = train_data.sample(opt.train_batch_size)
#         weights, biases, extra_params = model.get_parameters(mode="wbx")
#         nll = compute_loss(x, mask, regime, model, weights, biases, extra_params,
#                            opt.intervention, opt.intervention_type,
#                            opt.intervention_knowledge)
#
#         nlls.append(nll.item())
#         model.eval()
#
#         # compute regularizer
#         # w_adj = model.get_w_adj()
#         # reg = opt.reg_coeff * compute_penalty([w_adj], p=1)
#         # reg /= w_adj.shape[0]**2
#
#         # if opt.coeff_interv_sparsity > 0 and opt.intervention_knowledge == "unknown" :
#         #     interv_w = 1 - model.gumbel_interv_w.get_proba()
#         #     group_norm = torch.norm(interv_w, p=1, dim=1, keepdim=True)
#         #     reg_interv = opt.coeff_interv_sparsity * (group_norm).sum()
#         # else:
#         #     reg_interv = torch.tensor(0)
#
#         reg = torch.tensor(0)
#         reg_interv = torch.tensor(0)
#
#         # compute augmented langrangian
#         loss = nll
#
#         # optimization step on augmented lagrangian
#         optimizer.zero_grad()
#         loss.backward()
#         _, lr = optimizer.step() if opt.optimizer == "rmsprop" else optimizer.step(), opt.lr
#
#         # compute augmented lagrangian moving average
#         losses.append(loss.item())
#         grad_norms.append(model.get_grad_norm("wbx").item())
#         grad_norm_ma[iter + 1] = grad_norm_ma[iter] + 0.01 * (grad_norms[-1] - grad_norm_ma[iter])
#
#         # compute loss on whole validation set
#         if iter % 1000 == 0:
#             with torch.no_grad():
#                 x, mask, regime = test_data.sample(test_data.num_samples)
#                 nll_val = compute_loss(x, mask, regime, model, weights, biases,
#                                        extra_params, opt.intervention,
#                                        opt.intervention_type,
#                                        opt.intervention_knowledge)
#                 # nll_val = - torch.mean(model.compute_log_likelihood(x, weights, biases, extra_params)).item()
#                 nlls_val.append(nll_val)
#                 losses_val.append([iter, nll_val + reg.item()])
#
#                 # nll_val the best?
#                 if nll_val < best_nll_val:
#                     best_nll_val = nll_val
#                     patience = opt.patience
#                     best_model = copy.deepcopy(model)
#                 else:
#                     patience -= 1
#
#         # log metrics
#         if iter % 100 == 0:
#             print("Iteration:", iter)
#             metrics_callback(stage=stage_name, step=iter,
#                              metrics={"loss": loss.item(),
#                                       "loss-val": losses_val[-1][1],
#                                       "nll": nlls[-1],
#                                       "nll-val": nlls_val[-1],
#                                       "grad-norm-moving-average": grad_norm_ma[iter + 1],
#                                       "w_prop_0": sum([(w == 0).long().sum().item() for w in weights]) /
#                                                   model.numel_weights,
#                                       "patience": patience,
#                                       "best-nll-val": best_nll_val})
#
#         # plot
#         if iter % opt.plot_freq == 0:
#             plot_learning_curves_retrain(losses, losses_val, nlls, nlls_val, save_path)
#
#         # Have we converged?
#         if patience == 0:
#             timing = time.time() - time0
#
#             # save
#             dump(best_nll_val, save_path, 'best-nll-val', txt=True)
#             dump(opt.__dict__, save_path, 'opt')
#             dump(nlls, save_path, 'nlls-train')
#             dump(nlls_val, save_path, 'nlls-val')
#             dump(losses, save_path, 'losses')
#             dump(losses_val, save_path, 'losses-val')
#             dump(grad_norms, save_path, 'grad-norms')
#             dump(grad_norm_ma[:iter], save_path, 'grad-norm-ma')
#             dump(timing, save_path, 'timing')
#
#             # plot
#             plot_learning_curves_retrain(losses, losses_val, nlls, nlls_val, save_path)
#
#             return model
