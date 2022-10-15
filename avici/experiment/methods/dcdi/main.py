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
import math
import torch
import numpy as np
from types import SimpleNamespace
import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

try:
    from avici.experiment.methods.dcdi.learnables import LearnableModel_NonLinGaussANM
    from avici.experiment.methods.dcdi.flows import DeepSigmoidalFlowModel
    from .train import train, compute_loss
    from .data import CustomManagerFile
    from .utils.save import dump

except ImportError:
    from avici.experiment.methods.dcdi.learnables import LearnableModel_NonLinGaussANM
    from avici.experiment.methods.dcdi.flows import DeepSigmoidalFlowModel
    from train import train, compute_loss
    from data import CustomManagerFile
    from utils.save import dump


def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)

def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))

def main(opt, x, interv_mask, heldout_data=None, metrics_callback=_print_metrics, plotting_callback=None):
    """
    :param opt: a Bunch-like object containing hyperparameter values
    :param metrics_callback: a function of the form f(step, metrics_dict)
        used to log metric values during training
    """

    # Control as much randomness as possible
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)

    if opt.lr_reinit is not None:
        assert opt.lr_schedule is None, "--lr-reinit and --lr-schedule are mutually exclusive"

    # Dump hyperparameters to disk
    # dump(opt.__dict__, opt.exp_path, 'opt')

    # Initialize metric logger if needed
    if metrics_callback is None:
        metrics_callback = _print_metrics

    # adjust some default hparams
    if opt.lr_reinit is None: opt.lr_reinit = opt.lr

    # Use GPU
    if opt.gpu:
        if opt.float:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        if opt.float:
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')

    # # create experiment path
    # if not os.path.exists(opt.exp_path):
    #     os.makedirs(opt.exp_path)

    # raise error if not valid setting
    if not(not opt.intervention or \
    (opt.intervention and opt.intervention_type == "perfect" and opt.intervention_knowledge == "known") or \
    (opt.intervention and opt.intervention_type == "perfect" and opt.intervention_knowledge == "unknown") or \
    (opt.intervention and opt.intervention_type == "imperfect" and opt.intervention_knowledge == "known")):
        raise ValueError("Not implemented")

    # if observational, force interv_type to perfect/known
    if not opt.intervention:
        print("No intervention")
        opt.intervention_type = "perfect"
        opt.intervention_knowledge = "known"

    # create DataManager for training
    train_data = CustomManagerFile(x, interv_mask, opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data,
                                 random_seed=opt.random_seed,
                                 intervention=opt.intervention,
                                 intervention_knowledge=opt.intervention_knowledge,
                                 dcd=opt.dcd,
                                 regimes_to_ignore=opt.regimes_to_ignore)
    test_data = CustomManagerFile(x, interv_mask, opt.train_samples, opt.test_samples, train=False,
                                normalize=opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                random_seed=opt.random_seed,
                                intervention=opt.intervention,
                                intervention_knowledge=opt.intervention_knowledge,
                                dcd=opt.dcd,
                                regimes_to_ignore=opt.regimes_to_ignore)

    if heldout_data is not None:
        val_x, val_interv_mask = heldout_data
        val_data = CustomManagerFile(val_x, val_interv_mask, 0, val_x.shape[0], train=False,
                                      normalize=opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                      random_seed=opt.random_seed,
                                      intervention=opt.intervention,
                                      intervention_knowledge=opt.intervention_knowledge,
                                      dcd=opt.dcd,
                                      regimes_to_ignore=opt.regimes_to_ignore)


    # create learning model and ground truth model
    if opt.model == "DCDI-G":
        model = LearnableModel_NonLinGaussANM(opt.num_vars,
                                              opt.num_layers,
                                              opt.hid_dim,
                                              nonlin=opt.nonlin,
                                              intervention=opt.intervention,
                                              intervention_type=opt.intervention_type,
                                              intervention_knowledge=opt.intervention_knowledge,
                                              num_regimes=train_data.num_regimes)
    elif opt.model == "DCDI-DSF":
        model = DeepSigmoidalFlowModel(num_vars=opt.num_vars,
                                       cond_n_layers=opt.num_layers,
                                       cond_hid_dim=opt.hid_dim,
                                       cond_nonlin=opt.nonlin,
                                       flow_n_layers=opt.flow_num_layers,
                                       flow_hid_dim=opt.flow_hid_dim,
                                       intervention=opt.intervention,
                                       intervention_type=opt.intervention_type,
                                       intervention_knowledge=opt.intervention_knowledge,
                                       num_regimes=train_data.num_regimes)
    else:
        raise ValueError("opt.model has to be in {DCDI-G, DCDI-DSF}")


    # save gt adjacency
    # dump(train_data.adjacency.detach().cpu().numpy(), opt.exp_path, 'gt-adjacency')

    # train until constraint is sufficiently close to being satisfied
    train(model, train_data.gt_interv, train_data, test_data, opt, metrics_callback, plotting_callback)

    # get predicted graph
    g_edges = model.adjacency.detach().cpu().numpy()
    g_edge_probs = (model.adjacency * model.get_w_adj()).detach().cpu().numpy()

    if np.allclose(g_edges, 1 - np.eye(g_edges.shape[-1])):
        warnings.warn("dcdi predicted all ones, probably not converged yet")

    pred = dict(g_edges=g_edges, g_edge_probs=g_edge_probs)

    # compute validation score if heldout data is given
    if heldout_data is not None:
        with torch.no_grad():
            x_val, masks_val, regimes_val = val_data.sample(val_data.num_samples)
            print(f"\nEvaluating trained model on heldout data of shape {x_val.shape}")

            weights, biases, extra_params = model.get_parameters(mode="wbx")
            neg_log_likelihood_val = compute_loss(x_val, masks_val, regimes_val, model, weights, biases, extra_params,
                                                  intervention=opt.intervention,
                                                  intervention_type=opt.intervention_type,
                                                  intervention_knowledge=opt.intervention_knowledge)

            pred["heldout_score"] = - neg_log_likelihood_val.cpu().numpy().item()

    return pred


def run_dcdi(seed, data, config, heldout_split=0.0):
    """
    The below code is minimally modified from https://github.com/slachapelle/dcdi
    to accept the hyperparameters without argparse and our data format
    """

    # concatenate all observations
    x_concat = np.concatenate([data["x_obs"], data["x_int"]], axis=-3)
    x_full, interv_mask_full = x_concat[..., 0], x_concat[..., 1]

    # to calibrate hyperparameters, split the data before running the algorithm
    if heldout_split > 0.0:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(x_full.shape[0])
        cutoff = math.floor(x_full.shape[0] * heldout_split)
        train_idx = perm[cutoff:]
        val_idx = perm[:cutoff]

        x = x_full[train_idx]
        interv_mask = interv_mask_full[train_idx]
        heldout_data = x_full[val_idx], interv_mask_full[val_idx]

        print(f"Splitting data into train size {train_idx.shape} and val size {val_idx.shape}")
    else:
        x = x_full
        interv_mask = interv_mask_full
        heldout_data = None

    # copy of the argparse arguments in https://github.com/slachapelle/dcdi/blob/master/main.py
    args = SimpleNamespace()
    args.random_seed = seed
    args.gpu = config.get("use_gpu", False)
    args.float = False

    args.num_vars = x.shape[1]
    args.train_samples = 0.8
    args.test_samples = None
    args.train_batch_size = min(config["train_batch_size"], math.floor(x.shape[0] * args.train_samples))
    args.num_train_iter = config["num_train_iter"]
    args.normalize_data = False
    args.regimes_to_ignore = None

    args.model = config["model"]
    args.num_layers = config["num_layers"]
    args.hid_dim = config["hid_dim"]
    args.nonlin = "leaky-relu"
    args.flow_num_layers = config["num_layers"]
    args.flow_hid_dim = config["hid_dim"]

    args.intervention = True
    args.dcd = False # Use DCD (DCDI with a loss not taking into account the intervention)
    args.intervention_type = "perfect"
    args.intervention_knowledge = "known"
    args.coeff_interv_sparsity = 1e-8

    args.optimizer = "rmsprop"
    args.lr = 1e-3
    args.lr_reinit = None
    args.lr_schedule = None
    args.stop_crit_win = 100
    args.reg_coeff = config["reg_coeff"]

    args.omega_gamma = 1e-4
    args.omega_mu = 0.9
    args.mu_init = 1e-8
    args.mu_mult_factor = 2
    args.gamma_init = 0.0
    args.h_threshold = 1e-8

    args.patience = 10
    args.train_patience = 5
    args.train_patience_post = 5
    args.lr_schedule = None
    args.lr_schedule = None

    args.no_w_adjs_log = True

    # run dcdi
    return main(args, x, interv_mask, heldout_data=heldout_data)


if __name__ == "__main__":

    from avici.utils.parse import load_data_config

    test_spec = load_data_config("config/linear_additive-0.yaml")["data"]["train"][0]
    testnvars = 10

    testrng = np.random.default_rng(0)
    testg, testeffect_sgn, testtoporder = test_spec.g(testrng, testnvars)
    testdata = test_spec.mechanism(spec=test_spec, rng=testrng, g=testg, effect_sgn=testeffect_sgn,
                                   toporder=testtoporder, n_vars=testnvars)

    print("true graph")
    print(testg)

    testpred = run_dcdi(
        42,
        testdata,
        dict(
            # model="DCDI-DSF",
            model="DCDI-G",
            num_layers=2,
            hid_dim=16,
            train_batch_size=64,
            num_train_iter=1000,
            reg_coeff=0.5,
        ),
        heldout_split=0.2,
    )
