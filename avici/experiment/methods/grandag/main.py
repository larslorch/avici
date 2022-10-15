import os
import argparse
import uuid
from types import SimpleNamespace
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir

import math
import numpy as np
import torch

from avici.experiment.methods.grandag.models.learnables import LearnableModel_NonLinGauss, LearnableModel_NonLinGaussANM
from avici.experiment.methods.grandag.train import pns, train, to_dag, cam_pruning, retrain
from avici.experiment.methods.grandag.data import DataManagerFile
from avici.experiment.methods.grandag.utils.save import load, dump

def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)

def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))


def main(opt, data, metrics_callback=None, plotting_callback=None):
    """
    :param opt: a Bunch-like object containing hyperparameter values
    :param data: data of shape [N, d]
    :param metrics_callback: a function of the form f(step, metrics_dict) used to log metric values during training

    """
    # Control as much randomness as possible
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)

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

    # create experiment path
    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)

    # create DataManager for training
    opt.train_batch_size = min(opt.train_batch_size, math.floor(opt.train_samples * data.shape[0]))
    opt.num_vars = data.shape[-1]
    train_data = DataManagerFile(data, opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data, random_seed=opt.random_seed)
    test_data = DataManagerFile(data, opt.train_samples, opt.test_samples, train=False,
                                normalize=opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                random_seed=opt.random_seed)

    # create learning model and ground truth model
    if opt.model == "NonLinGauss":
        model = LearnableModel_NonLinGauss(opt.num_vars, opt.num_layers, opt.hid_dim, nonlin=opt.nonlin,
                                           norm_prod=opt.norm_prod, square_prod=opt.square_prod)
    elif opt.model == "NonLinGaussANM":
        model = LearnableModel_NonLinGaussANM(opt.num_vars, opt.num_layers, opt.hid_dim, nonlin=opt.nonlin,
                                              norm_prod=opt.norm_prod,
                                              square_prod=opt.square_prod)
    else:
        raise ValueError("opt.model has to be in {NonLinGauss, NonLinGaussANM}")

    ### apply preliminary neighborhood selection
    # if opt.pns:
    if opt.num_neighbors is None:
        num_neighbors = opt.num_vars
    else:
        num_neighbors = opt.num_neighbors
    print("\n\nRunning PNS")
    pns(model, train_data, test_data, num_neighbors, opt.pns_thresh, opt.exp_path, metrics_callback,
        plotting_callback)

    # train until constraint is sufficiently close to being satisfied
    # if opt.train:
    print("\n\nRunning training")
    if file_exists(opt.exp_path, "pns"):
        print("Training with pns folder")
        model = load(os.path.join(opt.exp_path, "pns"), "model.pkl")
    else:
        print("Training from scratch")
    train(model, train_data, test_data, opt, metrics_callback,
          plotting_callback)

    ### remove edges until we have a DAG
    # if opt.to_dag:
    # load model
    assert file_exists(opt.exp_path, "train"), \
        "The /train folder is required to run --to_dag. Add --train to the command line"
    model = load(os.path.join(opt.exp_path, "train"), "model.pkl")

    # run
    to_dag(model, train_data, test_data, opt, metrics_callback, plotting_callback)

    ### do further pruning of the DAG
    # if opt.cam_pruning:
    # load model
    assert file_exists(opt.exp_path, "to-dag"), \
        "The /to-dag folder is required to run --cam-pruning. Add --to-dag to the command line"
    model = load(os.path.join(opt.exp_path, "to-dag"), "model.pkl")

    # run
    print("\n\nRunning CAM pruning")
    try:
        model = cam_pruning(model, train_data, test_data, opt, cutoff=float(opt.cam_pruning_cutoff),
                            metrics_callback=metrics_callback, plotting_callback=plotting_callback,
                            verbose=True)

    except RuntimeError as e:
        if  "RProcessError" in repr(e):
            print("\n*****************\n")
            print("CAM pruning failed; skipping pruning step")
            print("\n*****************\n")
            pass
        else:
            raise e

    g_edges = model.adjacency.detach().cpu().numpy().astype(int)

    print("\n\nGraN-DAG pipeline finished successfully")
    pred = dict(g_edges=g_edges)
    return pred, model


def run_grandag(seed, data, config, heldout_split=0.0):
    """
    The below code is minimally modified from https://github.com/kurowasan/GraN-DAG
    to accept the hyperparameters without argparse and our data format
    """

    # concatenate all observations and discard target mask
    x_full = np.concatenate([data["x_obs"], data["x_int"]], axis=-3)[..., 0]

    # to calibrate hyperparameters, split the data before running the algorithm
    if heldout_split > 0.0:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(x_full.shape[0])
        cutoff = math.floor(x_full.shape[0] * heldout_split)
        train_idx = perm[cutoff:]
        val_idx = perm[:cutoff]

        x = x_full[train_idx]
        heldout_x = x_full[val_idx]

        print(f"Splitting data into train size {train_idx.shape} and val size {val_idx.shape}")
    else:
        x = x_full
        heldout_x = None


    # create temporary directory
    run_dir = Path('{0!s}/grandag_{1!s}/{2!s}/'.format(gettempdir(), uuid.uuid4(), seed))
    run_dir.mkdir(exist_ok=True, parents=True)

    # copy of the argparse arguments in https://github.com/kurowasan/GraN-DAG/blob/master/main.py
    args = SimpleNamespace()
    args.random_seed = seed
    args.exp_path = run_dir

    # data
    # args.add_argument('--train-samples', type=int, default=0.8,
    #                     help='Number of samples used for training (default is 80% of the total size)')
    # args.add_argument('--test-samples', type=int, default=None,
    #                     help='Number of samples used for testing (default is whatever is not used for training)')
    # args.add_argument('--train-batch-size', type=int, default=64,
    #                     help='number of samples in a minibatch')
    # args.add_argument('--num-train-iter', type=int, default=100000,
    #                     help='number of meta gradient steps')
    # args.add_argument('--normalize-data', action="store_true",
    #                     help='(x - mu) / std')

    args.train_samples = 0.8
    args.test_samples = None
    args.train_batch_size = 64
    args.num_train_iter = 100000
    args.normalize_data = False

    # model
    # args.add_argument('--model', type=str, default="NonLinGaussANM",
    #                     help='model class')  # NonLinGaussANM or NonLinGauss
    # args.add_argument('--num-layers', type=int, default=config["num_layers"],
    #                     help="number of hidden layers")
    # args.add_argument('--hid-dim', type=int, default=config["hid_dim"],
    #                     help="number of hidden units per layer")
    # args.add_argument('--nonlin', type=str, default='leaky-relu',
    #                     help="leaky-relu | sigmoid")

    args.model = "NonLinGaussANM"
    args.num_layers = config["num_layers"]
    args.hid_dim = config["hid_dim"]
    args.nonlin = "leaky-relu"

    # optimization
    # args.add_argument('--optimizer', type=str, default="rmsprop",
    #                     help='sgd|rmsprop')
    # args.add_argument('--lr', type=float, default=config["lr"],
    #                     help='learning rate for optim')
    # args.add_argument('--lr-reinit', type=float, default=None,
    #                     help='Learning rate for optim after first subproblem. Default mode reuses --lr.')
    # args.add_argument('--scale-lr-with-mu', action="store_true",
    #                     help='Scale the learning rate wrt mu in the augmented lagrangian.')
    # args.add_argument('--stop-crit-win', type=int, default=100,
    #                     help='window size to compute stopping criterion')

    args.optimizer = "rmsprop"
    args.lr = config["lr"]
    args.lr_reinit = None
    args.scale_lr_with_mu = False
    args.stop_crit_win = 100

    # pns, pruning and thresholding
    # args.add_argument('--pns-thresh', type=float, default=config["pns_thres"],
    #                     help='threshold in PNS')
    # args.add_argument('--num-neighbors', type=int, default=None,
    #                     help='number of neighbors to select in PNS')
    # args.add_argument('--edge-clamp-range', type=float, default=1e-4,
    #                     help='as we train, clamping the edges (i,j) to zero when prod_ij is that close to zero. '
    #                          '0 means no clamping. Uses masks on inputs. Once an edge is clamped, no way back.')
    # args.add_argument('--cam-pruning-cutoff', nargs='+',
    #                     default=config["prune"],  # default=np.logspace(-6, 0, 10),
    #                     help='list of cutoff values. Higher means more edges')

    args.pns_thresh = config["pns_thresh"]
    args.num_neighbors = None
    args.edge_clamp_range = 1e-4
    args.cam_pruning_cutoff = config["prune"]

    # Augmented Lagrangian options
    # args.add_argument('--omega-lambda', type=float, default=1e-4,
    #                     help='Precision to declare convergence of subproblems')
    # args.add_argument('--omega-mu', type=float, default=0.9,
    #                     help='After subproblem solved, h should have reduced by this ratio')
    # args.add_argument('--mu-init', type=float, default=1e-3,
    #                     help='initial value of mu')
    # args.add_argument('--lambda-init', type=float, default=0.,
    #                     help='initial value of lambda')
    # args.add_argument('--h-threshold', type=float, default=1e-8,
    #                     help='Stop when |h|<X. Zero means stop AL procedure only when h==0. Should use --to-dag even '
    #                          'with --h-threshold 0 since we might have h==0 while having cycles (due to numerical issues).')

    args.omega_lambda = 1e-4
    args.omega_mu = 0.9
    args.mu_init = 1e-3
    args.lambda_init = 0.0
    args.h_threshold = 1e-8

    # misc
    # args.add_argument('--norm-prod', type=str, default="paths",
    #                     help='how to normalize prod: paths|none')
    # args.add_argument('--square-prod', action="store_true",
    #                     help="square weights instead of absolute value in prod")
    # args.add_argument('--jac-thresh', action="store_true",
    #                     help='threshold using the Jacobian instead of prod')
    # args.add_argument('--patience', type=int, default=10,
    #                     help='Early stopping patience in --retrain.')

    args.norm_prod = "paths"
    args.square_prod = False
    args.jac_thresh = False
    args.patience = 10

    # logging
    # args.add_argument('--plot-freq', type=int, default=1000000000,
    #                     help='plotting frequency')
    # args.add_argument('--no-w-adjs-log', action="store_true",
    #                     help='do not log weighted adjacency (to save RAM). One plot will be missing (A_\phi plot)')

    args.plot_freq = 1000000000
    args.no_w_adjs_log = False

    # device and numerical precision
    # args.add_argument('--gpu', action="store_true",
    #                     help="Use GPU")
    # args.add_argument('--float', action="store_true",
    #                     help="Use Float precision")

    args.gpu = False
    args.float = False

    # run and cleanup temporary files after
    try:
        grandag_pred, model = main(args, x)

    except Exception as e:
        rmtree(run_dir)
        raise e
    except KeyboardInterrupt:
        rmtree(run_dir)
        raise KeyboardInterrupt

    rmtree(run_dir)
    print("\ngrandag temporary files cleaned up.")

    # compute validation score if heldout data is given
    if heldout_x is not None:
        with torch.no_grad():
            print(f"\nEvaluating trained model on heldout data of shape {heldout_x.shape}")
            model.eval()
            weights, biases, extra_params = model.get_parameters(mode="wbx")
            neg_log_liks = - model.compute_log_likelihood(torch.tensor(heldout_x), weights, biases, extra_params)
            grandag_pred["heldout_score"] = torch.mean(neg_log_liks).cpu().numpy().item()

    return grandag_pred


if __name__ == "__main__":

    from avici.utils.parse import load_data_config

    test_spec = load_data_config("config/debug-linear_additive-0.yaml")["data"]["train"][0]
    testnvars = 7

    testrng = np.random.default_rng(0)
    testg, testeffect_sgn, testtoporder = test_spec.g(testrng, testnvars)
    testdata = test_spec.mechanism(spec=test_spec, rng=testrng, g=testg, effect_sgn=testeffect_sgn,
                                   toporder=testtoporder, n_vars=testnvars)

    print("true graph")
    print(testg)

    testpred = run_grandag(
        42,
        testdata,
        dict(
            num_layers=1,
            hid_dim=8,
            lr=1e-3,
            prune=1e-3,
            pns_thresh=0.75,
        ),
        heldout_split=0.2,
    )
    print()
    print(testpred)
