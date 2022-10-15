import os
import time
import cdt
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import random

from igsp import run_igsp, run_ut_igsp
import sys
sys.path.append("..")
from dcdi.plot import plot_adjacency
from dcdi.utils.save import dump
from dcdi.utils.metrics import edge_errors
from dcdi.data import DataManagerFile
import causaldag


def load_data(opt):
    train_data = DataManagerFile(opt.data_path, opt.i_dataset,
                                 opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data,
                                 random_seed=opt.random_seed, intervention=True,
                                 regimes_to_ignore=opt.regimes_to_ignore)

    gt_dag = train_data.adjacency.detach().cpu().numpy()
    train_data_pd = pd.DataFrame(train_data.dataset.detach().cpu().numpy())
    mask_pd = pd.DataFrame(train_data.masks)
    regimes = train_data.regimes

    return train_data_pd, mask_pd, regimes, gt_dag


def main(opt, metrics_callback=None, plotting_callback=None, verbose=False):
    time0 = time.time()
    np.random.seed(opt.random_seed)
    random.seed(opt.random_seed)

    # load data
    train_data_pd, mask_pd, regimes, gt_dag = load_data(opt)

    # run model
    if opt.model == "IGSP":
        dag, est_dag, targets_list = run_igsp(train_data_pd, targets=mask_pd,
                                              regimes=regimes, alpha=opt.alpha,
                       alpha_inv=opt.alpha_inv, ci_test=opt.ci_test)
    elif opt.model == "UTIGSP":
        dag, est_dag, targets_list, est_targets = run_ut_igsp(train_data_pd,
                                                              targets=mask_pd,
                                                              regimes=regimes,
                                   alpha=opt.alpha, alpha_inv=opt.alpha_inv,
                                   ci_test=opt.ci_test)
    else:
        raise ValueError("Method does not exist (should be IGSP or UTIGSP)")
    train_score, val_score = None, None


    # Compute SHD-CPDAG and SID metrics
    gt_dag_ = causaldag.classes.dag.DAG.from_amat(gt_dag)
    true_icpdag = gt_dag_.interventional_cpdag(targets_list, cpdag=gt_dag_.cpdag())
    est_icpdag = est_dag.interventional_cpdag(targets_list, cpdag=est_dag.cpdag())

    sid = float(cdt.metrics.SID(target=gt_dag, pred=dag))
    shd = float(cdt.metrics.SHD(target=gt_dag, pred=dag, double_for_anticausal=False))
    shd_cpdag = float(cdt.metrics.SHD_CPDAG(target=gt_dag, pred=dag))
    fn, fp, rev = edge_errors(dag, gt_dag)
    timing = time.time() - time0
    if verbose:
        print(f"SHD:{shd}, SID: {sid}, SHD_CPDAG:{shd_cpdag}")
        print(f"fn: {fn}, fp:{fp}, rev:{rev}")

    #save
    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)

    if metrics_callback is not None:
        metrics_callback(stage="igsp", step=0,
                         metrics={"train_score": train_score, "val_score": val_score, "sid": sid, "shd": shd,
                                  "shd_cpdag": shd_cpdag, "fn": fn, "fp": fp, "rev": rev}, throttle=False)

    dump(opt, opt.exp_path, 'opt')
    dump(timing, opt.exp_path, 'timing', True)
    dump(train_score, opt.exp_path, 'train_score', True)
    dump(val_score, opt.exp_path, 'test_score', True)

    results = f"shd: {shd},\nsid: {sid},\nfn: {fn},\nfp: {fp},\nrev: {rev}"
    dump(results, opt.exp_path, 'results', True)
    np.save(os.path.join(opt.exp_path, "DAG"), dag)

    plot_adjacency(gt_dag, dag, opt.exp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data files')
    parser.add_argument('--i-dataset', type=str, default=None,
                        help='dataset index')
    parser.add_argument('--exp-path', type=str, default='exp',
                        help='Path to experiments')
    parser.add_argument('--model', type=str, default="IGSP",
                        help='IGSP or UTIGSP')
    parser.add_argument('--alpha', type=float, default=1e-3,
                        help='Threshold for conditional indep tests')
    parser.add_argument('--alpha-inv', type=float, default=1e-3,
                        help='Threshold for invariance tests')
    parser.add_argument('--ci-test', type=str, default='gaussian',
                        help='Type of conditional independance test to use \
                        (gaussian, hsic, kci)')
    parser.add_argument('--regimes-to-ignore', nargs="+", type=int,
                        help='When loading data, will remove some regimes from data set')
    opt = parser.parse_args()

    opt.train_samples = 1.0
    opt.test_samples = None
    opt.random_seed = 43
    opt.normalize_data = False
    main(opt, verbose=True)
