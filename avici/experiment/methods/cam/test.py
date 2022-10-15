import os
import time
import cdt
import argparse
import pandas as pd
import numpy as np
import networkx as nx

import sys
sys.path.append("..")
from dcdi.plot import plot_adjacency
from dcdi.utils.save import dump
from dcdi.utils.metrics import edge_errors
from dcdi.data import DataManagerFile
from cam import CAM_with_score


def main(opt, metrics_callback, plotting_callback=None):
    time0 = time.time()
    opt.model = "cam"

    # load data
    train_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data, random_seed=opt.random_seed,
                                 intervention=opt.intervention, regimes_to_ignore=opt.regimes_to_ignore)
    test_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=False,
                                normalize=opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                random_seed=opt.random_seed, intervention=opt.intervention,
                                regimes_to_ignore=opt.regimes_to_ignore)

    gt_dag = train_data.adjacency.detach().cpu().numpy()

    train, mask_train, regime_train = train_data.sample(train_data.num_samples)
    test, mask_test, regime_test = test_data.sample(test_data.num_samples)
    train_data_pd = pd.DataFrame(train.detach().cpu().numpy())
    test_data_pd = pd.DataFrame(test.detach().cpu().numpy())

    # apply CAM
    obj = CAM_with_score(opt.score, opt.cutoff, opt.variable_sel, opt.sel_method,
                         opt.pruning, opt.prune_method)
    if opt.intervention:
        mask_train_pd = pd.DataFrame(mask_train.detach().cpu().numpy())
        mask_test_pd = pd.DataFrame(mask_test.detach().cpu().numpy())
        dag, train_score, val_score = obj.get_score(train_data_pd, test_data_pd,
                                                    mask_train_pd, mask_test_pd)
    else:
        dag, train_score, val_score = obj.get_score(train_data_pd, test_data_pd)

    dag = nx.to_numpy_matrix(dag)

    # Compute SHD and SID metrics
    sid = float(cdt.metrics.SID(target=gt_dag, pred=dag))
    shd = float(cdt.metrics.SHD(target=gt_dag, pred=dag, double_for_anticausal=False))
    shd_cpdag = float(cdt.metrics.SHD_CPDAG(target=gt_dag, pred=dag))
    fn, fp, rev = edge_errors(dag, gt_dag)
    timing = time.time() - time0

    #save
    exp_path = os.path.join(opt.exp_path, f"exp{opt.i_dataset}")
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    metrics_callback(stage="cam", step=0,
                     metrics={"train_score": train_score, "val_score": val_score, "sid": sid, "shd": shd,
                              "shd_cpdag": shd_cpdag, "fn": fn, "fp": fp, "rev": rev}, throttle=False)

    dump(opt, exp_path, 'opt')
    dump(timing, exp_path, 'timing', True)
    dump(train_score, exp_path, 'train_score', True)
    dump(val_score, exp_path, 'test_score', True)
    dump(sid, exp_path, 'sid', True)
    dump(shd, exp_path, 'shd', True)
    results = f"shd: {shd},\nsid: {sid},\nfn: {fn},\nfp: {fp},\nrev: {rev}"

    dump(results, exp_path, 'results', True)
    np.save(os.path.join(exp_path, "DAG"), dag)

    plot_adjacency(gt_dag, dag, exp_path)

def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data files')
    parser.add_argument('--i-dataset', type=str, default=None,
                        help='dataset index')
    parser.add_argument('--exp-path', type=str, default='exp',
                        help='Path to experiments')
    parser.add_argument('--intervention', action="store_true",
                        help='Use interventional data')
    parser.add_argument('--regimes-to-ignore', nargs="+", type=int,
                        help='When loading data, will remove some regimes from data set')


    # Variable selection (PNS)
    parser.add_argument('--variable-sel', action="store_true",
                        help='Perform a variable selection step')
    parser.add_argument('--cutoff', type=float, default=0.001,
                        help='Threshold value for vaiable selection')

    # Pruning
    parser.add_argument('--pruning', action="store_true",
                        help='Perform an initial pruning step')
    opt = parser.parse_args()
    opt.score = 'nonlinear'
    opt.sel_method = 'gamboost'
    opt.prune_method = 'gam'

    opt.train_samples = 0.8
    opt.test_samples = None
    opt.random_seed = 42
    opt.normalize_data = False

    main(opt, metrics_callback=_print_metrics)
