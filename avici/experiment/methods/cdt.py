import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"


import numpy as onp
import pandas as pd

from cdt.causality.graph import GES as rGES
from cdt.causality.graph import GIES as rGIES
from cdt.causality.graph import PC as rPC
from cdt.causality.graph import LiNGAM as rLiNGAM

from avici.utils.graph import nx_adjacency, random_consistent_expansion, is_acyclic, InvalidCPDAGError
from avici.synthetic.graph import orient_pdag_randomly

def run_ges(seed, data, config):
    """
    Greedy equivalence search
    The output is a Partially Directed Acyclic Graph (PDAG)
    (A markov equivalence class). The available scores assume linearity of
    mechanisms and gaussianity of the data.

    https://rdrr.io/cran/pcalg/man/ges.html
    https://cran.r-project.org/web/packages/pcalg/vignettes/vignette2018.pdf

    Uses the score 'obs' : `GaussL0penObsScore`

       l0-penalized Gaussian MLE estimator. By default,
       score = log(L(D)) − k * log(n)/2
       corresponding exactly to BIC with Gaussian likelihood
       Specifically, assumes linear structural equation model with Gaussian noise

    The 'int' score is used by GIES

    """
    rng = onp.random.default_rng(seed)
    x = onp.concatenate([data["x_obs"], data["x_int"]], axis=-3)[..., 0]

    # since GES infers a CPDAG, first infer CPDAG and then return a random DAG in the MEC
    pred_cpdag = rGES(seed, score="obs", verbose=False).predict(pd.DataFrame(data=x))
    cpdag = onp.array(nx_adjacency(pred_cpdag), dtype=onp.int32)

    # random consistent extension (DAG in MEC)
    pred = random_consistent_expansion(rng=rng, cpdag=cpdag)
    assert is_acyclic(pred)
    return dict(g_edges=pred, cpdag_edges=cpdag)


def run_gies(seed, data, config):
    """
    Greedy interventional equivalence search
    The output is a Partially Directed Acyclic Graph (PDAG)
    (A markov equivalence class). The available scores assume linearity of
    mechanisms and gaussianity of the data.

    https://rdrr.io/cran/pcalg/man/gies.html
    https://cran.r-project.org/web/packages/pcalg/vignettes/vignette2018.pdf

    Uses the score 'int' : `GaussL0penIntScore`.
    This is analogous to 'obs' used by GES but adapted for a mix of observational and interventional data

    """

    rng = onp.random.default_rng(seed)
    all_x = onp.concatenate([data["x_obs"], data["x_int"]], axis=-3)
    x = all_x[..., 0]
    target_mask = all_x[..., 1].astype(onp.int32)

    # convert mask [N, d] notation to the two expected target and target_indices variables (1-indexed for R)
    # `targets`: list of lists, where each inner list represents one "environment" by its intervened upon variables
    # `target_indices`: [N,] list containing environment index for each data sample
    unique, indices = onp.unique(target_mask, axis=0, return_inverse=True)
    targets = [((onp.where(v)[0] + 1).tolist() if v.sum() else [-1]) for v in unique]
    target_indices = (indices + 1).tolist()

    # flatten targets (nested list) into list; remember length of inner lists
    assert len(set(tuple(v) for v in targets)) == len(targets), \
        f"Each element in `targets` needs to be unique. \nGot {targets}"
    assert min(target_indices) == 1, "R is 1-indexed"
    assert max(target_indices) == len(targets), \
        "Provided more environments in `targets` than used in `target_indices`"
    flat_targets = []
    flat_target_lengths = []
    for idx, v in enumerate(targets):
        assert type(v) in [list, tuple]
        assert len(v) > 0 and len(set(v)) == len(v), "element of `targets` contains duplicates or is empty"
        assert not (any([w is None for w in v]) and len(v) > 1), \
            f"`None`, i.e. observational data, can only be an env by itself.\nGot: {v} in targets {targets}"
        flat_target_lengths.append(len(v))
        flat_targets += v

    # infer cpdag using GIES
    pred_cpdag = rGIES(seed, score="int", verbose=False).predict(
        pd.DataFrame(data=x),
        flat_targets=pd.DataFrame(data=flat_targets),
        flat_target_lengths=pd.DataFrame(data=flat_target_lengths),
        target_indices=pd.DataFrame(data=target_indices))

    cpdag = onp.array(nx_adjacency(pred_cpdag), dtype=onp.int32)

    # random consistent extension (DAG in MEC)
    pred = random_consistent_expansion(rng=rng, cpdag=cpdag)
    assert is_acyclic(pred)
    return dict(g_edges=pred, cpdag_edges=cpdag)


def run_pc(seed, data, config, max_tries=0):
    """
    Peter - Clark (PC) algorithm

    https://rdrr.io/cran/pcalg/man/pc.html
    https://cran.r-project.org/web/packages/pcalg/vignettes/vignette2018.pdf
    https://cran.r-project.org/web/packages/kpcalg/vignettes/kpcalg_tutorial.pdf

    ci_test:
        'binary',       # 0 "pcalg::binCItest",
        'discrete',     # 1 "pcalg::disCItest",
        'hsic_gamma',   # 2 "kpcalg::kernelCItest",
        'hsic_perm',    # 3 "kpcalg::kernelCItest",
        'hsic_clust',   # 4 "kpcalg::kernelCItest",
        'gaussian',     # 5 "pcalg::gaussCItest",
        'rcit',         # 6 "RCIT:::CItest",
        'rcot',         # 7 "RCIT:::CItest"}

    ci_alpha:   significance level for the individual CI tests

    """
    tries = 0
    finished_successfully = False
    while not finished_successfully:
        # run PC algorithm; if it produces an undirected cycle, restart with lower hypothesis testing alpha
        tries += 1
        try:
            rng = onp.random.default_rng(seed)
            x = onp.concatenate([data["x_obs"], data["x_int"]], axis=-3)[..., 0]

            # since PC infers a CPDAG, first infer CPDAG and then return a random DAG in the MEC
            # infer cpdag
            pred_cpdag = rPC(seed, CItest=config["ci_test"], alpha=config["ci_alpha"], njobs=None, verbose=False).predict(pd.DataFrame(data=x))
            cpdag = onp.array(nx_adjacency(pred_cpdag), dtype=onp.int32)

            # random consistent extension (DAG in MEC)
            pred = random_consistent_expansion(rng=rng, cpdag=cpdag)
            finished_successfully = True

        except InvalidCPDAGError:
            if tries >= max_tries:
                warnings.warn(f"PC returned cycles for {tries} try(s) and "
                              f"final alpha={config['ci_alpha']} for ci_test {config['ci_test']}."
                              f"Will randomly extend PDAG into a DAG.")
                pred = orient_pdag_randomly(rng, cpdag)
                finished_successfully = True
            else:
                config["ci_alpha"] *= 0.1

    return dict(g_edges=pred, cpdag_edges=cpdag)


def run_lingam(seed, data, config, rank_fail_add=1e-6, rank_check_tol=1e-6):
    """
    LiNGAM
    Assumes linear additive noise model, where the noise in non-Gaussian
    Based on Independent Component Analysis

    https://rdrr.io/cran/pcalg/man/LINGAM.html
    https://cran.r-project.org/web/packages/pcalg/vignettes/vignette2018.pdf
    """

    rng = onp.random.default_rng(seed)
    x = onp.concatenate([data["x_obs"], data["x_int"]], axis=-3)[..., 0]

    # check full rank of covariance matrix
    full_rank = onp.linalg.matrix_rank((x - x.mean(-2, keepdims=True)).T @
                                       (x - x.mean(-2, keepdims=True)), tol=rank_check_tol) == x.shape[-1]


    if not full_rank:
        zero_cols = onp.where((x == 0.0).all(-2))[0]
        zero_rows = onp.where((x == 0.0).all(-1))[0]
        warnings.warn(f"covariance matrix not full rank; "
                      f"we have {len(zero_rows)} zero rows and {len(zero_cols)} zero cols "
                      f"(can occur in gene expression data). "
                      f"Adding infinitesimal noise to observations.")

        x += rng.normal(loc=0, scale=rank_fail_add, size=x.shape)

    # lingam
    pred_dag = rLiNGAM(seed, verbose=False).predict(pd.DataFrame(data=x))
    pred = nx_adjacency(pred_dag)

    # matrix entries are coefficients corresponding to transpose of adjacency matrix
    pred = 1 - onp.isclose(pred.T, 0).astype(onp.int32)
    assert is_acyclic(pred)

    return dict(g_edges=pred)

