import math
import numpy as onp
import copy
from collections import defaultdict

from avici.synthetic.sergio.sim import Sergio as SergioSimulator
from avici.utils.load import load_yaml
from avici.definitions import SERGIO_NOISE_CONFIG
from avici.synthetic import MechanismModel, Data


class GRNSergio(MechanismModel):
    """
    SERGIO simulator for GRNs

    Args:
        b (Distribution): distribution for sampling basic reproduction rates. Example: `avici.synthetic.Uniform`
        k_param (Distribution): distribution for sampling (non-negative) interaction strenghts.
            Example: `avici.synthetic.Uniform`
        k_sign_p (Distribution): distribution of sampling probability for positive (vs. negative)
            interaction sign signs. Example: `avici.synthetic.Beta`
        hill (float): Hill function coefficient
        decays (float): decay rate
        noise_params (float): noise scale parameter
        cell_types (Distribution): distribution for sampling integer number of cell types.
            Example: `avici.synthetic.RandInt`
        noise_type (str): noise type in SERGIO simulator. Default: `dpd`
        sampling_state (int): configuration of SERGIO sampler. Default: 15
        dt (float): dt increment in stochastic process. Default: 0.01

        * Technical noise*

        tech_noise_config (str): specification of noise elvels.
            Select one of the keys in `avici/synthetic/sergio/noise_config.yaml`
        add_outlier_effect (bool): whether to simulate outlier effects based on `tech_noise_config`
        add_lib_size_effect (bool): whether to simulate library size effects based on `tech_noise_config`
        add_dropout_effect (bool): whether to simulate dropout effects based on `tech_noise_config`
        return_count_data (bool): whether to return Poisson count data of the float mean expression levels

        * Interventions *

        n_ko_genes (int): no. unique genes knocked out in all of data collected; -1 indicates all genes
    """
    def __init__(self,
                 b,
                 k_param,
                 k_sign_p,
                 hill,
                 decays,
                 noise_params,
                 cell_types,
                 noise_type='dpd',
                 sampling_state=15,
                 dt=0.01,
                 tech_noise_config=None,
                 add_outlier_effect=False,
                 add_lib_size_effect=False,
                 add_dropout_effect=False,
                 return_count_data=False,
                 # interventions
                 n_ko_genes=0):

        self.b = b
        self.k_param = k_param
        self.k_sign_p = k_sign_p
        self.hill = hill
        self.decays = decays
        self.noise_params = noise_params
        self.cell_types = cell_types
        self.noise_type = noise_type
        self.sampling_state = sampling_state
        self.dt = dt
        self.tech_noise_config = tech_noise_config
        self.add_outlier_effect = add_outlier_effect
        self.add_lib_size_effect = add_lib_size_effect
        self.add_dropout_effect = add_dropout_effect
        self.return_count_data = return_count_data
        self.n_ko_genes = n_ko_genes

    def __call__(self, rng, g, n_observations_obs, n_observations_int):

        # sample interaction terms K
        n_vars = g.shape[-1]
        k = onp.abs(self.k_param(rng, shape=(n_vars, n_vars)))
        effect_sgn = rng.binomial(1, self.k_sign_p(rng, shape=(n_vars, 1)), size=g.shape) * 2.0 - 1.0

        k = k * effect_sgn.astype(onp.float32)
        assert onp.array_equal(k != 0, effect_sgn != 0)
        assert set(onp.unique(effect_sgn)).issubset({-1.0, 0.0, 1.0})

        # sample number of cell types to be sampled (i.e. number of unique master regulator reproduction rates)
        n_cell_types = int(self.cell_types(rng, shape=1))
        assert n_cell_types > 0, "Need at least 1 cell type to be simulated"

        # master regulator basal reproduction rate
        basal_rates = self.b(rng, shape=(n_vars, n_cell_types)) # assuming 1 cell type is simulated

        # hill coeff
        hills = self.hill * onp.ones((n_vars, n_vars))

        # sample technical noise configuration
        if self.add_outlier_effect or self.add_lib_size_effect or self.add_dropout_effect:
            config = load_yaml(path=SERGIO_NOISE_CONFIG)
            assert self.tech_noise_config in config.keys(), f"tech_noise_config `{self.tech_noise_config}` " \
                                                            f"not in config keys: `{list(config.keys())}`"
            outlier_prob = rng.choice(config[self.tech_noise_config]["outlier_prob"])
            outlier_mean = rng.choice(config[self.tech_noise_config]["outlier_mean"])
            outlier_scale = rng.choice(config[self.tech_noise_config]["outlier_scale"])
            lib_size_mean = rng.choice(config[self.tech_noise_config]["lib_size_mean"])
            lib_size_scale = rng.choice(config[self.tech_noise_config]["lib_size_scale"])
            dropout_shape = rng.choice(config[self.tech_noise_config]["dropout_shape"])
            dropout_percentile = rng.choice(config[self.tech_noise_config]["dropout_percentile"]).astype(int)

        # sample targets for experiments performed (wild-type, knockout)
        ko_targets = []

        simulate_observ_data = n_observations_obs > 0
        if simulate_observ_data:
            ko_targets += [None]

        simulate_interv_data = n_observations_int > 0
        if simulate_interv_data:
            assert self.n_ko_genes != 0, f"Need n_ko_genes != 0 to have interventional data for SERGIO"
            if self.n_ko_genes == -1:
                n_ko_genes = n_vars
            ko_targets += sorted(rng.choice(n_vars, size=min(n_vars, n_ko_genes), replace=False).tolist())


        # simulate for wild type and each ko target
        data = defaultdict(lambda: defaultdict(list))
        for ko_target in ko_targets:

            if ko_target is None:
                # observational/wild type
                data_type = "obs"
                kout = onp.zeros(n_vars).astype(bool)
                number_sc = math.ceil(n_observations_obs / n_cell_types)

            else:
                # interventional/knockout
                data_type = "int"
                kout = onp.eye(n_vars)[ko_target].astype(bool)
                number_sc = math.ceil(n_observations_int / (n_cell_types * n_ko_genes))

            # setup simulator
            sim = SergioSimulator(
                rng=rng,
                number_genes=n_vars,
                number_bins=n_cell_types,
                number_sc=number_sc,
                noise_params=self.noise_params,
                noise_type=self.noise_type,
                decays=self.decays,
                sampling_state=self.sampling_state,
                kout=kout,
                dt=self.dt,
                safety_steps=10,
            )

            sim.custom_graph(
                g=g,
                k=k,
                b=basal_rates,
                hill=hills,
            )

            # run steady-state simulations
            assert number_sc >= 1, f"Need to have number_sc >= 1: number_sc {number_sc} data_type {data_type}"
            sim.simulate()

            # Get the clean simulated expression matrix after steady_state simulations
            # shape: [number_bins (#cell types), number_genes, number_sc (#cells per type)]
            expr = sim.getExpressions()

            # add technical noise (assume 1 experimental setting for wild type and KO experiments)
            # 1) outlier genes
            if self.add_outlier_effect:
                expr = sim.outlier_effect(expr, outlier_prob, outlier_mean, outlier_scale)

            # 2) library size
            if self.add_lib_size_effect:
                _, expr = sim.lib_size_effect(expr, lib_size_mean, lib_size_scale)

            # 3) dropout
            if self.add_dropout_effect:
                binary_ind = sim.dropout_indicator(expr, dropout_shape, dropout_percentile)
                expr = onp.multiply(binary_ind, expr)

            # 4) mRNA count data
            if self.return_count_data:
                expr = sim.convert_to_UMIcounts(expr)

            # Aggregate by concatenating gene expressions of all cell types into 2d array
            # [number_genes (#genes), number_bins * number_sc (#cells  = #cell_types * #cells_per_type)]
            expr_agg = onp.concatenate(expr, axis=1)

            # Now each row represents a gene and each column represents a simulated single-cell
            # Gene IDs match their row in this expression matrix
            # [number_bins * number_sc, number_genes]
            x = expr_agg.T
            x = rng.permutation(x, axis=0)

            # generate intervention mask
            # [number_bins * number_sc, number_genes] with True/False depending on whether gene was knocked out
            ko_mask = onp.tile(kout, (x.shape[0], 1)).astype(onp.float32)

            # advance rng outside for faithfullness/freshness of data in for loop
            rng = copy.deepcopy(sim.rng)

            data[data_type]["x"].append(x)
            data[data_type]["ko_mask"].append(ko_mask)


        # concatenate interventional data by interweaving rows to have balanced intervention target counts
        if simulate_observ_data:
            x_obs = onp.stack(data["obs"]["x"]).reshape(-1, n_vars, order="F")
            x_obs_msk = onp.stack(data["obs"]["ko_mask"]).reshape(-1, n_vars, order="F")
        else:
            x_obs = onp.zeros((0, n_vars))  # dummy
            x_obs_msk = onp.zeros((0, n_vars))  # dummy

        if simulate_interv_data:
            x_int = onp.stack(data["int"]["x"]).reshape(-1, n_vars, order="F")
            x_int_msk = onp.stack(data["int"]["ko_mask"]).reshape(-1, n_vars, order="F")
        else:
            x_int = onp.zeros((0, n_vars)) # dummy
            x_int_msk = onp.zeros((0, n_vars)) # dummy

        # clip number of observations to be invariant to n_cell_types due to rounding
        # [n_observations, n_vars, 2]
        x_obs = onp.stack([x_obs, x_obs_msk], axis=-1)[:n_observations_obs, :, :]
        x_int = onp.stack([x_int, x_int_msk], axis=-1)[:n_observations_int, :, :]

        assert x_obs.size != 0 or x_int.size != 0, f"Need to sample at least some observations; " \
                                                   f"got shapes x_obs {x_obs.shape} x_int {x_int.shape}"

        # collect data
        return Data(
            x_obs=x_obs,
            x_int=x_int,
            is_count_data=True,
        )