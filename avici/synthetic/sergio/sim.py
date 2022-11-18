import sys
import numpy as np

class Gene(object):

    def __init__(self, geneID, geneType, binID = -1):

        """
        geneType: 'MR' master regulator or 'T' target
        bindID is optional
        """

        self.ID = geneID
        self.Type = geneType
        self.binID = binID
        self.Conc = []
        self.Conc_S = []
        self.dConc = []
        self.k = [] #For dynamics simulation it stores k1 to k4 for Rung-Kutta method, list of size 4 * num_c_to_evolve
        self.k_S = [] #For dynamics simulation it stores k1 to k4 for Rung-Kutta method, list of size 4 * num_c_to_evolve
        self.simulatedSteps_ = 0
        self.converged_ = False
        self.converged_S_ = False
        self.ss_U_ = 0.0 #This is the steady state concentration of Unspliced mRNA
        self.ss_S_ = 0.0 #This is the steady state concentration of Spliced mRNA

    def append_Conc (self, currConc):
        if isinstance(currConc, list):
            if currConc[0] < 0.0:
                self.Conc.append([0.0])
            else:
                self.Conc.append(currConc)
        else:
            if currConc < 0.0:
                self.Conc.append(0.0)
            else:
                self.Conc.append(currConc)


    def append_Conc_S (self, currConc):
        if isinstance(currConc, list):
            if currConc[0] < 0.0:
                self.Conc_S.append([0.0])
            else:
                self.Conc_S.append(currConc)
        else:
            if currConc < 0.0:
                self.Conc_S.append(0.0)
            else:
                self.Conc_S.append(currConc)

    def append_dConc (self, currdConc):
        self.dConc.append(currdConc)

    def append_k (self, list_currK):
        self.k.append(list_currK)

    def append_k_S (self, list_currK):
        self.k_S.append(list_currK)

    def del_lastK_Conc(self, K):
        for k in range(K):
            self.Conc.pop(-1)

    def del_lastK_Conc_S(self, K):
        for k in range(K):
            self.Conc_S.pop(-1)

    def clear_Conc (self):
        """
        This method clears all the concentrations except the last one that may
        serve as intial condition for rest of the simulations
        """
        self.Conc = self.Conc[-1:]

    def clear_dConc (self):
        self.dConc = []

    def incrementStep (self):
        self.simulatedSteps_ += 1

    def setConverged (self):
        self.converged_ = True

    def setConverged_S (self):
        self.converged_S_ = True

    def set_scExpression(self, list_indices):
        """
        selects input indices from self.Conc and form sc Expression
        """
        self.scExpression = np.array(self.Conc)[list_indices]

    def set_ss_conc_U(self, u_ss):
        if u_ss < 0.0:
            u_ss = 0.0

        self.ss_U_ = u_ss

    def set_ss_conc_S(self, s_ss):
        if s_ss < 0.0:
            s_ss = 0.0

        self.ss_S_ = s_ss

    def clear_k(self):
        self.k = []

    def clear_k_S(self):
        self.k_S = []


class Sergio:

    def __init__(self,
                 rng,
                 number_genes,
                 number_bins,
                 number_sc,
                 noise_params,
                 noise_type,
                 decays,
                 kout=None,
                 kdown=None,
                 dynamics = False,
                 sampling_state = 10,
                 tol = 1e-3,
                 window_length = 100,
                 dt = 0.01,
                 optimize_sampling = False,
                 safety_steps=0,
                 bifurcation_matrix = None,
                 noise_params_splice = None,
                 noise_type_splice = None,
                 splice_ratio = 4, dt_splice = 0.01,
                 migration_rate = None):
        """
        Noise is a gaussian white noise process with zero mean and finite variance.
        noise_params: The amplitude of noise in CLE. This can be a scalar to use
        for all genes or an array with the same size as number_genes.
        Tol: p-Value threshold above which convergence is reached
        window_length: length of non-overlapping window (# time-steps) that is used to realize convergence
        dt: time step used in  CLE
        noise_params and decays: Could be an array of length number_genes, or single value to use the same value for all genes
        number_sc: number of single cells for which expression is simulated
        sampling_state (>=1): single cells are sampled from sampling_state * number_sc steady-state steps
        optimize_sampling: useful for very large graphs. If set True, may help finding a more optimal sampling_state and so may ignore the input sampling_state
        noise_type: We consider three types of noise, 'sp': a single intrinsic noise is associated to production process, 'spd': a single intrinsic noise is associated to both
        production and decay processes, 'dpd': two independent intrinsic noises are associated to production and decay processes
        dynamics: whether simulate splicing or not
        bifurcation_matrix: is a numpy array (nBins_ * nBins) of <1 values; bifurcation_matrix[i,j] indicates whether cell type i differentiates to type j or not. Its value indicates the rate of transition. If dynamics == True, this matrix should be specified
        noise_params_splice: Same as "noise_params" but for splicing. if not specified, the same noise params as pre-mRNA is used
        noise_type_splice: Same as "noise_type" but for splicing. if not specified, the same noise type as pre-mRNA is used
        splice_ratio: it shows the relative amount of spliced mRNA to pre-mRNA (at steady-state) and therefore tunes the decay rate of spliced mRNA as a function of unspliced mRNA. Could be an array of length number_genes, or single value to use the same value for all genes
        dt_splice = time step for integrating splice SDE


        Note1: It's assumed that no two or more bins differentiate into the same new bin i.e. every bin has either 0 or 1 parent bin
        Note2: differentitation rates (e.g. type1 -> type2) specified in bifurcation_matrix specifies the percentage of cells of type2 that are at the vicinity of type1
        """

        self.rng = rng

        self.nGenes_ = number_genes
        self.nBins_ = number_bins
        self.nSC_ = number_sc
        self.sampling_state_ = sampling_state
        self.tol_ = tol
        self.winLen_ = window_length
        self.dt_ = dt
        self.optimize_sampling_ = optimize_sampling
        self.safety_steps = safety_steps
        self.level2verts_ = {}
        self.gID_to_level_and_idx = {} # This dictionary gives the level and idx in self.level2verts_ of a given gene ID
        self.binDict = {} # This maps bin ID to list of gene objects in that bin; only used for dynamics simulations
        self.maxLevels_ = 0
        self.init_concs_ = np.zeros((number_genes, number_bins))
        self.meanExpression = -1 * np.ones((number_genes, number_bins))
        self.noiseType_ = noise_type
        self.dyn_ = dynamics
        self.nConvSteps = np.zeros(number_bins) # This holds the number of simulated steps till convergence

        if kout is None:
            kout = np.zeros(number_genes).astype(bool)
        self.kout = kout

        if kdown is None:
            kdown = np.zeros(number_genes).astype(bool)
        self.kdown = kdown

        ############
        # This graph stores for each vertex: parameters(interaction
        # parameters for non-master regulators and production rates for master
        # regulators), tragets, regulators and level
        ############
        self.graph_ = {}

        if np.isscalar(noise_params):
            self.noiseParamsVector_ = np.repeat(noise_params, number_genes)
        elif np.shape(noise_params)[0] == number_genes:
            self.noiseParamsVector_ = noise_params
        else:
            print ("Error: expect one noise parameter per gene")


        if np.isscalar(decays) == 1:
            self.decayVector_ = np.repeat(decays, number_genes)
        elif np.shape(decays)[0] == number_genes:
            self.decayVector_ = decays
        else:
            print ("Error: expect one decay parameter per gene")
            sys.exit()


    def custom_graph(self, *, g, k, b, hill):
        """
        Prepare custom graph model and coefficients
        Args:
            g: [nGenes_, nGenes_] graph
            k: [nGenes_, nGenes_] interaction coeffs
            b: [nGenes_, nBins_] basal reproduction rate of source nodes (master regulators)
            hill: [nGenes_, nGenes_] hill coefficients of nonlinear interactions
        """

        # check inputs
        assert g.shape == k.shape
        assert g.shape == hill.shape
        assert g.shape[0] == self.nGenes_
        assert g.shape[1] == self.nGenes_
        assert b.shape[0] == self.nGenes_
        assert b.shape[1] == self.nBins_
        assert np.allclose(g[np.diag_indices(g.shape[0])], 0.0), f"No self loops allowed"

        # following steps of original function
        for i in range(self.nGenes_):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []


        self.master_regulators_idx_ = set()

        for j in range(self.nGenes_):

            is_parent = g[:, j]

            # master regulator (no parents)
            if is_parent.sum() == 0:

                self.master_regulators_idx_.add(j)
                self.graph_[j]['rates'] = b[j]
                self.graph_[j]['regs'] = []
                self.graph_[j]['level'] = -1

            # regular gene (target)
            else:

                currInteraction = []
                currParents = []
                for u in np.where(is_parent == 1)[0]:
                    currInteraction.append((u, k[u, j], hill[u, j], 0))  # last zero shows half-response, it is modified in another method
                    currParents.append(u)
                    self.graph_[u]['targets'].append(j)

                self.graph_[j]['params'] = currInteraction
                self.graph_[j]['regs'] = currParents
                self.graph_[j]['level'] = -1  # will be modified later


        self.find_levels_(self.graph_)

    def find_levels_ (self, graph):
        """
        # This is a helper function that takes a graph and assigns layer to all
        # verticies. It uses longest path layering algorithm from
        # Hierarchical Graph Drawing by Healy and Nikolovself. A bottom-up
        # approach is implemented to optimize simulator run-time. Layer zero is
        # the last layer for which expression are simulated
        # U: verticies with an assigned layer
        # Z: vertizies assigned to a layer below the current layer
        # V: set of all verticies (genes)

        This also sets a dictionary that maps a level to a matrix (in form of python list)
        of all genes in that level versus all bins

        Note to self:
        This is like DFS topsort, but compressing the length of levels as much as possible
        Essentially, root nodes have the highest level (to be simulated first) and sink nodes have level 0,
        and any node upstream of a node has higher level
        Sets:
            level2verts_
                {l: [M, bins] where M is number of genes in on level l}

            gID_to_level_and_idx
                {v: (level, j) where j in index of vertex v in `level2verts[level]` for vertex v in graph
        """

        U = set()
        Z = set()
        V = set(graph.keys())

        currLayer = 0
        self.level2verts_[currLayer] = []
        idx = 0

        while U != V:
            currVerts = set(filter(lambda v: set(graph[v]['targets']).issubset(Z), V-U))

            for v in currVerts:
                graph[v]['level'] = currLayer
                U.add(v)
                if {v}.issubset(self.master_regulators_idx_):
                    allBinList = [Gene(v,'MR', i) for i in range(self.nBins_)]
                    self.level2verts_[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1
                else:
                    allBinList = [Gene(v,'T', i) for i in range(self.nBins_)]
                    self.level2verts_[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1

            currLayer += 1
            Z = Z.union(U)
            self.level2verts_[currLayer] = []
            idx = 0

        self.level2verts_.pop(currLayer)
        self.maxLevels_ = currLayer - 1

        if not self.dyn_:
            self.set_scIndices_()

    def set_scIndices_ (self):
        """
        # First updates sampling_state_ if optimize_sampling_ is set True: to optimize run time,
        run for less than 30,000 steps in first level
        # Set the single cell indices that are sampled from steady-state steps
        # Note that sampling should be performed from the end of Concentration list
        # Note that this method should run after building graph(and layering) and should
        be run just once!
        """

        if self.optimize_sampling_:
            state = np.true_divide(30000 - self.safety_steps * self.maxLevels_, self.nSC_)
            if state < self.sampling_state_:
                self.sampling_state_ = state

        # time indices when to collect "single-cell" expression snapshots
        self.scIndices_ = self.rng.integers(low = - self.sampling_state_ * self.nSC_, high = 0, size = self.nSC_)

    def calculate_required_steps_(self, level):
        """
        # Calculates the number of required simulation steps after convergence at each level.
        # safety_steps: estimated number of steps required to reach convergence (same), although it is not neede!
        """
        #TODO: remove this safety step

        # Note to self: as safety measure leaving this safety step to double check
        # that knockouts/knockdowns have reached steady-state
        # however, we do initialize the concentrations correctly so should be fine without it

        return self.sampling_state_ * self.nSC_ + level * self.safety_steps

    def calculate_half_response_(self, level):
        """
        Calculates the half response for all interactions between previous layer
        and current layer
        """

        currGenes = self.level2verts_[level]

        for g in currGenes: # g is list of all bins for a single gene
            c = 0
            if g[0].Type == 'T':
                for interTuple in self.graph_[g[0].ID]['params']:
                    regIdx = interTuple[0]
                    meanArr = self.meanExpression[regIdx]

                    if set(meanArr) == set([-1]):
                        print ("Error: Something's wrong in either layering or simulation. Expression of one or more genes in previous layer was not modeled.")
                        sys.exit()

                    self.graph_[g[0].ID]['params'][c] = (self.graph_[g[0].ID]['params'][c][0], self.graph_[g[0].ID]['params'][c][1], self.graph_[g[0].ID]['params'][c][2], np.mean(meanArr))
                    c += 1
            #Else: g is a master regulator and does not need half response

    def hill_(self, reg_conc, half_response, coop_state, repressive = False):
        """
        So far, hill function was used in the code to model 1 interaction at a time.
        So the inputs are single values instead of list or array. Also, it models repression based on this assumption.
        if decided to make it work on multiple interaction, repression should be taken care as well.
        """
        if reg_conc == 0:
            if repressive:
                return 1
            else:
                return 0
        else:
            if repressive:
                return 1 - np.true_divide(np.power(reg_conc, coop_state), (np.power(half_response, coop_state) + np.power(reg_conc, coop_state)) )
            else:
                return np.true_divide(np.power(reg_conc, coop_state), (np.power(half_response, coop_state) + np.power(reg_conc, coop_state)) )

    def init_gene_bin_conc_ (self, level):
        """
        Initilizes the concentration of all genes in the input level

        Note: calculate_half_response_ should be run before this method
        """

        currGenes = self.level2verts_[level]
        for g in currGenes:
            gID = g[0].ID

            # adjust initial concentratino for possible knockout/knockdown experiment
            if self.kout[gID]:
                assert not self.kdown[gID], "Cannot knockout and knockdown the same gene"
                interv_factor = 0.0
            elif self.kdown[gID]:
                interv_factor = 0.5
            else:
                interv_factor = 1.0

            # initialize at expected steady-state
            if g[0].Type == 'MR':
                allBinRates = self.graph_[gID]['rates']

                for bIdx, rate in enumerate(allBinRates):
                    g[bIdx].append_Conc(np.true_divide(interv_factor * rate, self.decayVector_[g[0].ID]))

            else:
                params = self.graph_[g[0].ID]['params']

                for bIdx in range(self.nBins_):
                    rate = 0
                    for interTuple in params:
                        meanExp = self.meanExpression[interTuple[0], bIdx]
                        rate += np.abs(interTuple[1]) * self.hill_(meanExp, interTuple[3], interTuple[2], interTuple[1] < 0)

                    g[bIdx].append_Conc(np.true_divide(interv_factor * rate, self.decayVector_[g[0].ID]))

    def calculate_prod_rate_(self, bin_list, level):
        """
        calculates production rates for the input list of gene objects in different bins but all associated to a single gene ID
        """
        type = bin_list[0].Type

        if (type == 'MR'):
            rates = self.graph_[bin_list[0].ID]['rates']
            return np.array([rates[gb.binID] for gb in bin_list])

        else:
            params = self.graph_[bin_list[0].ID]['params']
            Ks = [np.abs(t[1]) for t in params]
            regIndices = [t[0] for t in params]
            binIndices = [gb.binID for gb in bin_list]
            currStep = bin_list[0].simulatedSteps_
            lastLayerGenes = np.copy(self.level2verts_[level + 1])
            hillMatrix = np.zeros((len(regIndices), len(binIndices)))

            for tupleIdx, rIdx in enumerate(regIndices):
                regGeneLevel = self.gID_to_level_and_idx[rIdx][0]
                regGeneIdx = self.gID_to_level_and_idx[rIdx][1]
                regGene_allBins = self.level2verts_[regGeneLevel][regGeneIdx]
                for colIdx, bIdx in enumerate(binIndices):
                    hillMatrix[tupleIdx, colIdx] = self.hill_(regGene_allBins[bIdx].Conc[currStep], params[tupleIdx][3], params[tupleIdx][2], params[tupleIdx][1] < 0)

            return np.matmul(Ks, hillMatrix)


    def CLE_simulator_(self, level):

        self.calculate_half_response_(level)
        self.init_gene_bin_conc_(level)
        nReqSteps = self.calculate_required_steps_(level)

        # list of lists of genes at the current level
        # each inner list contains `nBins` elements, representing expressions of genes of a given cell type (bin) at time t
        # level2verts_: {level: [[...], ...., [...]]}
        sim_set = np.copy(self.level2verts_[level]).tolist()

        # simulate expression of each gene at this level (vectorized for all bins)
        while sim_set != []:

            delIndicesGenes = []

            # g: list of gene objects of length 'nBins'
            for gi, bin_list in enumerate(sim_set):

                # gene id (row/col in adjacency matrix)
                gID = bin_list[0].ID

                # level in graph, index in list of expressions per bin (same as gi)
                gLevel, gIDX = self.gID_to_level_and_idx[gID]
                assert level == gLevel, "Levels should match"
                assert gi == gIDX, "index in gene-bin matrix should match"

                # [nBins,] current expressions
                currExp = np.array([gb.Conc[-1] for gb in bin_list])

                # [nBins,] production rate of gene given parents
                # if knocked out, set production rate to 0
                # if knocked down, multiply production rate by 0.5
                if self.kout[gID]:
                    prod_rate = np.zeros(len(currExp))
                elif self.kdown[gID]:
                    prod_rate = 0.5 * self.calculate_prod_rate_(bin_list, level)
                else:
                    prod_rate = self.calculate_prod_rate_(bin_list, level)

                # [nBins,] decay rate
                decay = np.multiply(self.decayVector_[gID], currExp)

                # sample noise
                if self.noiseType_ == 'sp':
                    # This notation is inconsistent with our formulation, dw should
                    # include dt^0.5 as well, but here we multipy dt^0.5 later
                    # [nBins, ]
                    dw = self.rng.normal(size = len(currExp))
                    amplitude = np.multiply (self.noiseParamsVector_[gID] , np.power(prod_rate, 0.5))
                    noise = np.multiply(amplitude, dw)

                elif self.noiseType_ == "spd":
                    # [nBins, ]
                    dw = self.rng.normal(size = len(currExp))
                    amplitude = np.multiply (self.noiseParamsVector_[gID] , np.power(prod_rate, 0.5) + np.power(decay, 0.5))
                    noise = np.multiply(amplitude, dw)


                elif self.noiseType_ == "dpd":
                    # [nBins, ]
                    dw_p = self.rng.normal(size = len(currExp))
                    dw_d = self.rng.normal(size = len(currExp))

                    amplitude_p = np.multiply (self.noiseParamsVector_[gID] , np.power(prod_rate, 0.5))
                    amplitude_d = np.multiply (self.noiseParamsVector_[gID] , np.power(decay, 0.5))

                    noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)

                else:
                    raise KeyError(f"Unknown noise type {self.noiseType_}")

                # [nBins,] change in expression per bin
                dxdt = self.dt_ * (prod_rate - decay) + np.power(self.dt_, 0.5) * noise

                # update expression for each bin
                delIndices = []
                for bIDX, gObj in enumerate(bin_list):

                    # append new concentration level to list of expressions in bin
                    binID = gObj.binID
                    gObj.append_Conc(gObj.Conc[-1] + dxdt[bIDX])
                    gObj.incrementStep()

                    # check whether we collected enough samples
                    if len(gObj.Conc) == nReqSteps:
                        # if so, extract and save expressions at preset time snapshots
                        gObj.set_scExpression(self.scIndices_)
                        self.meanExpression [gID, binID] = np.mean(gObj.scExpression)
                        self.level2verts_[level][gIDX][binID] = gObj
                        delIndices.append(bIDX)

                # remove bins to be simulated when they are done
                sim_set[gi] = [i for j, i in enumerate(bin_list) if j not in delIndices]

                if sim_set[gi] == []:
                    delIndicesGenes.append(gi)

            # remove genes to be simulated if done
            sim_set = [i for j, i in enumerate(sim_set) if j not in delIndicesGenes]


    def simulate(self):
        for level in range(self.maxLevels_, -1, -1):
            self.CLE_simulator_(level)


    def getExpressions(self):
        ret = np.zeros((self.nBins_, self.nGenes_, self.nSC_))
        for l in range(self.maxLevels_ + 1):
            currGeneBins = self.level2verts_[l]
            for g in currGeneBins:
                gIdx = g[0].ID

                for gb in g:
                    ret[gb.binID, gIdx, :] = gb.scExpression

        return ret

    """""""""""""""""""""""""""""""""""""""
    "" This part is to add technical noise
    """""""""""""""""""""""""""""""""""""""
    def outlier_effect(self, scData, outlier_prob, mean, scale):
        """
        Args:
            scData: shape [#cell_types, #genes, #cells_per_type].
        """
        out_indicator = self.rng.binomial(n = 1, p = outlier_prob, size = self.nGenes_)
        outlierGenesIndx = np.where(out_indicator == 1)[0]
        numOutliers = len(outlierGenesIndx)

        #### generate outlier factors ####
        outFactors = self.rng.lognormal(mean = mean, sigma = scale, size = numOutliers)
        ##################################

        scData = np.concatenate(scData, axis = 1)
        for i, gIndx in enumerate(outlierGenesIndx):
            scData[gIndx,:] = scData[gIndx,:] * outFactors[i]

        # return np.split(scData, self.nBins_, axis = 1) # BUG in original code; should return same shape as input
        return np.stack(np.split(scData, self.nBins_, axis = 1))


    def lib_size_effect(self, scData, mean, scale):
        """
        This functions adjusts the mRNA levels in each cell seperately to mimic
        the library size effect. To adjust mRNA levels, cell-specific factors are sampled
        from a log-normal distribution with given mean and scale.

        scData: the simulated data representing mRNA levels (concentrations);
        np.array (#bins * #genes * #cells)

        mean: mean for log-normal distribution

        var: var for log-normal distribution

        returns libFactors ( np.array(nBin, nCell) )
        returns modified single cell data ( np.array(nBin, nGene, nCell) )
        """

        #TODO make sure that having bins does not intefere with this implementation
        ret_data = []

        libFactors = self.rng.lognormal(mean = mean, sigma = scale, size = (self.nBins_, self.nSC_))
        for binExprMatrix, binFactors in zip(scData, libFactors):
            normalizFactors = np.sum(binExprMatrix, axis = 0 )
            binFactors = binFactors / np.where(normalizFactors == 0.0, 1.0, normalizFactors)
            binFactors = binFactors.reshape(1, self.nSC_)
            binFactors = np.repeat(binFactors, self.nGenes_, axis = 0)

            ret_data.append(np.multiply(binExprMatrix, binFactors))


        return libFactors, np.array(ret_data)


    def dropout_indicator(self, scData, shape = 1, percentile = 65):
        """
        This is similar to Splat package

        Input:
        scData can be the output of simulator or any refined version of it
        (e.g. with technical noise)

        shape: the shape of the logistic function

        percentile: the mid-point of logistic functions is set to the given percentile
        of the input scData

        returns: np.array containing binary indactors showing dropouts
        """
        scData = np.array(scData)
        scData_log = np.log(np.add(scData,1))
        log_mid_point = np.percentile(scData_log, percentile)
        prob_ber = np.true_divide (1, 1 + np.exp( -1*shape * (scData_log - log_mid_point) ))

        binary_ind = self.rng.binomial( n = 1, p = prob_ber)

        return binary_ind

    def convert_to_UMIcounts (self, scData):
        """
        Input: scData can be the output of simulator or any refined version of it
        (e.g. with technical noise)
        """
        return self.rng.poisson(scData)

