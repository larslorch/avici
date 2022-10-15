"""CAM algorithm.

Imported from the Pcalg package.
Adapted from:

Author: Diviyan Kalainathan
.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""
import os
import uuid
import warnings
import networkx as nx
from shutil import rmtree
from cdt.causality.graph.model import GraphModel
from pandas import read_csv
from cdt.utils.Settings import SETTINGS
from cdt.utils.R import RPackages, launch_R_script


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'

warnings.formatwarning = message_warning


class CAM_with_score(GraphModel):
    r"""CAM algorithm **[R model]**.

    **Description:** Causal Additive models, a causal discovery algorithm
    relying on fitting Gaussian Processes on data, while considering all noises
    additives and additive contributions of variables.

    **Data Type:** Continuous

    **Assumptions:** The data follows a generalized additive noise model:
    each variable :math:`X_i`  in the graph :math:`\mathcal{G}` is generated
    following the model :math:`X_i = \sum_{X_j \in \mathcal{G}} f(X_j) + \epsilon_i`,
    :math:`\epsilon_i` representing mutually independent noises variables
    accounting for unobserved variables.

    Args:
        score (str): Score used to fit the gaussian processes.
        cutoff (float): threshold value for variable selection.
        variablesel (bool): Perform a variable selection step.
        selmethod (str): Method used for variable selection.
        pruning (bool): Perform an initial pruning step.
        prunmethod (str): Method used for pruning.
        njobs (int): Number of jobs to run in parallel.
        verbose (bool): Sets the verbosity of the output.

    Available scores:
       + nonlinear: 'SEMGAM'
       + linear: 'SEMLIN'

    Available variable selection methods:
       + gamboost': 'selGamBoost'
       + gam': 'selGam'
       + lasso': 'selLasso'
       + linear': 'selLm'
       + linearboost': 'selLmBoost'

    Default Parameters:
       + FILE: '/tmp/cdt_CAM/data.csv'
       + SCORE: 'SEMGAM'
       + VARSEL: 'TRUE'
       + SELMETHOD: 'selGamBoost'
       + PRUNING: 'TRUE'
       + PRUNMETHOD: 'selGam'
       + NJOBS: str(SETTINGS.NJOBS)
       + CUTOFF: str(0.001)
       + VERBOSE: 'FALSE'
       + OUTPUT: '/tmp/cdt_CAM/result.csv'

    .. note::
       Ref:
       Bühlmann, P., Peters, J., & Ernest, J. (2014). CAM: Causal additive
       models, high-dimensional order search and penalized regression. The
       Annals of Statistics, 42(6), 2526-2556.

    .. warning::
       This implementation of CAM does not support starting with a graph.
       The adaptation will be made at a later date.

    Example:
        >>> import networkx as nx
        >>> from cdt.causality.graph import CAM
        >>> from cdt.data import load_dataset
        >>> data, graph = load_dataset("sachs")
        >>> obj = CAM()
        >>> output = obj.predict(data)
    """

    def __init__(self, score='nonlinear', cutoff=0.001, variablesel=True,
                 selmethod='gamboost', pruning=False, prunmethod='gam',
                 njobs=None, verbose=None):
        """Init the model and its available arguments."""
        if not RPackages.CAM:
            raise ImportError("R Package CAM is not available.")

        super(CAM_with_score, self).__init__()
        self.scores = {'nonlinear': 'SEMGAM',
                       'linear': 'SEMLIN'}
        self.var_selection = {'gamboost': 'selGamBoost',
                              'gam': 'selGam',
                              'lasso': 'selLasso',
                              'linear': 'selLm',
                              'linearboost': 'selLmBoost'}
        self.arguments = {'{FOLDER}': '/tmp/cdt_CAM/',
                          '{FILE_TRAIN}': 'train_data.csv',
                          '{FILE_VALID}': 'valid_data.csv',
                          '{TARGETS_TRAIN}': 'targets_train.csv',
                          '{TARGETS_VALID}': 'targets_valid.csv',
                          '{SCORE}': 'SEMGAM',
                          '{VARSEL}': 'TRUE',
                          '{SELMETHOD}': 'selGamBoost',
                          '{PRUNING}': 'TRUE',
                          '{PRUNMETHOD}': 'selGam',
                          '{NJOBS}': str(SETTINGS.NJOBS),
                          '{CUTOFF}': str(0.001),
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': 'result.csv'}
        self.score = score
        self.cutoff = cutoff
        self.variablesel = variablesel
        self.selmethod = selmethod
        self.pruning = pruning
        self.prunmethod = prunmethod
        self.njobs = SETTINGS.get_default(njobs=njobs)
        self.verbose = SETTINGS.get_default(verbose=verbose)

    def get_score(self, train_data, valid_data, train_mask=None,
                  valid_mask=None, **kwargs):
        """Apply causal discovery on data using CAM and return
        a training and a validation score (likelihood).

        Args:
            data (pandas.DataFrame): DataFrame containing the data

        Returns:
            Training score and validation score.
        """

        # Building setup w/ arguments.
        self.arguments['{SCORE}'] = self.scores[self.score]
        self.arguments['{CUTOFF}'] = str(self.cutoff)
        self.arguments['{VARSEL}'] = str(self.variablesel).upper()
        self.arguments['{SELMETHOD}'] = self.var_selection[self.selmethod]
        self.arguments['{PRUNING}'] = str(self.pruning).upper()
        self.arguments['{PRUNMETHOD}'] = self.var_selection[self.prunmethod]
        self.arguments['{NJOBS}'] = str(self.njobs)
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()
        self.arguments['{OUTPUT2}'] = 'result2.csv'

        results = self._run_cam_with_score(train_data, valid_data, train_mask,
                                           valid_mask, verbose=self.verbose)

        dag = results[0]
        train_score = results[1][0][0]
        val_score = results[1][1][0]

        return nx.relabel_nodes(nx.DiGraph(dag),
                                {idx: i for idx, i in enumerate(train_data.columns)}), train_score, val_score


    def convert_masks(self, idxs):
        masks_list = [self.masks[i] for i in idxs]
        masks = torch.ones((idxs.shape[0], self.dim))

        for i, m in enumerate(masks_list):
            for j in m:
                masks[i, j] = 0

        return masks


    def _run_cam_with_score(self, train_data, valid_data, train_mask=None,
                            valid_mask=None, verbose=True):
        """Setting up and running CAM with all arguments."""
        # Run CAM
        id = str(uuid.uuid4())
        os.makedirs('/tmp/cdt_CAM' + id + '/')
        self.arguments['{FOLDER}'] = '/tmp/cdt_CAM' + id + '/'

        def retrieve_result():
            return read_csv('/tmp/cdt_CAM' + id + '/result.csv', delimiter=',').values, read_csv('/tmp/cdt_CAM' + id + '/result2.csv', delimiter=',').values

        try:
            train_data.to_csv('/tmp/cdt_CAM' + id + '/train_data.csv', header=False, index=False)
            valid_data.to_csv('/tmp/cdt_CAM' + id + '/valid_data.csv', header=False, index=False)

            if train_mask is not None:
                train_mask.to_csv('/tmp/cdt_CAM' + id + '/targets_train.csv', header=False, index=False)
                valid_mask.to_csv('/tmp/cdt_CAM' + id + '/targets_valid.csv', header=False, index=False)
                self.arguments['{INTERVENTION}'] = 'TRUE'
            else:
                self.arguments['{INTERVENTION}'] = 'FALSE'

            cam_result = launch_R_script("{}/cam_with_score.R".format(os.path.dirname(os.path.realpath(__file__))),
                                         self.arguments, output_function=retrieve_result, verbose=verbose)

        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_CAM' + id + '')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_CAM' + id + '/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_CAM' + id + '')
        return cam_result


    def _run_cam(self, data, fixedGaps=None, verbose=True):
        """Setting up and running CAM with all arguments."""
        # Run CAM
        id = str(uuid.uuid4())
        os.makedirs('/tmp/cdt_CAM' + id + '/')
        self.arguments['{FOLDER}'] = '/tmp/cdt_CAM' + id + '/'

        def retrieve_result():
            return read_csv('/tmp/cdt_CAM' + id + '/result.csv', delimiter=',').values

        try:
            data.to_csv('/tmp/cdt_CAM' + id + '/data.csv', header=False, index=False)
            cam_result = launch_R_script("{}/R_templates/cam.R".format(os.path.dirname(os.path.realpath(__file__))),
                                         self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_CAM' + id + '')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_CAM' + id + '/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_CAM' + id + '')
        return cam_result
