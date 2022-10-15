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
import torch
import numpy as np
from scipy.linalg import expm

try:
    from .utils.gumbel import gumbel_sigmoid
except ImportError:
    from utils.gumbel import gumbel_sigmoid


class TrExpScipy(torch.autograd.Function):
    """
    autograd.Function to compute trace of an exponential of a matrix
    """

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            # send tensor to cpu in numpy format and compute expm using scipy
            expm_input = expm(input.detach().cpu().numpy())
            # transform back into a tensor
            expm_input = torch.as_tensor(expm_input)
            if input.is_cuda:
                # expm_input = expm_input.cuda()
                assert expm_input.is_cuda
            # save expm_input to use in backward
            ctx.save_for_backward(expm_input)

            # return the trace
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            expm_input, = ctx.saved_tensors
            return expm_input.t() * grad_output


def compute_dag_constraint(w_adj):
    """
    Compute the DAG constraint of w_adj
    :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
    """
    assert (w_adj >= 0).detach().cpu().numpy().all()
    h = TrExpScipy.apply(w_adj) - w_adj.shape[0]
    return h


def is_acyclic(adjacency):
    """
    Return true if adjacency is a acyclic
    :param np.ndarray adjacency: adjacency matrix
    """
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0: return False
    return True


class GumbelAdjacency(torch.nn.Module):
    """
    Random matrix M used for the mask. Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    """
    def __init__(self, num_vars):
        super(GumbelAdjacency, self).__init__()
        self.num_vars = num_vars
        self.log_alpha = torch.nn.Parameter(torch.zeros((num_vars, num_vars)))
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.reset_parameters()

    def forward(self, bs, tau=1, drawhard=True):
        adj = gumbel_sigmoid(self.log_alpha, self.uniform, bs, tau=tau, hard=drawhard)
        return adj

    def get_proba(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha)

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 5)


class GumbelIntervWeight(torch.nn.Module):
    """
    Random matrix R used for the intervention in the unknown case.
    Can sample a matrix and backpropagate using the Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    :param int num_regimes: number of regimes in the data
    """
    def __init__(self, num_vars, num_regimes):
        super(GumbelIntervWeight, self).__init__()
        self.num_vars = num_vars
        self.num_regimes = num_regimes

        # the column associated to the observational regime is set to one
        self.log_alpha_obs = torch.ones((num_vars, 1)) * 10000
        self.log_alpha = torch.nn.Parameter(torch.ones((num_vars, num_regimes-1)) * 3)

        self.uniform = torch.distributions.uniform.Uniform(0, 1)

    def forward(self, bs, regime, tau=1, drawhard=True):
        # if observational regime, always return a mask full of one
        log_alpha = torch.cat((self.log_alpha_obs, self.log_alpha), dim=1)
        regime = regime.type(torch.LongTensor)
        interv_w = gumbel_sigmoid(log_alpha[:,regime], self.uniform, 1, tau=tau, hard=drawhard)
        interv_w = interv_w.squeeze().transpose(0, 1)
        return interv_w

    def get_proba(self):
        """Returns probability of getting one"""
        log_alpha = torch.cat((self.log_alpha_obs, self.log_alpha), dim=1)
        return torch.sigmoid(log_alpha)
