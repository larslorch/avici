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
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from avici.experiment.methods.dcdi.dag_optim import GumbelAdjacency, GumbelIntervWeight
except ImportError:
    from avici.experiment.methods.dcdi.dag_optim import GumbelAdjacency, GumbelIntervWeight

class BaseModel(nn.Module):
    def __init__(self, num_vars, num_layers, hid_dim, num_params, nonlin="leaky-relu",
                 intervention=False, intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1):
        """
        :param int num_vars: number of variables in the system
        :param int num_layers: number of hidden layers
        :param int hid_dim: number of hidden units per layer
        :param int num_params: number of parameters per conditional *outputted by MLP*
        :param str nonlin: which nonlinearity to use
        :param boolean intervention: if True, use loss that take into account interventions
        :param str intervention_type: type of intervention: perfect or imperfect
        :param str intervention_knowledge: if False, don't use the intervention targets
        :param int num_regimes: total number of regimes
        """
        super(BaseModel, self).__init__()
        self.num_vars = num_vars
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.num_params = num_params
        self.nonlin = nonlin
        self.gumbel = True
        self.intervention = intervention
        self.intervention_type = intervention_type
        self.intervention_knowledge = intervention_knowledge
        self.num_regimes = num_regimes

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        # Those parameter might be learnable, but they do not depend on parents.
        self.extra_params = []

        if not(not self.intervention or \
        (self.intervention and self.intervention_type == "perfect" and self.intervention_knowledge == "known") or \
        (self.intervention and self.intervention_type == "perfect" and self.intervention_knowledge == "unknown") or \
        (self.intervention and self.intervention_type == "imperfect" and self.intervention_knowledge == "known")):
            raise ValueError("Not implemented")

        if not self.intervention:
            print("No intervention")
            self.intervention_type = "perfect"
            self.intervention_knowledge = "known"

        # initialize current adjacency matrix
        self.adjacency = torch.ones((self.num_vars, self.num_vars)) - torch.eye(self.num_vars)
        self.gumbel_adjacency = GumbelAdjacency(self.num_vars)

        if self.intervention_knowledge == 'unknown' and self.intervention:
            self.gumbel_interv_w = GumbelIntervWeight(self.num_vars, self.num_regimes)

        self.zero_weights_ratio = 0.
        self.numel_weights = 0

        # Instantiate the parameters of each layer in the model of each variable
        for i in range(self.num_layers + 1):
            in_dim = self.hid_dim
            out_dim = self.hid_dim

            # first layer
            if i == 0:
                in_dim = self.num_vars

            # last layer
            if i == self.num_layers:
                out_dim = self.num_params

            # if interv are imperfect or unknown, generate 'num_regimes' MLPs per conditional
            if self.intervention and (self.intervention_type == 'imperfect' or
                                      self.intervention_knowledge == 'unknown'):
                self.weights.append(nn.Parameter(torch.zeros(self.num_vars,
                                                             out_dim, in_dim,
                                                             self.num_regimes)))
                self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim,
                                                            self.num_regimes)))
                self.numel_weights += self.num_vars * out_dim * in_dim * self.num_regimes
            # for perfect interv, generate only one MLP per conditional
            elif not self.intervention or self.intervention_type == 'perfect':
                self.weights.append(nn.Parameter(torch.zeros(self.num_vars, out_dim, in_dim)))
                self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim)))
                self.numel_weights += self.num_vars * out_dim * in_dim
            else:
                if self.intervention_type not in ['perfect', 'imperfect']:
                    raise ValueError(f'{intervention_type} is not a valid for intervention type')
                if self.intervention_knowledge not in ['known', 'unknown']:
                    raise ValueError(f'{intervention_knowledge} is not a valid value for intervention knowledge')


    def get_interv_w(self, bs, regime):
        return self.gumbel_interv_w(bs, regime)

    def forward_given_params(self, x, weights, biases, mask=None, regime=None):
        """
        :param x: batch_size x num_vars
        :param weights: list of lists. ith list contains weights for ith MLP
        :param biases: list of lists. ith list contains biases for ith MLP
        :param mask: tensor, batch_size x num_vars
        :param regime: np.ndarray, shape=(batch_size,)
        :return: batch_size x num_vars * num_params, the parameters of each variable conditional
        """
        bs = x.size(0)
        num_zero_weights = 0

        for layer in range(self.num_layers + 1):
            # First layer, apply the mask
            if layer == 0:
                # sample the matrix M that will be applied as a mask at the MLP input
                M = self.gumbel_adjacency(bs)
                adj = self.adjacency.unsqueeze(0)

                if not self.intervention:
                    x = torch.einsum("tij,bjt,ljt,bj->bti", weights[layer], M, adj, x) + biases[layer]
                elif self.intervention_type == "perfect" and self.intervention_knowledge == "known":
                    # the mask is not applied here, it is applied in the loss term
                    x = torch.einsum("tij,bjt,ljt,bj->bti", weights[layer], M, adj, x) + biases[layer]
                else:
                    assert mask is not None, 'Mask is not set!'
                    assert regime is not None, 'Regime is not set!'

                    regime = torch.from_numpy(regime)
                    R = mask

                    if self.intervention_knowledge == "unknown":
                        # sample the matrix R and totally mask the
                        # input of MLPs that are intervened on (in R)
                        self.interv_w = self.gumbel_interv_w(bs, regime)
                        R = self.interv_w
                        M = torch.einsum("bjt,bt->bjt", M, R)

                    # transform the mask format from bs x num_vars
                    # to bs x num_vars x num_regimes, in order to select the
                    # MLP parameter corresponding to the regime
                    R = (1 - R).type(torch.int64)
                    R = R * regime.unsqueeze(1)
                    R = torch.zeros(R.size(0), self.num_vars, self.num_regimes).scatter_(2, R.unsqueeze(2), 1)

                    # apply the first MLP layer with the mask M and the
                    # parameters 'selected' by R
                    w = torch.einsum('tijk, btk -> btij', weights[layer], R)
                    x = torch.einsum("btij, bjt, ljt, bj -> bti", w, M, adj, x)
                    x += torch.einsum("btk,tik->bti", R, biases[layer])

            # 2nd layer and more
            else:
                if self.intervention and (self.intervention_type == "imperfect" or self.intervention_knowledge == "unknown"):
                    w = torch.einsum('tijk, btk -> btij', weights[layer], R)
                    x = torch.einsum("btij, btj -> bti", w, x)
                    x += torch.einsum("btk,tik->bti", R, biases[layer])
                else:
                    x = torch.einsum("tij,btj->bti", weights[layer], x) + biases[layer]

            # count number of zeros
            num_zero_weights += weights[layer].numel() - weights[layer].nonzero().size(0)

            # apply non-linearity
            if layer != self.num_layers:
                x = F.leaky_relu(x) if self.nonlin == "leaky-relu" else torch.sigmoid(x)

        self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)

        return torch.unbind(x, 1)

    def get_w_adj(self):
        """Get weighted adjacency matrix"""
        return self.gumbel_adjacency.get_proba() * self.adjacency

    def reset_params(self):
        with torch.no_grad():
            for node in range(self.num_vars):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('leaky_relu'))
                for i, b in enumerate(self.biases):
                    b = b[node]
                    b.zero_()

    def get_parameters(self, mode="wbx"):
        """
        Will get only parameters with requires_grad == True
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        params = []

        if 'w' in mode:
            weights = []
            for w in self.weights:
                weights.append(w)
            params.append(weights)
        if 'b'in mode:
            biases = []
            for b in self.biases:
                biases.append(b)
            params.append(biases)

        if 'x' in mode:
            extra_params = []
            for ep in self.extra_params:
                if ep.requires_grad:
                    extra_params.append(ep)
            params.append(extra_params)

        return tuple(params)

    def set_parameters(self, params, mode="wbx"):
        """
        Will set only parameters with requires_grad == True
        :param params: tuple of parameter lists to set, the order should be coherent with `get_parameters`
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: None
        """
        with torch.no_grad():
            k = 0
            if 'w' in mode:
                for i, w in enumerate(self.weights):
                    w.copy_(params[k][i])
                k += 1

            if 'b' in mode:
                for i, b in enumerate(self.biases):
                    b.copy_(params[k][i])
                k += 1

            if 'x' in mode and len(self.extra_params) > 0:
                for i, ep in enumerate(self.extra_params):
                    if ep.requires_grad:
                        ep.copy_(params[k][i])
                k += 1

    def get_grad_norm(self, mode="wbx"):
        """
        Will get only parameters with requires_grad == True, simply get the .grad
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        grad_norm = 0

        if 'w' in mode:
            for w in self.weights:
                grad_norm += torch.sum(w.grad ** 2)

        if 'b'in mode:
            for b in self.biases:
                grad_norm += torch.sum(b.grad ** 2)

        if 'x' in mode:
            for ep in self.extra_params:
                if ep.requires_grad:
                    grad_norm += torch.sum(ep.grad ** 2)

        return torch.sqrt(grad_norm)

    def save_parameters(self, exp_path, mode="wbx"):
        params = self.get_parameters(mode=mode)
        # save
        with open(os.path.join(exp_path, "params_"+mode), 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, exp_path, mode="wbx"):
        with open(os.path.join(exp_path, "params_"+mode), 'rb') as f:
            params = pickle.load(f)
        self.set_parameters(params, mode=mode)

    def get_distribution(self, density_params):
        raise NotImplementedError
