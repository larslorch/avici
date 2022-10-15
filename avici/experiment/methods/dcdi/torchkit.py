"""
Copyright Chin-Wei Huang
"""
import numpy as np
import torch

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

delta = 1e-6
c = - 0.5 * np.log(2 * np.pi)


def log(x):
    return torch.log(x * 1e2) - np.log(1e2)


def log_normal(x, mean, log_var, eps=0.00001):
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + c


def logsigmoid(x):
    return -softplus(-x)


def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    def maximum(x):
        return x.max(axis)[0]

    A_max = oper(A, maximum, axis, True)

    def summation(x):
        return sum_op(torch.exp(x - A_max), axis)

    B = torch.log(oper(A, summation, axis, True)) + A_max
    return B


def oper(array, oper, axis=-1, keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for s in array.size():
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def softplus(x):
    return F.softplus(x) + delta


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


class BaseFlow(torch.nn.Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
                np.ones((n, self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


class SigmoidFlow(BaseFlow):
    """
    Layer used to build Deep sigmoidal flows

    Parameters:
    -----------
    num_ds_dim: uint
        The number of hidden units

    """

    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim

    def act_a(self, x):
        return softplus(x)

    def act_b(self, x):
        return x

    def act_w(self, x):
        return softmax(x, dim=2)

    def forward(self, x, logdet, dsparams, mollify=0.0, delta=delta):
        ndim = self.num_ds_dim
        # Apply activation functions to the parameters produced by the hypernetwork
        a_ = self.act_a(dsparams[:, :, 0: 1 * ndim])
        b_ = self.act_b(dsparams[:, :, 1 * ndim: 2 * ndim])
        w = self.act_w(dsparams[:, :, 2 * ndim: 3 * ndim])

        a = a_ * (1 - mollify) + 1.0 * mollify
        b = b_ * (1 - mollify) + 0.0 * mollify

        pre_sigm = a * x[:, :, None] + b  # C
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)  # D
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)  # Logit function (so H)
        xnew = x_

        logj = F.log_softmax(dsparams[:, :, 2 * ndim: 3 * ndim], dim=2) + \
               logsigmoid(pre_sigm) + \
               logsigmoid(-pre_sigm) + log(a)

        logj = log_sum_exp(logj, 2).sum(2)

        logdet_ = logj + np.log(1 - delta) - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
        logdet += logdet_

        return xnew, logdet
