# -*- coding: utf-8 -*-
import torch


def exp(x, alpha):
    """Element wise non-linearity
    kernel_exp is defined as k(x)=exp(alpha * (x-1))
    return:
        same shape tensor as x
    """
    return torch.exp(alpha*(x - 1.))


def add_exp(x, alpha):
    return 0.5 * (exp(x, alpha) + x)


kernels = {
    "exp": exp,
    "add_exp": add_exp
}
