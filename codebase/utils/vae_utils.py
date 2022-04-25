# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/utils.py

from math import pi as pi
import torch
from .experiment_utils import freeze_model
from torch import nn
from torch import optim
from torch.nn import functional as F


def KL_div(mu, logvar):
    """Code taken from
        https://github.com/pytorch/examples/blob/master/vae/main.py
    """
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return KL


def get_vae_optimizer(hparams, model):
    """Gets the optimizer for the model such that the optimizer will
        only have access to the parameters that we want to train on

    Arguments:
        model: a PyTorch model that is callable with enc and dec attributes
    """
    params_list = []
    if hparams.freeze_decoder:
        params_list.extend(model.enc.parameters())
        freeze_model(model.dec)
    elif hparams.freeze_encoder:
        params_list.extend(model.dec.parameters())
        freeze_model(model.enc)
    else:
        params_list.extend(model.parameters())
    optimizer = optim.Adam(
        params_list,
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay if hparams.weight_decay else 0)
    return optimizer


def log_normal_likelihood(x, mean, logvar):
    """Implementation WITH constant
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py

    Args:
        x: [B,Z]
        mean,logvar: [B,Z]

    Returns:
        output: [B]
    """
    dim = mean.size(1)
    return -0.5 * ((logvar + (x - mean)**2 / torch.exp(logvar)).sum(1) +
                   torch.log(torch.tensor(2 * pi)) * dim)


def log_mean_exp(x, dim=1):
    """ based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    """
    max_, _ = torch.max(x, dim, keepdim=True, out=None)
    return torch.log(torch.mean(torch.exp(x - max_),
                                dim)) + torch.squeeze(max_)


def log_mean_exp_batch(x, dim=1, num_batch=1):
    """ based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    """
    max_, _ = torch.max(x, dim, keepdim=True, out=None)
    return torch.log(
        torch.mean(torch.exp(x - max_), dim) / num_batch) + torch.squeeze(max_)


def log_normal(x, mean, logvar):
    """
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    log normal WITHOUT constant, since the constants in p(z)
    and q(z|x) cancels out later
    Args:s
        x: [B,Z]
        mean,logvar: [B,Z]

    Returns:
        output: [B]
    """
    return -0.5 * (logvar.sum(1) +
                   ((x - mean).pow(2) / torch.exp(logvar)).sum(1))
