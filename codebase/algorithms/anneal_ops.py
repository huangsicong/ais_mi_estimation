# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/ais.py

#!/usr/bin/env python3

import numpy as np
import torch
from ..utils.vae_utils import log_normal_likelihood
from ..utils.experiment_utils import note_taking, save_comparison

from ..utils.mi_utils import Guassian_test

_ANNOPS = dict()


def register(name):

    def add_to_dict(fn):
        global _ANNOPS
        _ANNOPS[name] = fn
        return fn

    return add_to_dict


def get_anneal_operators(target_dist):
    return _ANNOPS[target_dist]()


def distortion2ais(t, distortion):
    return (1. / torch.exp(distortion))**t


@register("variational")
def variational():
    """ 
    Annealing distribution for starting with a learned q. 
    """

    def anneal_dist(z, data, t, model, hparams, task_params, prior_dist):
        """
        Compute unnormalized density for intermediate distributions
        """

        z_zeros = torch.zeros_like(z)
        log_prior = log_normal_likelihood(z, z_zeros, z_zeros)
        variational_posterior = prior_dist.log_prob(z).sum(1).to(
            hparams.device)
        x_mean, x_logvar = model.decode(z)
        x_logvar_full = torch.zeros_like(data) + x_logvar
        log_likelihood = log_normal_likelihood(data, x_mean, x_logvar_full)

        if hparams.messenger.save_post_images:
            save_comparison(data,
                            x_mean,
                            task_params.batch_size,
                            hparams,
                            beta=hparams.messenger.beta,
                            mode=hparams.messenger.mode)
            hparams.messenger.save_post_images = False

        return variational_posterior * (1 - t) + (log_prior +
                                                  log_likelihood) * t

    return anneal_dist


@register("joint_xz")
def joint_xz():
    """ 
    Annealing distribution for using unit prior. 
    """

    def anneal_dist(z, data, t, model, hparams, task_params, prior_dist):
        """
        Compute unnormalized density for intermediate distribution:
        p_t = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        log p_t = log p(z) + t * log p(x|z)
        """

        z_zeros = torch.zeros_like(z)
        log_prior = log_normal_likelihood(z, z_zeros, z_zeros)
        x_mean, x_logvar = model.decode(z)
        x_logvar_full = torch.zeros_like(data) + x_logvar
        log_likelihood = log_normal_likelihood(data, x_mean, x_logvar_full)

        if hparams.messenger.save_post_images:
            save_comparison(data,
                            x_mean,
                            task_params.batch_size,
                            hparams,
                            beta=hparams.messenger.beta,
                            mode=hparams.messenger.mode)
            hparams.messenger.save_post_images = False

        return log_prior + log_likelihood * t

    return anneal_dist
