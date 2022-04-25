# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import numpy as np
import torch
from .experiment_utils import note_taking
from torch.distributions import Normal


def singleton_repeat(x, n):
    """
    Repeat a batch of data n times.
    It's the safe way to repeat
    First add an additional dimension, repeat that dimention, then reshape it back.
    So that later when reshaping, it's guranteed to follow the same tensor convention.
     """
    if n == 1:
        return x
    else:
        singleton_x = torch.unsqueeze(x, 0)
        repeated_x = singleton_x.repeat(n, 1, 1)
        return repeated_x.view(-1, x.size()[-1])


def singleton_repeat_3d(x, n):
    """
    Repeat a batch of data n times.
    It's the safe way to repeat
    First add an additional dimension, repeat that dimention, then reshape it back.
    So that later when reshaping, it's guranteed to follow the same tensor convention.
     """
    if n == 1:
        return x
    else:
        singleton_x = torch.unsqueeze(x, 0)
        repeated_x = singleton_x.repeat(n, 1, 1, 1, 1)
        return repeated_x.view(-1, *x.shape[1:])


def normalized_weights_test(normalized_logws, n_chains, batch_size,
                            unnormalized_logws):
    test_logws_sum = torch.sum(normalized_logws, dim=0, keepdim=True)
    note_taking("Testing whether normalized_logws sum up to 1: {}".format(
        test_logws_sum))


def normalize_logws(logws, batch_size, n_chains, hparams, dim=0):

    if hparams.mixed_precision:
        rearanged_logws = logws.view(n_chains, batch_size,
                                     -1).to(device=hparams.device,
                                            dtype=torch.float64)
    else:
        rearanged_logws = logws.view(n_chains, batch_size, -1)

    chain_sum_logws = torch.logsumexp(rearanged_logws, dim=0, keepdim=True)
    normalized_log_w = rearanged_logws - chain_sum_logws
    normalized_w = torch.exp(normalized_log_w)
    normalizing_constants_logws = chain_sum_logws - torch.log(
        torch.tensor(n_chains).to(hparams.tensor_type))

    if hparams.mixed_precision:
        normalizing_constants_logws = normalizing_constants_logws.to(
            device=hparams.device, dtype=hparams.tensor_type)
        normalized_w = normalized_w.to(device=hparams.device,
                                       dtype=hparams.tensor_type)

    return normalized_w, normalizing_constants_logws


def bits_per_dimension(loss, num_data, input_dims):
    """This also switches the base of the log to base 2
    """
    deno = num_data * np.prod(input_dims) * np.log(2.)
    bits_per_dim = loss / deno
    return bits_per_dim


def get_beta_distribution_params(mean, var):
    """Compute alpha and beta params for the beta distribution given
        mean and variance
    """
    mean_reciprocal = 1 / mean
    alpha = ((1 - mean).div(var) - mean_reciprocal).mul(mean.pow(2))
    beta = alpha.mul(mean_reciprocal - 1)
    return alpha, beta


def negative_log_likelihood(bpd, num_data, input_dims, base_2):
    if base_2:
        nll = bpd * num_data * np.prod(input_dims) * np.log(2)
    else:
        nll = bpd * num_data * np.prod(input_dims)
    return nll


def sig_digit_dict(source_dict, n=3):
    for key, value in source_dict.items():
        source_dict[key] = round(source_dict[key], n)


def sig_digit_dict_py(source_dict, n=3):
    for key, value in source_dict.items():
        source_dict[key] = round(float(source_dict[key]), n)


def unit_prior(batch_size, z_size, hparams):
    std = torch.ones((batch_size, z_size)).to(hparams.device)
    mean = torch.zeros((batch_size, z_size)).to(hparams.device)
    unit_normal_dist = Normal(loc=mean, scale=std)
    return unit_normal_dist