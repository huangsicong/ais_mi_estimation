# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/simulate.py

import numpy as np
import torch
from torch.distributions import Normal
from torchvision.utils import save_image
from ..utils.experiment_utils import note_taking

from os.path import join as pjoin
""" 
The simulated data is returned as a list. 
In most cases for computational efficiency it's a list of only one batch of data.  
 """


def save_simulated_data(hparams, x_logvar, x_mean, x, i, beta=None):
    if beta is None:
        mean_path = hparams.messenger.readable_dir + '/simulated_mean_iteration_' + str(
            i) + '.png'
        x_path = hparams.messenger.readable_dir + '/simulated_data_iteration_' + str(
            i) + '.png'
    else:
        mean_path = "{}/simulated_mean_beta_{}_iteration_{}.png".format(
            hparams.messenger.readable_dir, beta, str(i))
        x_path = "{}/simulated_data_beta_{}_iteration_{}.png".format(
            hparams.messenger.readable_dir, beta, str(i))
    if hparams.tanh_output:
        x = x / 2 + 0.5
        x_mean = x_mean / 2 + 0.5
    save_image(x.view(-1, *hparams.dataset.input_dims), x_path)
    save_image(x_mean.view(-1, *hparams.dataset.input_dims), mean_path)


def simulate_data(model, batch_size, n_batch, hparams, beta=None):
    """
    Simulate data for a decoder based generative model.
    Args:
        model: model for data simulation
        batch_size: batch size for simulated data
        n_batch: number of batches

    Returns:
        list of batches of torch Tensor pair x, z
    """

    gpu_batches = list()

    if hparams.dataparam is not None:
        loadable_path = pjoin(hparams.output_root_dir, "result_loadable",
                              hparams.dataparam, "simulated_data.pt")
        x_mean_cpu, x_logvar_cpu, x_cpu, z_cpu = torch.load(loadable_path)
        z_cpu.requires_grad_()
        paired_batch = ((x_mean_cpu.to(hparams.device),
                         x_logvar_cpu.to(hparams.device),
                         x_cpu.to(hparams.device)), z_cpu.to(hparams.device))
        note_taking(
            f"Loaded data from {loadable_path}, \n Log variance {x_logvar_cpu}"
        )

    else:
        # NOTE using only one batch for GPU efficiency
        for i in range(1):
            z = torch.randn([batch_size, hparams.model_train.z_size
                             ]).to(device=hparams.device)

            x_mean, x_logvar = model.decode(z)
            std = torch.ones(x_mean.size()).to(hparams.device).mul(
                torch.exp(x_logvar * 0.5))
            x_normal_dist = Normal(loc=x_mean, scale=std)
            x = x_normal_dist.sample().to(device=hparams.device)

            paired_batch = ((x_mean, x_logvar, x), z)
            data_pt_path = pjoin(hparams.messenger.loadable_dir,
                                 "simulated_data.pt")

            # NOTE Only support one batch
            torch.save((x_mean.cpu(), x_logvar.cpu(), x.cpu(), z.cpu()),
                       data_pt_path)
            save_simulated_data(hparams, x_logvar, x_mean, x, i, beta)
            z.requires_grad_()

    gpu_batches.append(paired_batch)

    return gpu_batches
