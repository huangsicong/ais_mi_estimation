# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/ais.py
# and the R code in R. Neil's paper https://arxiv.org/pdf/1206.1901.pdf

import numpy as np
import gc
import torch
import os
from ..utils.vae_utils import log_mean_exp, log_normal_likelihood
from ..utils.computation_utils import singleton_repeat
from ..utils.experiment_utils import note_taking, init_dir, save_comparison, set_random_seed
from .hmc import hmc_trajectory, accept_reject
from tqdm import tqdm
from .anneal_ops import get_anneal_operators
from .variational_posterior import compute_vp_reverse
from ..utils.mi_utils import Guassian_test, compute_CI

from os.path import join as pjoin


class AIS_core(object):

    def __init__(self, target_dist, prior_dist, model, task_params, hparams,
                 rep_model):
        self.target_dist = target_dist
        self.model = model
        self.task_params = task_params
        self.hparams = hparams
        self.prior_dist = prior_dist
        self.anneal_dist_lookup = get_anneal_operators(task_params.target_dist)
        self.rep_model = rep_model
        self.prior_dist_obj = None

    def update_prior_dist(self, prior_dist_obj):
        self.prior_dist_obj = prior_dist_obj
        print("updating prior_dist_obj:", prior_dist_obj)

    def anneal_dist(self, z, data, t):
        if "fid" in self.task_params.target_dist:
            return self.anneal_dist_lookup(z, data, t, self.model,
                                           self.hparams, self.task_params,
                                           self.prior_dist_obj, self.rep_model)
        else:
            return self.anneal_dist_lookup(z, data, t, self.model,
                                           self.hparams, self.task_params,
                                           self.prior_dist_obj)

    def U(self, z, batch, t):
        return -self.anneal_dist(z, batch, t)

    def grad_U(self, z, batch, t):
        # grad w.r.t. outputs; mandatory in this case
        grad_outputs = torch.ones(
            self.task_params.num_total_chains).to(device=self.hparams.device)
        U_temp = self.U(z, batch, t)
        # grad = torch.autograd.grad(U_temp, z, grad_outputs=grad_outputs,allow_unused=True)[0]
        grad = torch.autograd.grad(U_temp, z, grad_outputs=grad_outputs)[0]
        # clip by norm to avoid numerical instability
        grad = torch.clamp(
            grad, -self.task_params.num_total_chains *
            self.hparams.model_train.z_size * 100,
            self.task_params.num_total_chains *
            self.hparams.model_train.z_size * 100)
        grad.requires_grad_()
        return grad

    def normalized_kinetic(self, v):
        zeros = torch.zeros_like(v)
        return -log_normal_likelihood(v, zeros, zeros)
        # note: can change to log likelihood without constant

    def logpx_z(self, z, data, model, hparams, task_params):
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
        return log_likelihood

    def logpz(self, z):
        z_zeros = torch.zeros_like(z)
        log_prior = log_normal_likelihood(z, z_zeros, z_zeros)
        return log_prior

    def logqz(self, z, hparams):
        variational_posterior = self.prior_dist_obj.log_prob(z).sum(1).to(
            hparams.device)
        return variational_posterior


def run_ais_chain_simplified(model,
                             loader,
                             mode,
                             schedule,
                             hparams,
                             task_params,
                             start_state=None,
                             init_weights=None,
                             init_step_size=None,
                             init_history=None,
                             init_traj_length=0,
                             step_sizes_dir=None,
                             prior_dist=None,
                             rep_model=None,
                             anneal_ops=None):
    """Compute annealed importance sampling trajectories for a batch of data
    Args:
        model : A trained model in pytorch, ideally pytorch 1.0.
        loader (iterator): iterator or list that returns pairs, with first component being `x`,
            second being z. This code only supports one batch for computational effciency.
        mode (string): run forward or backward chain
        schedule (list or 1D np.ndarray): temperature schedule of the geometric annealling distributions.
            forward chain has increasing values, whereas backward has decreasing values
        task_params: specific hyper parameters for this task, just in 
            case some parameter are different for different tasks
        start_state: the initial z's. If starting from prior, then set this to None
        init_weights the initial AIS weights(in log scale). If starting from prior, then set this to None
        init_step_size: initial step sizes. If starting from prior, then set this to None
        init_history: initial acceptance history. If starting from prior, then set this to None 
        init_traj_length: The current AIS step finished in total. If starting from prior, then set this to 0
        prior_dist: prior distribution. If it's a standard unit Guassian, set this to None. 


    Returns:
        logws:A list of tensors where each tensor contains the log importance weights
        for the given batch of data. Note that weights of independent chains of a datapoint is averaged out. 
        approx_post_zs: The samples at the end of AIS chain
        data_loader: The data actually used in this AIS run
        epsilon: The step sizes 
        accept_hist: The acceptance history of HMC
        current_traj_length: A counter, counting how many steps in total have been run. 
    """

    if hparams.tvo:
        logw_tvo = list()
        logw_tvo_integrated = list()
    else:
        logw_tvo = None
        logw_tvo_integrated = None

    hparams.messenger.mode = mode
    if mode == 'forward':
        approx_post_zs = list()
        if not hparams.messenger.fix_data_simulation:
            data_loader = list()
        else:
            data_loader = loader
    note_taking('=' * 80)
    note_taking('In {} mode'.format(mode).center(80))
    for i, (batch, post_z) in enumerate(loader, 0):
        if hparams.dataset.name == "simulate":
            batch = batch[2]

        if not hparams.messenger.fix_data_simulation:
            flattened_batch = batch.view(
                -1, hparams.dataset.input_vector_length).to(
                    device=hparams.device, )
            batch = singleton_repeat(flattened_batch, task_params.n_chains)

        if init_step_size is None:
            epsilon = 0.01 * torch.ones(
                task_params.num_total_chains).to(device=hparams.device)
        else:
            epsilon = init_step_size

        # accept/reject history for tuning step size
        if init_history is None:
            accept_hist = torch.zeros_like(epsilon)
        else:
            accept_hist = init_history

        if init_weights is None:
            logw = torch.zeros_like(epsilon)
        else:
            logw = init_weights[i]

        if mode == 'forward':
            if start_state is None:
                if prior_dist is not None:
                    prior_dist_obj = compute_vp_reverse(model, batch, hparams)

                    hparams.messenger.prior_dist_obj = prior_dist_obj
                    current_z = prior_dist_obj.sample().requires_grad_()
                    anneal_ops.update_prior_dist(prior_dist_obj)

                else:
                    # NOTE equivalent as resetting the RNG state. Make sure this z is independent of z ~ p(z|x)
                    set_random_seed(hparams.random_seed + 1)
                    current_z = torch.randn(
                        task_params.num_total_chains,
                        hparams.model_train.z_size).to(
                            device=hparams.device).requires_grad_()

            else:
                current_z = start_state[i]

                current_z.requires_grad_()
        else:
            current_z = singleton_repeat(
                post_z, task_params.n_chains).requires_grad_()

        next_z = None
        for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]),
                                          1)):
            current_traj_length = j + init_traj_length
            if step_sizes_dir is not None:
                step_sizes_path = step_sizes_dir + str(
                    current_traj_length) + ".pt"
            else:
                note_taking("WARNING! Empty step size dir. ")

            # overwrite step size if there's a target to load
            if hparams.step_sizes_target is not None:
                epsilon = torch.load(step_sizes_path)

            if j == len(schedule) - 1:
                if i == 0:
                    hparams.messenger.save_post_images = True
            else:
                hparams.messenger.save_post_images = False

            if next_z is not None:
                current_z = next_z
                current_z.requires_grad_()
            with torch.no_grad():

                log_px_z = anneal_ops.logpx_z(current_z, batch, model, hparams,
                                              hparams.rd)
                if hparams.rd.prior_dist is None:
                    logw_t = log_px_z

                else:
                    log_pz = anneal_ops.logpz(current_z)
                    logqz_x = anneal_ops.logqz(current_z, hparams)
                    logw_t = (log_px_z + log_pz - logqz_x)
                logw_t_integrand = (t1 - t0) * logw_t
                logw += logw_t_integrand
                if hparams.tvo:
                    logw_tvo.append(logw_t)
                    logw_tvo_integrated.append(logw)

            current_v = torch.randn(current_z.size()).to(device=hparams.device)
            hmc_z, hmc_v = hmc_trajectory(current_z, batch, t1, current_v,
                                          anneal_ops, epsilon, hparams,
                                          task_params)
            # checkpoint here?
            # accept-reject step

            with torch.no_grad():
                next_z, epsilon, accept_hist = accept_reject(
                    current_z, batch, t1, current_v, hmc_z, hmc_v, epsilon,
                    accept_hist, current_traj_length, anneal_ops, hparams,
                    task_params)

                if hparams.step_sizes_target is None:
                    if hparams.save_step_sizes:
                        torch.save(epsilon, step_sizes_path)
            if hparams.ais_gc is None:
                hparams.ais_gc = 1
            if j % hparams.ais_gc == 0:
                torch.cuda.empty_cache()
                gc.collect()

        chains_npy_path = pjoin(hparams.messenger.loadable_dir,
                                f"{hparams.hparam_set}_{mode}_chains_logw.pt")
        torch.save(logw, chains_npy_path)
        logw_chains = logw.clone()

        logw = log_mean_exp(logw.view(task_params.n_chains, -1), dim=0)

        if mode == 'backward':
            logw = -logw
        else:
            approx_post_zs.append(next_z)
            if not hparams.messenger.fix_data_simulation:
                data_loader.append((batch, post_z))

        if i == hparams.rd.n_batch - 1:
            break

    if mode == 'forward':
        return logw, approx_post_zs, data_loader, logw_chains, epsilon, accept_hist, current_traj_length, logw_tvo, logw_tvo_integrated

    else:
        return logw, logw_chains, logw_tvo, logw_tvo_integrated
