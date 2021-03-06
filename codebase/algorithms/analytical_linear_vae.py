# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import os
import torch
from ..utils.vae_utils import log_normal_likelihood
from ..utils.experiment_utils import note_taking, log_down_likelihood
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.distributions import Normal
from math import pi as pi
from torch.distributions.multivariate_normal import MultivariateNormal
from ..utils.anneal_schedules import get_schedule


def extract_Wb(model, hparams):
    """ 
    Extract the weights and bias for analytical solution.
    """
    decoder_weights = model.dec.dec.weight
    decoder_bias = model.dec.dec.bias
    return decoder_weights, decoder_bias


def plot_analytical_post(mean, covariance, beta, hparams):
    """ Plot analytical posterior for visualization. """
    mu = mean[:, 0]
    cov = covariance

    m = MultivariateNormal(mu, cov)
    samples = m.sample([500])
    nbins = 300
    x = np.linspace(-2, 2, nbins)
    y = np.linspace(-2, 2, nbins)
    x_grid, y_grid = np.meshgrid(np.linspace(-2, 2, nbins),
                                 np.linspace(-2, 2, nbins))
    samples_grid = np.vstack([x_grid.flatten(), y_grid.flatten()]).transpose()
    density = torch.exp(
        m.log_prob(
            torch.from_numpy(samples_grid).to(hparams.tensor_type).to(
                hparams.device)))

    plt.contourf(x_grid,
                 y_grid,
                 density.view((nbins, nbins)).detach().cpu().numpy(),
                 20,
                 cmap='jet')
    plt.colorbar()
    path = hparams.messenger.arxiv_dir + hparams.hparam_set + "_anaytical_q_b{}.pdf".format(
        beta)
    plt.savefig(path)
    plt.close()


def analytical_q_svd(data, decoder_weights, decoder_bias, hparams, beta=1):
    '''
    Compute the mean and covariance of the analytical q_beta with SVD

    '''

    W = decoder_weights
    b = decoder_bias

    WT = torch.t(W)
    I_x = torch.eye(data.size()[-1]).to(device=hparams.device)
    I_z = torch.eye(W.size()[-1]).to(device=hparams.device)
    data = torch.t(data)
    b = torch.unsqueeze(b, dim=1)
    U, D, V = torch.svd(W)
    denominator = 1. / beta + D**2
    diagonal = torch.diag(torch.div(D, denominator))
    core = torch.matmul(torch.matmul(V, diagonal), torch.t(U))
    mu = torch.matmul(core, (data - b))
    cov = I_z - torch.matmul(core, W)

    return mu, cov, D


def analytical_direct(decoder_weights, hparams):
    '''
    Compute the analytical H

    '''

    W = decoder_weights
    WT = torch.t(W)
    k = W.size()[0]
    # note_taking(f'k={k}')
    I_x = torch.eye(W.size()[0]).to(device=hparams.device)

    cov = I_x + torch.matmul(W, WT)
    det = torch.det(cov)
    MI = 0.5 * torch.log(det)
    cH = k / 2. + (k / 2.) * torch.log(torch.tensor(2. * pi))
    H = cH + MI

    return cH, H, MI


def analytical_q_cholesky(data,
                          decoder_weights,
                          decoder_bias,
                          hparams,
                          beta=1):
    '''
    Compute the mean and covariance of the analytical q_beta with cholesky decomposition

    '''

    W = decoder_weights
    b = decoder_bias

    WT = torch.t(W)
    I_x = torch.eye(data.size()[-1]).to(device=hparams.device)
    I_z = torch.eye(W.size()[-1]).to(device=hparams.device)
    data = torch.t(data)

    b = torch.unsqueeze(b, dim=1)
    subcore = torch.matmul(W, WT) + (1. / beta) * I_x

    L = torch.cholesky(subcore, upper=False)
    LT_XT = torch.trtrs(W, L, upper=False)[0]
    X_T = torch.trtrs(LT_XT, torch.t(L), upper=True)[0]

    core = torch.t(X_T)

    mu = torch.matmul(core, (data - b))

    cov = I_z - torch.matmul(core, W)

    return mu, cov


def analytical_q(data, decoder_weights, decoder_bias, hparams, beta=1):
    '''
    Compute the mean and covariance of the analytical q_beta naively 
    '''

    W = decoder_weights
    b = decoder_bias
    WT = torch.t(W)
    I_x = torch.eye(data.size()[-1]).to(device=hparams.device)
    I_z = torch.eye(W.size()[-1]).to(device=hparams.device)
    data = torch.t(data)
    b = torch.unsqueeze(b, dim=1)
    # print("beta", beta)
    subcore = torch.matmul(W, WT) + (1. / beta) * I_x
    subsubcore_woodb = I_z + beta * torch.matmul(WT, W)
    _, s, _ = torch.svd(subsubcore_woodb)
    #Check condition number to ensure numerical stability.
    condition_number = torch.max(s) / torch.min(s)
    note_taking("Condition number for beta={} is: {}".format(
        beta, condition_number),
                print_=False)
    inversed_subsubcore_woodb = torch.inverse(subsubcore_woodb)
    inversed_subcore = beta * I_x - (beta**2) * torch.matmul(
        torch.matmul(W, inversed_subsubcore_woodb), WT)
    core = torch.matmul(WT, inversed_subcore)
    mu = torch.matmul(core, (data - b))
    cov = I_z - torch.matmul(core, W)

    return mu, cov


def analytical_rate_point(data, mu, cov, beta, singular_values):
    '''
    compute the optimal rate for a batch of data
    '''

    latent_size = int(list(cov.size())[0])

    if singular_values is not None:
        first_term = torch.log(torch.tensor(
            1. / beta)) * singular_values.size()[0]
        second_term = torch.sum(torch.log(singular_values**2 + (1. / beta)))
        log_det = first_term - second_term
        trace_cov = torch.sum((1. / beta) / (singular_values**2 + (1. / beta)))
        # print("trace_cov", trace_cov)

    else:
        det_cov = torch.det(cov)
        log_det = torch.log(det_cov)
        trace_cov = torch.trace(cov)

    mu_product = torch.sum(torch.mul(mu, mu), dim=0)
    mu_batch_mean = torch.mean(mu_product)
    rate = 0.5 * (trace_cov + mu_batch_mean - latent_size - log_det)
    return rate


def analytical_distortion_point(data, mu, cov, decoder_weights, decoder_bias):
    '''
    compute the optimal distortion for a batch of data
    data.size() torch.Size([100, 784])
    '''
    latent_size = int(list(data.size())[-1])
    log_const = (latent_size / 2.) * torch.log(torch.tensor(2. * pi))
    W = decoder_weights
    b = decoder_bias
    WT = torch.t(W)
    b = torch.unsqueeze(b, dim=0)
    xb_dot_product = torch.sum(torch.mul((data - b), (data - b)), dim=1)
    xb_dot_batch_mean = torch.mean(xb_dot_product)
    cross_term_batch = torch.sum(torch.mul(torch.t(torch.matmul(W, mu)),
                                           (data - b)),
                                 dim=1)
    cross_term_batch_mean = torch.mean(cross_term_batch)
    E_Y = torch.matmul(W, mu)
    cov_Y = torch.matmul(torch.matmul(W, cov), WT)
    E_Y_squared_batch = torch.sum(torch.mul(E_Y, E_Y), dim=0)
    E_Y_squared_batch_mean = torch.mean(E_Y_squared_batch)
    E_Wz = E_Y_squared_batch_mean + torch.trace(cov_Y)
    distortion = log_const + 0.5 * xb_dot_batch_mean - cross_term_batch_mean + 0.5 * E_Wz

    return distortion


def analytical_rate_distortion(decoder_weights,
                               decoder_bias,
                               rd_data,
                               data_type,
                               n_batch,
                               hparams,
                               model=None):
    # rd_params = hparams.rd
    rate_list = list()
    distortion_list = list()
    for i, (data, _) in enumerate(rd_data, 1):
        # print("data_type", data_type)
        if data_type == "simulate":
            data = data[2]
            # print("data", data.shape)
        if data.size()[-1] != hparams.dataset.input_vector_length:
            data = data.view(-1, hparams.dataset.input_vector_length).to(
                device=hparams.device, )

        if hparams.cholesky:
            q_mean, q_cov = analytical_q_cholesky(data, decoder_weights,
                                                  decoder_bias, hparams,
                                                  hparams.messenger.beta)

        elif hparams.svd:
            q_mean, q_cov, D = analytical_q_svd(data, decoder_weights,
                                                decoder_bias, hparams,
                                                hparams.messenger.beta)

        else:
            q_mean, q_cov = analytical_q(data, decoder_weights, decoder_bias,
                                         hparams, hparams.messenger.beta)

        rate = analytical_rate_point(data, q_mean, q_cov,
                                     hparams.messenger.beta,
                                     D if hparams.svd else None)
        distortion = analytical_distortion_point(data, q_mean, q_cov,
                                                 decoder_weights, decoder_bias)
        rate_list.append(rate)
        distortion_list.append(distortion)
        if i == n_batch:
            break

    cated_rate = torch.stack(rate_list, dim=0)
    cated_distortion = torch.stack(distortion_list, dim=0)

    if hparams.messenger.beta == 1:
        analytical_elbo = -cated_rate - cated_distortion
        note_taking(
            "(analytical log-likelihood for each batch of {} data: {}".format(
                data_type,
                analytical_elbo.cpu().numpy()),
            print_=False)

    rate_expecation = torch.mean(cated_rate).cpu().numpy()
    distortion_expectation = torch.mean(cated_distortion).cpu().numpy()

    return rate_expecation, distortion_expectation


class MI_Analytical:
    """ Compute MI for an Analytical VAE """

    def __init__(self, data_loader, model, hparams, to_print=False):
        self.hparams = hparams
        self.model = model
        self.data_loader = data_loader
        self.results = {}
        self.results_batch = {}
        self.t_schedule = get_schedule(hparams.rd)
        self.decoder_weights, self.decoder_bias = extract_Wb(model, hparams)

    def run_analytical_mi(self, model, hparams, data_loader, n_batch):

        with torch.no_grad():
            analytic_rate, analytic_distortion = analytical_rate_distortion(
                self.decoder_weights, self.decoder_bias, data_loader,
                hparams.dataset.name, n_batch, hparams, model)
            log_down_likelihood(analytic_rate, analytic_distortion,
                                hparams.dataset.name, hparams)

        return analytic_rate, analytic_distortion

    def run(self):
        analytical_cH_direct, analytical_H_direct, analytical_MI_direct = analytical_direct(
            self.decoder_weights, self.hparams)
        R, D = self.run_analytical_mi(self.model,
                                      self.hparams,
                                      self.data_loader,
                                      n_batch=self.hparams.mi.n_batch)
        R_batch, D_batch = self.run_analytical_mi(self.model,
                                                  self.hparams,
                                                  self.data_loader,
                                                  n_batch=1)

        H = R + D
        H_batch = R_batch + D_batch
        self.results['R'] = R
        self.results['D'] = D
        self.results['H'] = H
        self.results['R_batch'] = R_batch
        self.results['D_batch'] = D_batch
        self.results['H_batch'] = H_batch
        self.results['analytical_MI_direct'] = analytical_MI_direct
        self.results['analytical_H_direct'] = analytical_H_direct
        self.results['analytical_cH_direct'] = analytical_cH_direct
        summery_npy_path = self.hparams.messenger.arxiv_dir + f"{self.hparams.dataset.name}_Analytical.npz"
        np.savez(summery_npy_path, self.results)
        return self.results