# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import torch
import torch.nn as nn
from ..utils.vae_utils import log_normal, log_mean_exp, log_normal_likelihood, KL_div
from ..utils.computation_utils import singleton_repeat, singleton_repeat_3d, unit_prior
from ..utils.experiment_utils import note_taking
from ..registry import register, get_encoder, get_decoder
from torch.nn import functional as F
from torch.distributions import Bernoulli, Normal


class VAEBase(nn.Module):

    def __init__(self, hparams, likelihoodfn):
        super().__init__()
        self.enc = get_encoder(hparams)
        self.dec = get_decoder(hparams)
        self.observation_log_likelihood_fn = likelihoodfn
        self.hparams = hparams
        self.x_logvar = self.dec.x_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        logqz = log_normal(z, mu, logvar)
        zeros = torch.zeros_like(z)
        logpz = log_normal(z, zeros, zeros)
        return z, logpz, logqz

    def set_decoder(self, decoder):
        self.dec = decoder

    def set_encoder(self, encoder):
        self.enc = encoder

    def decode(self, z):
        if "gan" in self.hparams.model_name:
            self.dec.decode(z)
        else:
            return self.dec(z)

    def sample(self, device, z=None, batch_size=None, is_train=False):
        with torch.set_grad_enabled(is_train):
            if z is None:
                z = torch.randn(
                    batch_size,
                    self.hparams.model_train.z_size).to(device=device)
            if "conv_decoder_categorical" in self.hparams.decoder_name:
                logits, x_mean, x_logvar = self.decode(z)
                x_samples, _ = self.dec.get_image_sample(logits)
            else:
                x_mean, x_logvar = self.decode(z)
                std = torch.ones_like(x_mean).mul(torch.exp(x_logvar * 0.5))
                x_normal_dist = Normal(loc=x_mean, scale=std)
                x_samples = x_normal_dist.sample().to(device=device)

            if x_logvar == None:
                return x_mean, x_logvar, x_samples, z

            return x_mean, x_logvar.unsqueeze(0).repeat(
                (x_mean.size(0), 1)), x_samples, z

    def encode(self, x):
        input_vector_length = self.hparams.dataset.input_vector_length
        flattened_x = x.view(-1, input_vector_length)
        return self.enc(flattened_x)

    def forward(self, x, num_iwae=1, exact_kl=False):
        input_vector_length = self.hparams.dataset.input_vector_length
        flattened_x = x.view(-1, input_vector_length)
        flattened_x_k = singleton_repeat(flattened_x, num_iwae)
        if self.hparams.dataset.input_dims[0] == 1:
            mu, logvar = self.enc(flattened_x_k)
        else:
            mu, logvar = self.enc(
                flattened_x_k.view(-1, *self.hparams.dataset.input_dims))
        z, logpz, logqz = self.reparameterize(mu, logvar)

        if "vae" in self.hparams.model_name:
            x_mean, x_logvar = self.decode(z)
        else:
            x_mean, x_logvar = self.dec.decode(z)
        x_logvar_full = x_logvar.expand_as(x_mean)
        likelihood = self.observation_log_likelihood_fn(
            flattened_x_k, x_mean, x_logvar)
        if exact_kl:
            kl = KL_div(mu, logvar)
        else:
            kl = logqz - logpz
        elbo = likelihood - kl
        if num_iwae != 1:
            elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
            logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
            logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
            likelihood = log_mean_exp(likelihood.view(num_iwae, -1), dim=0)
        elbo = torch.mean(elbo)
        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)
        likelihood = torch.mean(likelihood)
        return x_mean, elbo, mu, logvar, -likelihood, logqz - logpz, z


@register
def gaussian_observation_vae(hparams):
    """
    The deep VAE model used in the experiments. The decoder variance is not fixed
    This will is also used for blurry vae by setting hparams.blur_std
    """

    class VAE(VAEBase):

        def __init__(self, hparams):
            super().__init__(hparams, log_normal_likelihood)
            if self.x_logvar is not None:
                self.x_logvar.requires_grad = True

    return VAE(hparams)


def get_proper_conv_params(nc, nf, input_dims):
    """Gets nc and nf in case there is an expected combination
    """
    num_dataset_channels = input_dims[0]
    if num_dataset_channels == 3 and (nc != 3 or nf != 64):
        note_taking(f"wrong combo nc: {nc} nf: {nf} specified, "
                    "overwriting nc to 3 and nf to 64")
        nc, nf = 3, 64
    elif num_dataset_channels == 1 and (nc != 1 or nf != 32):
        note_taking(f"wrong channel number {nc} specified, "
                    "overwriting nc to 1 and nf to 32")
        nc, nf = 1, 32
    return nc, nf


@register
def dc_vae(hparams):
    """
    Based upon the DCGAN version of VAE model used in the Likelihood Regret paper
    https://arxiv.org/abs/2003.02977
    Author's code is at https://github.com/XavierXiao/Likelihood-Regret
    """

    class VAE(VAEBase):

        def __init__(self, hparams):
            nc, nf = hparams.conv_params.nc, hparams.conv_params.nf
            hparams.conv_params.nc, hparams.conv_params.nf = get_proper_conv_params(
                nc, nf, hparams.dataset.input_dims)
            super().__init__(hparams, nn.CrossEntropyLoss(reduction='none'))

        def decode(self, z):
            _, mean, logvar = self.dec(z)
            return mean, logvar

        def encode(self, x):
            return self.enc(x)

        def sample(self, device, z=None, batch_size=None, is_train=False):
            with torch.set_grad_enabled(is_train):
                if z is None:
                    z = torch.randn(
                        batch_size,
                        self.hparams.model_train.z_size).to(device=device)
                if "conv_decoder_categorical" in self.hparams.decoder_name:
                    logits, x_mean, x_logvar = self.dec(z)
                    x_samples = self.dec.get_image_sample(logits)[0]
                else:
                    x_mean, x_logvar = self.decode(z)
                    std = torch.ones_like(x_mean).mul(torch.exp(x_logvar *
                                                                0.5))
                    x_normal_dist = Normal(loc=x_mean, scale=std)
                    x_samples = x_normal_dist.sample().to(device=device)

                if x_logvar == None:
                    return x_mean, x_logvar, x_samples, z

                return x_mean, x_logvar.unsqueeze(0).repeat(
                    (x_mean.size(0), 1)), x_samples, z

        def forward(self, x, num_iwae=1, exact_kl=False):
            B = x.size(0)
            x = x.repeat((num_iwae, 1, 1, 1))
            mu, logvar = self.enc(x)
            z, logpz, logqz = self.reparameterize(mu, logvar)
            reconstructed, mean, _ = self.dec(z)
            target = (x.view(-1) * 255).long()  #line from author's code
            recon = reconstructed.view(-1, 256)  #line from author's code
            neg_likelihood = torch.sum(self.observation_log_likelihood_fn(
                recon, target).view(B * num_iwae, -1),
                                       dim=-1)
            if exact_kl:
                kl = KL_div(mu, logvar)
            else:
                kl = logqz - logpz
            elbo = -neg_likelihood - kl
            if num_iwae > 1:
                elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
                logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
                logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
                neg_likelihood = log_mean_exp(neg_likelihood.view(
                    num_iwae, -1),
                                              dim=0)
            elbo, logpz, logqz, neg_likelihood = [
                torch.mean(x) for x in [elbo, logqz, logqz, neg_likelihood]
            ]
            return mean, elbo, mu, logvar, neg_likelihood, logqz - logpz, z

    return VAE(hparams)


@register
def gaussian_observation_vae_fixed_var(hparams):
    """
    The VAE model used for Analytical solution.
    """

    class VAE(VAEBase):

        def __init__(self, hparams):
            super().__init__(hparams, log_normal_likelihood)
            if self.x_logvar is not None:
                self.x_logvar.requires_grad = False

    return VAE(hparams)


@register
def dcvae_gaussian(hparams):

    class VAE(VAEBase):

        def __init__(self, hparams):
            nc, nf = hparams.conv_params.nc, hparams.conv_params.nf
            hparams.conv_params.nc, hparams.conv_params.nf = get_proper_conv_params(
                nc, nf, hparams.dataset.input_dims)
            super().__init__(hparams, log_normal_likelihood)
            if self.x_logvar is not None:
                self.x_logvar.requires_grad = True

        def decode(self, z):
            z = z.view(*z.size(), 1, 1)
            mean, logvar = self.dec(z)
            B = mean.size(0)
            return mean.view(B, -1), logvar

        def encode(self, x):
            return self.enc(x)

        def forward(self, x, num_iwae=1, exact_kl=False):
            B = x.size(0)
            img_dims = x.size()[1:]
            x = x.repeat((num_iwae, 1, 1, 1))

            mu, logvar = self.encode(x)
            z, logpz, logqz = self.reparameterize(mu, logvar)
            x_mean, x_logvar = self.decode(z)
            x_mean = x_mean.view(B * num_iwae, -1)
            x_logvar = x_logvar.expand_as(x_mean)

            likelihood = self.observation_log_likelihood_fn(
                x.view(B * num_iwae, -1), x_mean, x_logvar)
            if exact_kl:
                kl = KL_div(mu, logvar)
            else:
                kl = logqz - logpz
            elbo = likelihood - kl
            if num_iwae != 1:
                elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
                logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
                logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
                likelihood = log_mean_exp(likelihood.view(num_iwae, -1), dim=0)
            elbo = torch.mean(elbo)
            logpz = torch.mean(logpz)
            logqz = torch.mean(logqz)
            likelihood = torch.mean(likelihood)
            return x_mean.view(
                B * num_iwae,
                *img_dims), elbo, mu, logvar, -likelihood, logqz - logpz, z

    return VAE(hparams)
