import numpy as np
import torch
import torch.nn as nn
from ..utils.cnn_utils import get_layer, get_conv, weights_init
from ..registry import register
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


@register
def linear_decoder(hparams):
    """
    The VAE model used for Analytical solution.
    """

    class Decoder(nn.Module):

        def __init__(self, hparams):
            super(Decoder, self).__init__()
            self.dec = nn.Linear(hparams.model_train.z_size,
                                 hparams.dataset.input_vector_length)
            self.x_logvar = nn.Parameter(torch.log(
                torch.tensor(hparams.model_train.x_var)),
                                         requires_grad=False)
            self.x_logvar = nn.Parameter(torch.log(torch.tensor(1.0)),
                                         requires_grad=False)

        def forward(self, z):
            return self.dec(z), self.x_logvar

    return Decoder(hparams)


@register
def deep_vae_decoder(hparams):
    """
    The deep VAE model used in the experiments.
    """

    class Decoder(nn.Module):

        def __init__(self, hparams):
            super(Decoder, self).__init__()
            layers = [nn.Linear(hparams.model_train.z_size, 1024), nn.Tanh()]
            num_hidden = hparams.num_hidden if hparams.num_hidden is not None else 2
            for i in range(num_hidden):
                layers += [nn.Linear(1024, 1024), nn.Tanh()]
            layers += [
                nn.Linear(1024, hparams.dataset.input_vector_length),
                nn.Sigmoid()
            ]
            self.dec = nn.Sequential(*layers)
            self.x_logvar = nn.Parameter(torch.log(
                torch.tensor(hparams.model_train.x_var)),
                                         requires_grad=True)

        def forward(self, z):
            return self.dec(z), self.x_logvar

    return Decoder(hparams)
