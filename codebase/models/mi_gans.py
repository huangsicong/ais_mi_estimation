from __future__ import print_function
from codebase.registry import register
import torch
import torch.nn as nn

# Adopted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


@register
def cifar_G(hparams):

    class Generator(nn.Module):

        def __init__(self, hparams):
            super(Generator, self).__init__()

            self.x_logvar = torch.log(torch.tensor(hparams.model_train.x_var))

            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(hparams.model_train.z_size,
                                   hparams.conv_params.ngf * 4,
                                   4,
                                   1,
                                   0,
                                   bias=False),
                nn.BatchNorm2d(hparams.conv_params.ngf * 4),
                nn.ReLU(),
                nn.ConvTranspose2d(hparams.conv_params.ngf * 4,
                                   hparams.conv_params.ngf * 2,
                                   4,
                                   2,
                                   1,
                                   bias=False),
                nn.BatchNorm2d(hparams.conv_params.ngf * 2),
                nn.ReLU(),
                nn.ConvTranspose2d(hparams.conv_params.ngf * 2,
                                   hparams.conv_params.ngf,
                                   4,
                                   2,
                                   1,
                                   bias=False),
                nn.BatchNorm2d(hparams.conv_params.ngf),
                nn.ReLU(),
                nn.ReLU(),
                nn.ConvTranspose2d(hparams.conv_params.ngf,
                                   hparams.dataset.input_dims[0],
                                   4,
                                   2,
                                   1,
                                   bias=False),
                nn.Tanh())

        def forward(self, input):
            return self.main(input)

        def decode(self, z):
            return self.forward(z.unsqueeze(-1).unsqueeze(-1)).view(
                z.size(0), -1), self.x_logvar

    return Generator(hparams)


@register
def cifar_D(hparams):

    class Discriminator(nn.Module):

        def __init__(self, hparams):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(hparams.dataset.input_dims[0],
                          hparams.conv_params.ngf,
                          4,
                          2,
                          1,
                          bias=False), nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(hparams.conv_params.ngf,
                          hparams.conv_params.ngf * 2,
                          4,
                          2,
                          1,
                          bias=False),
                nn.BatchNorm2d(hparams.conv_params.ngf * 2),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(hparams.conv_params.ngf * 2,
                          hparams.conv_params.ngf * 4,
                          4,
                          2,
                          1,
                          bias=False),
                nn.BatchNorm2d(hparams.conv_params.ngf * 4),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(hparams.conv_params.ngf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid())

        def forward(self, input):
            return self.main(input)

    return Discriminator(hparams)
