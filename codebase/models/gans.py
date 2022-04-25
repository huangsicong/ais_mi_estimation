# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang and Yanshuai Cao

#!/usr/bin/env python3

from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class sheldonMLPGenerator(nn.Module):

    def __init__(self,
                 hdim=1024,
                 img_dims=(1, 28, 28),
                 zdim=100,
                 spectral_norm=False,
                 flat_out=True):
        super(sheldonMLPGenerator, self).__init__()

        self.img_dims = img_dims
        out_len = int(np.prod(img_dims))
        self.de1 = nn.Linear(zdim, hdim)
        self.de2 = nn.Linear(hdim, hdim)
        self.de3 = nn.Linear(hdim, hdim)
        self.out = nn.Linear(hdim, out_len)

        self.nl = torch.tanh
        self.default_flat_out = flat_out

    def forward(self, z, flat=None):
        h = self.de1(z)
        h = self.nl(h)

        h = self.de2(h)
        h = self.nl(h)

        h = self.de3(h)
        h = self.nl(h)

        out = self.out(h)
        out = torch.sigmoid(out)

        if ((flat is None and not self.default_flat_out)
                or (flat is not None and not flat)):
            out = out.reshape(z.size(0), *self.img_dims)

        return out

    def decode(self, z):
        return self.forward(z, flat=True), self.x_logvar


class sheldonMLPGenerator3(nn.Module):

    def __init__(self,
                 hdim=512,
                 img_dims=(1, 28, 28),
                 zdim=100,
                 spectral_norm=False,
                 flat_out=True):
        super(sheldonMLPGenerator3, self).__init__()

        self.img_dims = img_dims
        out_len = int(np.prod(img_dims))
        self.de1 = nn.Linear(zdim, hdim)
        self.de2 = nn.Linear(hdim, hdim)
        self.out = nn.Linear(hdim, out_len)

        self.nl = torch.tanh
        self.default_flat_out = flat_out

    def forward(self, z, flat=None):
        h = self.de1(z)
        h = self.nl(h)

        h = self.de2(h)
        h = self.nl(h)

        out = self.out(h)
        out = torch.sigmoid(out)

        if ((flat is None and not self.default_flat_out)
                or (flat is not None and not flat)):
            out = out.reshape(z.size(0), *self.img_dims)

        return out

    def decode(self, z):
        return self.forward(z, flat=True), self.x_logvar


class sheldonMLPGenerator2(nn.Module):

    def __init__(self,
                 hdim=512,
                 img_dims=(1, 28, 28),
                 zdim=100,
                 spectral_norm=False,
                 flat_out=True):
        super(sheldonMLPGenerator2, self).__init__()

        self.img_dims = img_dims
        out_len = int(np.prod(img_dims))
        self.de1 = nn.Linear(zdim, hdim)
        self.out = nn.Linear(hdim, out_len)

        self.nl = torch.tanh
        self.default_flat_out = flat_out

    def forward(self, z, flat=None):
        h = self.de1(z)
        h = self.nl(h)
        out = self.out(h)
        out = torch.sigmoid(out)

        if ((flat is None and not self.default_flat_out)
                or (flat is not None and not flat)):
            out = out.reshape(z.size(0), *self.img_dims)

        return out

    def decode(self, z):
        return self.forward(z, flat=True), self.x_logvar
