from ..registry import (register, get_model, get_hparams, get_augment,
                        get_pipeline)
from .data_utils import get_str
from torchvision import datasets
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from .experiment_utils import load_checkpoint, get_path_from_hparam, logging
import math
import random
from torchvision.transforms import ToPILImage
from codebase.hparams.hparam import Hparam as hp
from codebase.registry import get_dset
from copy import deepcopy


@register
def constant(hparams, transform):

    class constant_dset(Dataset):

        def __init__(self, hparams, transform=None):
            if hparams.dataset.input_dims[0] != 3:
                raise ValueError(
                    "Constant Dataset can only work with 3 colour channels")

            num_samples = hparams.dataset.source_dataset.N
            default_samples = 50000
            if hparams.dataset.source_dataset.train is False:
                default_samples = 10000
            self.num_samples = num_samples if num_samples is not None else default_samples
            self.dims = hparams.dataset.input_dims
            self.transform = transform
            self.pallette = [[31, 119, 180], [255, 127, 14], [44, 160, 44],
                             [214, 39, 40], [148, 103, 189], [140, 86, 75],
                             [227, 119, 194], [127, 127, 127], [188, 189, 34],
                             [23, 190, 207]]

        def __getitem__(self, index):
            temp = torch.zeros(self.dims, dtype=torch.uint8)
            colour = random.choice(self.pallette)
            for i, value in enumerate(colour):
                temp[i] = value
            temp = ToPILImage()(temp)
            if self.transform is not None:
                temp = self.transform(temp)
            return temp, 0

        def __len__(self):
            return self.num_samples

    return constant_dset(hparams, transform)


@register
def noise(hparams, transform):

    class noise_dset(Dataset):
        """
        Only loads in CPU
        """

        def __init__(self, hparams, transform=None):
            num_samples = hparams.dataset.source_dataset.N
            default_samples = 50000
            if hparams.dataset.source_dataset.train is False:
                default_samples = 10000
            self.num_samples = num_samples if num_samples is not None else default_samples
            self.dims = hparams.dataset.input_dims
            self.transform = transform

        def __getitem__(self, index):
            img = torch.zeros(self.dims, dtype=torch.uint8)
            img.random_(0, 256)
            img = transforms.ToPILImage()(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, 0

        def __len__(self):
            return self.num_samples

    return noise_dset(hparams, transform)


@register
def simulate(hparams):
    """
    Simulate data for a decoder based generative model.
    """

    class simulate_data(Dataset):

        def __init__(self, hparams):
            num_samples = hparams.dataset.source_dataset.N
            default_samples = 50000
            if hparams.dataset.source_dataset.train is False:
                default_samples = 10000
            self.num_samples = num_samples if num_samples is not None else default_samples
            self.dims = hparams.dataset.input_dims
            self.hparams = hparams
            self.batch_size = hparams.batch_size
            self.sim_model_hp = hparams.dataset.source_dataset.sim_model
            self.load_sim_model(hparams)
            self.simulate()

        def load_sim_model(self, hparams):

            if "pixel_cnn_pp" in self.sim_model_hp.model_name:
                model = torch.nn.DataParallel(get_model(self.sim_model_hp)).to(
                    hparams.device)
            else:
                model = get_model(self.sim_model_hp).to(hparams.device)
            if self.sim_model_hp.ckpt_name:
                ckpt_path = get_path_from_hparam(self.sim_model_hp.ckpt_name,
                                                 self.sim_model_hp.ckpt_epoch)
                load_checkpoint(ckpt_path,
                                None,
                                False,
                                model,
                                load_rng=False,
                                device=hparams.device)
            model.eval()
            self.model = model

        def simulate(self, is_train=False):

            assert self.dims == self.sim_model_hp.dataset.input_dims, f"The given model does not generate samples of required input_dims, {self.dims} != {self.sim_model_hp.dataset.input_dims}"
            with torch.set_grad_enabled(is_train):
                samples = [[] for i in range(3)]
                targets = [[]]
                z = torch.randn([
                    self.num_samples, self.sim_model_hp.model_train.z_size
                ]).to(device=self.hparams.device)
                for i in range(
                        int(math.ceil(self.num_samples / self.batch_size))):
                    z_batch = z[self.batch_size * i:self.batch_size * (i + 1)]

                    x_mean_batch, x_logvar_batch, x_samples_batch, z_batch = self.model.sample(
                        self.hparams.device, z_batch, is_train=is_train)

                    samples[0].append(x_mean_batch.detach())
                    if x_logvar_batch != None:
                        samples[1].append(x_logvar_batch.detach())
                    samples[2].append(x_samples_batch.detach())
                    targets[0].append(z_batch.detach())

                self.data = [torch.cat(i) for i in samples if i != []
                            ]  # list of tensors
                self.targets = [torch.cat(i) for i in targets if i != []
                               ]  # list of tensors

        def __getitem__(self, index):
            return [i[index] for i in self.data
                   ], [i[index] for i in self.targets]

        def __len__(self):
            return self.num_samples

    return simulate_data(hparams)


def save_one_sample(fpath, idx, sample, label):
    os.makedirs(fpath, exist_ok=True)
    path_name = os.path.join(os.path.join(fpath, str(idx) + '.pt'))
    if os.path.exists(path_name):
        os.remove(path_name)
    torch.save((sample, label), path_name)
    return path_name

def torchdset(hparams, dset_type, data_dir, train, transform):
    temp_hp = deepcopy(hparams)
    temp_hp.dataset.augmentation = None
    dset = get_dset(dset_type, temp_hp, transform, train)
    return dset
