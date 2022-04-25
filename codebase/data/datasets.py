"""
Dataset registry for all the datasets in the code base
Note that each function must return a pytorch dataset to be loaded into dataloaders
"""

from ..registry import register, get_model, get_hparams
from torchvision import datasets

from ..utils.data_utils import *
from ..utils.data_load_utils import *

"""
input: hparams. 
function: knows the reading format and procedure of that type. 
output: dataset instance ready for Dataloader

Note: saving format: torch.save((sample, label), os.path.join(root_dset_dir, dset_hash_name, train, class, str(idx) + '.pt'))

"""


@register
def cifar10(hparams, transform, train):
    dset = datasets.CIFAR10(hparams.data_dir,
                            train=train,
                            download=True,
                            transform=transform)
    dset = get_target_cifar(dset, hparams.dataset.classes)
    return dset


@register
def mnist(hparams, transform, train):
    dset = datasets.MNIST(hparams.data_dir,
                          train=train,
                          download=True,
                          transform=transform)
    dset = get_target(dset, hparams.dataset.classes)
    return dset


@register
def fmnist(hparams, transform, train):
    dset = datasets.FashionMNIST(hparams.data_dir,
                                 train=train,
                                 download=True,
                                 transform=transform)
    dset = get_target(dset, hparams.dataset.classes)
    return dset


@register
def cifar100(hparams, transform, train):
    dset = datasets.CIFAR100(hparams.data_dir,
                             train=train,
                             download=True,
                             transform=transform)
    dset = get_target_cifar(dset, hparams.dataset.classes)
    return dset


@register
def svhn(hparams, transform, train):
    dset = datasets.SVHN(hparams.data_dir,
                         split=get_str(train),
                         download=True,
                         transform=transform)
    dset = get_target_svhn(dset, hparams.dataset.classes)
    return dset

@register
def kmnist(hparams, transform, train):
    dset = datasets.KMNIST(hparams.data_dir,
                           train=train,
                           download=True,
                           transform=transform)
    dset = get_target(dset, hparams.dataset.classes)
    return dset


 