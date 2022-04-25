import os
import torch
import numpy as np
from torch.utils.data import random_split
"""
Utils for loading data
"""


def split_data(hparams, dset, split_percent):
    """Splits a PyTorch dataset according to the split_percentage 

    Arguments:
        hparams: a Hparam object
        dset: PyTorch dataset to split on 
        split_percent: The percentage of dset that goes into the split

    Returns:
        subdset of size len(dset) * (1-split_percent) and subdset of size len(dset)*(split_percent)
    """
    len_dset = len(dset)
    valt_length = int(len_dset * split_percent)
    train_length = len_dset - valt_length
    train_dset, valt_dset = random_split(
        dset, (train_length, valt_length),
        generator=torch.Generator(device="cpu").manual_seed(
            hparams.random_seed))
    return train_dset, valt_dset


def get_target_cifar(dset, classes):
    if classes == None:
        return dset
    idx = []
    dset.targets = torch.tensor(dset.targets)
    for i in classes:
        if idx == []:
            idx = dset.targets == i
        else:
            idx = idx | (dset.targets == i)
    dset.targets = dset.targets[idx]
    dset.data = dset.data[idx.numpy().astype(np.bool)]
    dset.targets = dset.targets.tolist()
    return dset


def get_target_svhn(dset, classes):
    if classes == None:
        return dset
    idx = []
    for i in classes:
        if idx == []:
            idx = dset.labels == i
        else:
            idx = idx | (dset.labels == i)
    dset.labels = dset.labels[idx]
    dset.data = dset.data[idx]
    return dset


def get_target_imagefolder(path, train, classes, target):
    if classes == None:
        classes = list(range(10))
    for j in classes:  # types of images needed
        src = os.path.join(path, train, str(j))
        dst = os.path.join(path, target, train)
        os.makedirs(os.path.realpath(dst), exist_ok=True)
        if not os.path.exists(os.path.join(os.path.realpath(dst), str(j))):
            os.symlink(os.path.realpath(src),
                       os.path.join(os.path.realpath(dst), str(j)))


def get_target_domainnet(path, train, classes, image_type, target):
    if classes == None:
        classes = list(range(345))
    if image_type == None:
        image_type = [
            'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
        ]
    for j in image_type:  # types of images needed
        for k in classes:  # number of classes needed
            # Source file path
            src = os.path.join(path, train, str(j), str(k))
            # Destination file path
            dst = os.path.join(path, target, train, str(j))
            os.makedirs(os.path.realpath(dst), exist_ok=True)
            if not os.path.exists(os.path.join(os.path.realpath(dst), str(k))):
                os.symlink(os.path.realpath(src),
                           os.path.join(os.path.realpath(dst), str(k)))


def get_target_augdataset(dset_path, split, classes, target):
    if classes == None:  # if not selected, use all classes
        classes = os.listdir(os.path.join(dset_path, split))
    print(classes)

    dst = os.path.join(dset_path, target, split)
    os.makedirs(os.path.realpath(dst), exist_ok=True)

    for k in classes:  # list of classes needed
        src = os.path.join(dset_path, split, str(k))
        dst_class = os.path.join(os.path.realpath(dst), str(k))
        if os.path.exists(dst_class):
            pass
        else:
            os.symlink(os.path.realpath(src), dst_class)
