import torch
import numpy as np
from torch.utils.data import random_split
from .experiment_utils import note_taking


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


def load_checkpoint(checkpoint_path):
    """Returns the loaded dictionary from the checkpoint path
    """
    loaded_dict = torch.load(checkpoint_path)
    return loaded_dict


def load_model(checkpoint_path, model):
    """General function that mutates the model and prints out some
        useful info
    """
    loaded_dict = load_checkpoint(checkpoint_path)
    epoch = loaded_dict["epoch"]
    last_val = loaded_dict["validation_loss_list"][-1]

    model.load_state_dict(loaded_dict["state_dict"])

    note_taking(
        "Model loaded from {} at epoch {} with validation_loss {}".format(
            checkpoint_path, epoch, last_val))
