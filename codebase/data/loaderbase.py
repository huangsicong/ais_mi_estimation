import torch
from torchvision import transforms
from torch.utils.data import random_split
from ..registry import get_dset, get_augment, get_pipeline
from ..utils.experiment_utils import note_taking
from typing import List
from ..utils.data_load_utils import split_data


class LoaderBase:
    """Base class for all objects that manage dataloaders

    Arguments:
        hparams: an instance of the Hparam object
        overwrite_transforms(iterable), a list of transforms,
            must contain everything you need
    """

    def __init__(self, hparams, augmentation: List = None):
        no_workers = hparams.num_workers if hparams.num_workers is not None else 4
        pin_memory = hparams.pin_memory if hparams.pin_memory else True

        # TODO Add documentation
        transform_list = get_pipeline(augmentation, hparams)
        self.transform = transforms.Compose(transform_list)

        self.loader_args = {
            'num_workers': no_workers,
            'pin_memory': pin_memory
        }
        self.hparams = hparams

    def _get_loader(self, dset, shuffle, overwrite_batch_size=None):
        """Private function to get a dataloader from a dataset object using
        the appropriate parameters in hparams

        Arguments:
            dset: a pytorch dataset object
            shuffle (boolean): to shuffle the dataset when loading
            overwrite_batch_size (int): batch size for the loader,
                can be None to use hparam setting
        """
        if overwrite_batch_size:
            batch_size = overwrite_batch_size
        else:
            batch_size = self.hparams.model_train.batch_size
        loader = torch.utils.data.DataLoader(dset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             **self.loader_args)
        return loader

    def get_loader(self,
                   name,
                   istrain=True,
                   shuffle=False,
                   overwrite_batch_size=None):
        """
        Get dataloader with the specified dataset name using hparam settings

        Arguments:
            name (string): the name of the dataset
            istrain (boolean): indicates if we want to get the training dataloader or test
            shuffle (boolean): to shuffle the dataset when loading
            overwrite_batch_size (int): batch size for the loader,
                can be None to use hparam setting
        """
        dset = get_dset(name, self.hparams, self.transform, istrain)
        return self._get_loader(dset,
                                shuffle=shuffle,
                                overwrite_batch_size=overwrite_batch_size)

    def get_test_loader(self, name, shuffle=False, overwrite_batch_size=None):
        """A wrapper function to get a PyTorch dataloader to the test set
            of the specified dataset name

        Arguments:
            name (string): the name of the dataset
            shuffle (boolean): to shuffle the dataset when loading
            overwrite_batch_size (int): batch size for the loader,
                can be None to use hparam setting
        """
        return self.get_loader(name,
                               istrain=False,
                               shuffle=shuffle,
                               overwrite_batch_size=overwrite_batch_size)

    def get_train_loader(self, name, shuffle=True, overwrite_batch_size=None):
        """A wrapper function to get a PyTorch dataloader to the train set
            of the specified dataset name

        NOTE: This will load the train set specified by the dataset registry
            so it may load the entire trainset without the validation split.
            If you are looking for the validation loader as well, call
            get_train_and_val

        Arguments:
            name (string): the name of the dataset
            shuffle (boolean): to shuffle the dataset when loading
            overwrite_batch_size (int): batch size for the loader,
                can be None to use hparam setting
        """
        return self.get_loader(name,
                               istrain=True,
                               shuffle=shuffle,
                               overwrite_batch_size=overwrite_batch_size)

    def get_train_and_val(self,
                          percent_val=0.1,
                          shuffle_train=True,
                          shuffle_val=False):
        """Gets the train and validation dataloader using the hparams parameters

        Arguments:
            percent_val (float): percent of the training loader we want to use as validation,
                range between 0 and 1
            shuffle (boolean): to shuffle the dataset when loading
        """
        train_dset = get_dset(self.hparams.dataset.source_dataset.type,
                              self.hparams, self.transform, True)
        if self.hparams.dataset.source_dataset.val_name:
            val_dset = get_dset(self.hparams.dataset.source_dataset.val_name,
                                self.hparams, self.transform, True)
        else:
            train_dset, val_dset = split_data(self.hparams, train_dset,
                                              percent_val)
        return self._get_loader(train_dset,
                                shuffle=shuffle_train), self._get_loader(
                                    val_dset, shuffle=shuffle_val)

    def get_train_dset(self):
        return get_dset(self.hparams.dataset.source_dataset.type, self.hparams,
                        self.transform, True)

    def get_val_loader(self, percent_val=0.1, shuffle=False):
        """Gets the validation dataloader using the hparams parameters

        Arguments:
            percent_val (float): percent of the training loader we want to use as validation,
                range between 0 and 1, used when the dataset doesn't come with
                a seperate validation set
            shuffle (boolean): to shuffle the dataset when loading
        """
        _, val_loader = self.get_train_and_val(percent_val, shuffle)
        return val_loader
