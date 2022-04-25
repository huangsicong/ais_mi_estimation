import os
from copy import deepcopy
import hashlib
from torchvision.utils import save_image
from .experiment_utils import note_taking

KEYS_TO_HASH = {
    "seed",
    "source_dataset",
    "augmentation",
    "input_dims",
}


def get_target(dset, classes):
    if classes == None:
        return dset
    idx = []
    for i in classes:
        if idx == []:
            idx = dset.targets == i
        else:
            idx = idx | (dset.targets == i)
    dset.targets = dset.targets[idx]
    dset.data = dset.data[idx]
    return dset


def visualize_one_batch(
        fpath,
        data,
        batchsize=16):  # remove the one in experiment_utils and its references
    """data in tensor format"""
    n = min(data.size(0), batchsize)
    save_image(data[:n], fpath, nrow=n)


def hash_dataset_name(data_hp):
    if data_hp.train is not None and data_hp.source_dataset.train is not None:
        if data_hp.train != data_hp.source_dataset.train:
            note_taking(
                "Source dataset train value is {} but dataset train value is {}, overwriting the dataset train value to {}"
                .format(data_hp.source_dataset.train, data_hp.train,
                        data_hp.source_dataset.train))
            data_hp.train = data_hp.source_dataset.train
    temp = deepcopy(data_hp.to_dict())

    dict_to_hash = {
        k: temp[k] for k in sorted(temp.keys()) if k in KEYS_TO_HASH
    }
    encoded = str(dict_to_hash).encode('utf-8')
    return hashlib.sha224(encoded).hexdigest()


def get_str(train):
    return "train" if train else "test"


def get_dset_path(hparams):
    train_bool = hparams.dataset.source_dataset.train
    hash_name = hash_dataset_name(hparams.dataset)
    for i in os.scandir(hparams.data_dir):
        if f'_{hash_name}' in i.name:
            fpath = os.path.join(hparams.data_dir, i.name, get_str(train_bool))
            return fpath
