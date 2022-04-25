import torchvision.transforms as transforms

import warnings
from ..registry import register, get_augment
"""Presaved=True means that the dataset has already been saved onto disk
    here, we don't want to apply the totensor call
"""


def get_normalize_transform(hparams):
    normalized_transform = []
    normalize = hparams.dataset.normalize
    if normalize:
        num_channels = hparams.dataset.input_dims[0]
        mean_norm, std_norm = normalize
        normalized_transform.append(
            transforms.Normalize((mean_norm, ) * num_channels,
                                 (std_norm, ) * num_channels))
    return normalized_transform


def tensor_resize_aug_recover_pipeline(augmentation, hparams, saved_dataset):
    """For generating the dataset, the user should apply the normalization
        themselves

    NOTE: only this part of the augmentation gets saved!
    """
    transform_list = []

    # if it's not saved then we apply totensor
    if not saved_dataset:  # data set not saved to disk
        transform_list.append(transforms.ToTensor())

    resize = hparams.dataset.input_dims

    if resize:
        transform_list.append(transforms.Resize(resize[1:]))

    transform_list.append(get_augment("RecoverColorChannelLength", hparams))

    if augmentation is not None:
        transform_list.extend(augmentation)

    return transform_list


@register
def default_data_pipeline(augmentation, hparams, saved_dataset=False):
    normalized_transform, transform_list = [], []
    if not saved_dataset:
        # dataset not saved, need to apply both
        transform_list = tensor_resize_aug_recover_pipeline(
            augmentation, hparams, saved_dataset)

    normalized_transform = get_normalize_transform(hparams)

    return transform_list + normalized_transform


@register
def pixel_cnn_data_pipeline(augmentation, hparams, saved_dataset=False):
    transform_list = []
    if not saved_dataset:
        # dataset not saved, need to apply both
        transform_list = tensor_resize_aug_recover_pipeline(
            augmentation, hparams, saved_dataset)

    normalized_transform = get_normalize_transform(hparams)

    # no normalize specified, weird for pixelcnn
    if len(normalized_transform) == 0:
        num_channels = hparams.dataset.input_dims[0]
        normalized_transform.append(
            transforms.Normalize((0.5, ) * num_channels,
                                 (0.5, ) * num_channels))
        warnings.warn("You didn't specify a normalize for pixelcnn, using the"
                      " default value of Normalize=[0.5,0.5]")

    return transform_list + normalized_transform