import os
from pathlib import Path
import warnings
import random
import numpy as np
import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from codebase.registry import get_augment
from ..data.loaderbase import LoaderBase
from .experiment_utils import note_taking


def save_from_dset(dset,
                   save_name,
                   readable_dir,
                   arxiv_dir,
                   max_num_to_save=16,
                   tanh=False):
    """Assumes that the dset returns PyTorch tensors,
        saves the first max_num_to_save images as .png and .pt
    """
    img_list = [dset[i][0] for i in range(min(max_num_to_save, len(dset)))]
    imgs = torch.stack(img_list, dim=0)
    if tanh:
        note_taking(
            f"save_from_dset Comparison before re-normalization max={imgs.max()}, min={imgs.min()} "
        )
        imgs = imgs / 2 + 0.5
        note_taking(
            f"save_from_dset Comparison after re-normalization max={imgs.max()}, min={imgs.min()} "
        )
        note_taking("...Data renoramlzied from [-1,1] to [0,1] and saved. ")
    save_image(imgs, os.path.join(readable_dir, f"{save_name}.png"))
    torch.save(imgs, os.path.join(arxiv_dir, f"{save_name}.pt"))


def save_input_batch(hparams, max_num_to_save=16):
    """Saves both the augmented and raw input batch
    """
    readable_dir = hparams.messenger.readable_dir
    arvix_dir = hparams.messenger.arxiv_dir

    # save the raw input images
    raw_loader = LoaderBase(hparams)
    raw_train_dset = raw_loader.get_train_dset()
    save_from_dset(raw_train_dset, "raw_input", readable_dir, arvix_dir,
                   max_num_to_save, hparams.tanh_output)

    # if augmentation exists, save that too
    if hparams.dataset.augmentation is not None:
        aug_loader = LoaderBase(hparams,
                                augmentation=[
                                    get_augment(
                                        hparams.dataset.augmentation.type,
                                        hparams)
                                ])
        aug_train_dset = aug_loader.get_train_dset()
        save_from_dset(aug_train_dset, "aug_input", readable_dir, arvix_dir,
                       max_num_to_save)


def save_checkpoint(checkpoint_path, save_dict):
    """A more general way of saving checkpoints
    """
    preemptable_torch_save(checkpoint_path, save_dict)
    note_taking(
        f"Checkpoint at {checkpoint_path} with keys: {list(save_dict.keys())} saved"
    )


def preemptable_torch_save(save_path, save_dict):
    save_path = Path(save_path).resolve()
    temp_path = save_path.parent / "_you_got_preempted.pt"
    torch.save(save_dict, temp_path)

    if not os.path.exists(save_path):
        save_path.touch(exist_ok=True)

    temp_path.replace(save_path)


def preemptable_array_save(save_path, np_array):
    """Saves a numpy array that is preemption safe
    """
    save_path = Path(save_path).resolve()
    temp_path = save_path.parent / "_you_got_preempted.npy"
    np.save(temp_path, np_array)

    if not save_path.exists():
        save_path.touch(exist_ok=True)

    temp_path.replace(save_path)


def save_gif(img_list, save_name, duration=1e-6, loop=0):
    """Note that the images must be uint8 and be PIL images and save_name
        should include the .gif suffix
    """
    if len(img_list) > 0:
        img_list[0].save(fp=save_name,
                         format="GIF",
                         append_images=img_list[1:],
                         save_all=True,
                         duration=duration,
                         loop=loop)


def save_histogram(indist_array,
                   ood_array,
                   save_dir,
                   model_name,
                   indist_name,
                   ood_name,
                   axisname,
                   min_scale,
                   max_scale,
                   append=0,
                   y_limit=7000):
    """
    Adopted from https://github.com/XavierXiao/Likelihood-Regret
    """
    all_bins = np.linspace(min_scale, max_scale, 50)
    fig, ax = plt.subplots()
    ax.hist(indist_array, all_bins, alpha=0.5, label='In-distribution')
    ax.hist(ood_array, all_bins, alpha=0.5, label='OOD')
    ax.legend(loc='upper right')
    ax.set_title(f'{model_name} \n {indist_name} vs {ood_name}')
    ax.set_xlabel(str(axisname))
    ax.set_ylabel('Amount')
    ax.set_ylim(0, y_limit)
    axisname = axisname.replace(" ", "_")
    fig_path = os.path.join(save_dir, f"{model_name}_{axisname}_{append}.png")
    fig.savefig(fig_path)
    plt.clf()
    plt.cla()
    plt.close('all')
    return fig_path


class RNGmanager:

    def __init__(self, save_path, temp_path="_you_got_preempted.pt"):
        save_path = Path(save_path).resolve()
        self.temp_path = save_path.parent / temp_path
        self.save_path = save_path

    def save(self):
        to_save = {
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
            "torch_rng": torch.random.get_rng_state(),
            # note that this only gets it for 1 GPU
            "torch_cuda_rng": torch.cuda.get_rng_state(),
        }
        torch.save(to_save, self.temp_path)
        self.temp_path.replace(self.save_path)

    def load(self):
        if self.save_path.exists():
            saved_dictionary = torch.load(self.save_path)
            np_rng = saved_dictionary["np_rng"]
            py_rng = saved_dictionary["py_rng"]
            torch_rng = saved_dictionary["torch_rng"]
            torch_cuda_rng = saved_dictionary["torch_cuda_rng"]

            torch.random.set_rng_state(torch_rng)
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            torch.cuda.set_rng_state(torch_cuda_rng)
        else:
            warnings.warn(
                f"Tried to load RNG at {self.save_path} but it doesn't exist")
