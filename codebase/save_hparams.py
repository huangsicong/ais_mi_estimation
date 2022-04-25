import argparse
from .registry import get_hparams
from .utils.experiment_utils import (init_dir)
import pickle
from codebase.hparams.hparam import Hparam as hp

from os.path import join as pjoin


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam_set", default="default", type=str)
    parser.add_argument("--overwrite", default=False, type=bool)
    parser.add_argument("--jobid", default=None, type=str)
    args = parser.parse_args()
    return args


def main(args):
    hparams = get_hparams(args.hparam_set)
    hparams.overwrite = args.overwrite

    init_dir(hparams.save_dir)
    for (i, hparams_to_save_name) in enumerate(hparams.load_list):
        hparams_to_save = get_hparams(hparams_to_save_name)
        hparams_to_save.hparam_set = hparams_to_save_name
        hparams_to_save.wandb = hp()
        hparams_to_save.wandb.tags = hparams.wandb_construct.tags
        hparams_to_save.wandb.name = hparams.wandb_construct.name_list[i]
        hparams_to_save.wandb.job_type = hparams.wandb_construct.job_type
        hparams_to_save.wandb.group = hparams.wandb_construct.group
        hparams_to_save.wandb.project = hparams.wandb_construct.project
        save_path = pjoin(hparams.save_dir, hparams_to_save_name + ".p")
        with open(save_path, 'wb') as f:
            pickle.dump(hparams_to_save, f)


if __name__ == '__main__':
    args = prepare_args()
    main(args)
