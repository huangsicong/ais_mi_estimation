""" 
This is the root script to train generative models
"""
import argparse
from .registry import get_hparams
from .utils.experiment_utils import (logging, note_taking, set_random_seed,
                                     initialize_run, get_hparams_from_pickle)
from .algorithms.trainers import VAETrainer
from .algorithms.train_gan import GANTrainer


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparam_set", default=None, type=str)
    parser.add_argument("--hparam_from_pickle", default=None, type=str)
    parser.add_argument("--overwrite", default=False, type=bool)
    parser.add_argument("--jobid", default=None, type=str)
    args = parser.parse_args()
    return args


def prepare_hparams(args):

    if args.hparam_set:
        hparams = get_hparams(args.hparam_set)
    if args.hparam_from_pickle:
        hparams = get_hparams_from_pickle(args.hparam_from_pickle)

    initialize_run(hparams, args)
    hparams.overwrite = args.overwrite
    return hparams


def main(hparams):
    if "vae" in hparams.model_name:
        trainer = VAETrainer(hparams)
    elif "gan" in hparams.model_name:
        trainer = GANTrainer(hparams)
    else:
        note_taking("Model doesn't have an appropriate trainer")

    note_taking("About to run experiment on {} with z size={}".format(
        hparams.model_name, hparams.model_train.z_size))

    trainer.run()


if __name__ == '__main__':
    args = prepare_args()
    args_dict = vars(args)
    hparams = prepare_hparams(args)

    set_random_seed(hparams.random_seed)

    main(hparams)

    logging(args_dict,
            hparams.to_dict(),
            hparams.messenger.results_dir,
            hparams.messenger.dir_path,
            stage="final")
