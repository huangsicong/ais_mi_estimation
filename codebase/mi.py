#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from datetime import datetime
import subprocess
import time
import torch
from .registry import get_hparams, get_G
from .data.load_data import *
from .utils.experiment_utils import *
from .algorithms.analytical_linear_vae import MI_Analytical
from .algorithms.mi_ais import MIAISEstimator
import argparse
import random
import sys
from .utils.gan_utils import gan_bridge
from .utils.load_utils import load_checkpoint as load_checkpoint_ncgan

from os.path import join as pjoin

sys.stdout.flush()
parser = argparse.ArgumentParser()
parser.add_argument("--hparam_set", default=None, type=str)
parser.add_argument("--hparam_from_pickle", default=None, type=str)
parser.add_argument("--overwrite", default=False, type=bool)
parser.add_argument("--jobid", default=None, type=str)
args = parser.parse_args()
args_dict = vars(args)
if args.hparam_set:
    hparams = get_hparams(args.hparam_set)
if args.hparam_from_pickle:
    hparams = get_hparams_from_pickle(args.hparam_from_pickle)
    print(f'hparams {hparams}')

hparams.overwrite = args.overwrite


def main(hparams):

    set_random_seed(hparams.random_seed)

    if hparams.original_experiment:
        if "vae" in hparams.model_name:
            note_taking("About to run experiment on {} with z size={}".format(
                hparams.model_name, hparams.model_train.z_size))

            model = get_model(hparams).to(hparams.device)
            print(str(model))
            if hparams.specific_model_path:
                load_checkpoint_old(
                    path=hparams.checkpoint_path
                    if hparams.train_first else hparams.checkpoint_path,
                    optimizer=None,
                    reset_optimizer=False,
                    model=model)
            else:
                load_checkpoint(path=hparams.checkpoint_path,
                                optimizer=None,
                                reset_optimizer=False,
                                model=model)

        elif "gan" in hparams.model_name:
            if "cifar" in hparams.model_name:
                model = get_G(hparams).to(hparams.device)
                loaded_dict = load_checkpoint_ncgan(hparams.checkpoint_path)
                if "_s_" not in args.hparam_set:
                    note_taking(
                        "Resetting RNG state to the loaded checkpoint RNG state"
                    )
                    torch.random.set_rng_state(loaded_dict["rng"])
                else:
                    note_taking("Not setting RNG state.")
                model.load_state_dict(loaded_dict["G_weights"])
                epoch = loaded_dict["epoch"]
                model.eval()
                note_taking(
                    f"loaded eval() NCGAN with z size={hparams.model_train.z_size}, at epoch {epoch}"
                )

            else:
                model = gan_bridge(hparams)
                note_taking(
                    "About to run GAN experiment on {} with z size={}".format(
                        hparams.model_name, hparams.model_train.z_size))
        elif "aae" in hparams.model_name:
            model = aae_bridge(hparams)
            note_taking(
                "About to run AAE experiment on {} with z size={}".format(
                    hparams.model_name, hparams.model_train.z_size))

    data_loader = load_simulate_data(model,
                                     hparams,
                                     hparams.mi.batch_size,
                                     hparams.mi.n_batch,
                                     beta=1.)

    hparams.messenger.beta = 1.
    MI_estimator = MIAISEstimator(data_loader, model, hparams)
    BDMC_results = MI_estimator.compute_ais_mi()
    num_data = hparams.mi.n_batch * hparams.mi.batch_size

    end_time = datetime.now()
    hour_summery = compute_duration(hparams.messenger.start_time, end_time)
    note_taking(
        f"BDMC on {hparams.dataset.name} data finished. Took {hour_summery} hours"
    )
    note_taking(
        f'AIS steps: {hparams.rd.anneal_steps}, number of data points: {num_data}'
        .center(80))
    if hparams.analytical_MI:
        analytical_runner = MI_Analytical(data_loader, model, hparams)
        Analytical_results = analytical_runner.run()

    note_taking('H(x)'.center(80))
    if hparams.rd is not None:
        note_taking('BDMC H(x) lower bound: {}'.format(
            BDMC_results['lower_H']).center(80))
        note_taking('BDMC H(x) upper bound: {}'.format(
            BDMC_results['upper_H']).center(80))
        note_taking('BDMC gap: {}'.format(BDMC_results['gap_np']).center(80))
    if hparams.analytical_MI:
        note_taking('analytical H(x)(R+D)(one batch): {}'.format(
            Analytical_results['H_batch']).center(80))
        note_taking('analytical H(x)(direct): {} \n'.format(
            Analytical_results['analytical_H_direct']).center(80))
    note_taking(' '.center(80))

    note_taking('H(x|z)'.center(80))
    if hparams.rd is not None:
        note_taking('samples H(x|z)(one batch): {}'.format(
            BDMC_results['cond_H']).center(80))
    if hparams.analytical_MI:
        note_taking('analytical H(x|z)(D)(one batch): {}'.format(
            Analytical_results['D_batch']).center(80))

        note_taking('analytical H(x|z)(direct): {} \n'.format(
            Analytical_results['analytical_cH_direct']).center(80))
    note_taking(' '.center(80))

    note_taking('I(x;z)'.center(80))
    if hparams.rd is not None:
        note_taking('BDMC I(x;z) upper bound: {}. CI: [{},{}]'.format(
            BDMC_results['u_MI'], BDMC_results['u_MI_m'],
            BDMC_results['u_MI_p']).center(80))
        note_taking('BDMC I(x;z) lower bound: {}. CI: [{},{}]'.format(
            BDMC_results['l_MI'], BDMC_results['l_MI_m'],
            BDMC_results['l_MI_p']).center(80))

    if hparams.analytical_MI:
        note_taking('analytical I(x;z)(R)(one batch): {}'.format(
            Analytical_results['R_batch']).center(80))
        note_taking('analytical I(x;z) (direct): {} \n'.format(
            Analytical_results['analytical_MI_direct']).center(80))
    note_taking(' '.center(80))

    note_taking('Gaps'.center(80))
    if hparams.analytical_MI and hparams.rd is not None:
        note_taking('H(x) upper gap'.center(80))
        note_taking('(H(x) upper - analytical H(x)) {}'.format(
            BDMC_results['upper_H'] - Analytical_results['H']).center(80))

        note_taking('H(x) lower gap'.center(80))
        note_taking('(analytical H(x) - H(x) lower) : {}'.format(
            Analytical_results['H'] - BDMC_results['lower_H']).center(80))
        note_taking('H(x|z) gap'.center(80))
        note_taking('(sample H(x|z) - analytical H(x|z) ): {}'.format(
            BDMC_results['cond_H'] - Analytical_results['D']).center(80))

    if hparams.use_wandb:
        wandb.log(BDMC_results)
        if hparams.analytical_MI:
            wandb.log(Analytical_results)

    logging(args_dict,
            hparams.to_dict(),
            hparams.messenger.results_dir,
            hparams.messenger.dir_path,
            stage="final")
    return BDMC_results


if __name__ == '__main__':

    initialize_run(hparams, args)
    start_time = datetime.now()
    hparams.messenger.start_time = start_time
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        git_label = subprocess.check_output(
            ["cd " + dir_path + " && git describe --always && cd .."],
            shell=True).strip()
        if hparams.verbose:
            note_taking("The git label is {}".format(git_label))
    except:
        note_taking(
            "WARNING! Encountered unknwon error recording git label...")
    results = main(hparams)
    end_time = datetime.now()
    hour_summery = compute_duration(start_time, end_time)
    summery_npy_path = pjoin(hparams.messenger.arxiv_dir,
                             f"{hparams.dataset.name}_BDMC.npz")
    results['hours'] = hour_summery
    np.savez(summery_npy_path, results)
    note_taking(
        "Experiment finished, results written at: {}. Took {} hours".format(
            hparams.messenger.results_dir, hour_summery))
    sys.exit(0)
