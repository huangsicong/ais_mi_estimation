#!/usr/bin/env python3

from .ais import AIS_core, run_ais_chain_simplified
from ..utils.vae_utils import log_normal_likelihood
import numpy as np
import torch
from torch.distributions import Normal
from ..utils.anneal_schedules import get_schedule
from ..utils.experiment_utils import note_taking, init_dir
from ..utils.computation_utils import unit_prior
from ..utils.mi_utils import compute_CI
from ..registry import get_hparams
from .IWAE_estimators import IWAE_loglikelihood_estimator
from os.path import join as pjoin
from .variational_posterior import *
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def conditional_H(hparams, model, simulated_data, n_batch):

    cond_H_list = list()
    for i in range(n_batch):
        (x_bundle, z) = simulated_data[i]
        (x_mean, x_logvar, x) = x_bundle
        x_logvar_full = torch.zeros_like(x) + x_logvar
        cond_H = -log_normal_likelihood(x, x_mean,
                                        x_logvar_full).detach().cpu()
        cond_H_list.append(cond_H)

    cond_H = torch.cat(cond_H_list).cpu()
    return cond_H


def save_per_chain_csv(logw, hparams, mode):

    pt_path = pjoin(hparams.messenger.loadable_dir,
                    f"chain_logw_{mode}" + ".pt")

    torch.save(logw, pt_path)

    header_csv = f''
    for i in range(hparams.rd.batch_size):
        header_csv += f'image {i+1},'

    csv_array = logw
    save_path = pjoin(hparams.messenger.arxiv_dir, mode + "_chains_logw.csv")
    np.savetxt(save_path,
               csv_array,
               fmt='%.4f',
               delimiter=",",
               newline="\n",
               header=header_csv,
               comments='')


def save_logw_csv(logw, hparams, mode):

    pt_path = pjoin(hparams.messenger.loadable_dir, f"avelogw_{mode}" + ".pt")

    torch.save(logw, pt_path)
    header_csv = f''
    for i in range(hparams.rd.batch_size):
        header_csv += f'image {i+1},'

    csv_array = logw
    save_path = pjoin(hparams.messenger.arxiv_dir, mode + "_avg_logw.csv")
    np.savetxt(save_path,
               csv_array,
               fmt='%.4f',
               delimiter=",",
               newline="\n",
               header=header_csv,
               comments='')


def dict_to_numpy(pt_dict):
    for key, value in pt_dict.items():
        try:
            pt_dict[key] = value.cpu().numpy()
        except:
            pass


class MIAISEstimator:

    def __init__(self, data_loader, model, hparams, to_print=False):

        self.hparams = hparams
        self.model = model
        self.data_loader = data_loader
        self.results = {}
        self.t_schedule = get_schedule(hparams.rd)

        if hparams.IS_debug:
            note_taking(f'forward self.t_schedule={self.t_schedule}')

        if hparams.step_sizes_target is None:
            step_sizes_dir = pjoin(hparams.messenger.arxiv_dir, "step_sizes")
            init_dir(step_sizes_dir)
        else:
            target_hparams = get_hparams(hparams.step_sizes_target)
            step_sizes_dir = pjoin(target_hparams.output_root_dir,
                                   "results_out")
            if target_hparams.group_list:
                for subgroup in target_hparams.group_list:
                    step_sizes_dir = pjoin(step_sizes_dir, subgroup)

            step_sizes_dir = pjoin(step_sizes_dir, hparams.step_sizes_target)
            step_sizes_dir = pjoin(step_sizes_dir, "result_arxiv/step_sizes")
            note_taking("Will load step sizes from:{}".format(step_sizes_dir))
            # check random seed.
            # The random see should be different from the run where the step sizes are being loaded
            if target_hparams.random_seed == hparams.random_seed:
                note_taking(
                    "WARNING!\n Random seed({}) is the same with the experiment with the target step sizes!!"
                    .format(hparams.random_seed))
        self.step_sizes_dir = step_sizes_dir
        hparams.rd.num_total_chains = hparams.rd.batch_size * hparams.rd.n_chains

    def ais_entropy(self,
                    model,
                    t_schedule,
                    data_loader,
                    hparams,
                    prior_dist=None):
        self.step_sizes_folder = self.step_sizes_dir
        step_sizes_path = self.step_sizes_folder + "forward_"
        init_dir(self.step_sizes_folder)
        anneal_ops = AIS_core(hparams.rd.target_dist, prior_dist, model,
                              hparams.rd, hparams, None)

        # After log_mean_exp, logw is for per data point.
        forward_logws, _, _, logw_chains_forward, _, _, _, logw_tvo_forward, logw_tvo_integrated_forward = run_ais_chain_simplified(
            model,
            data_loader,
            mode='forward',
            schedule=t_schedule,
            hparams=hparams,
            task_params=hparams.rd,
            start_state=None,
            init_weights=None,
            step_sizes_dir=step_sizes_path,
            prior_dist=prior_dist,
            rep_model=None,
            anneal_ops=anneal_ops)

        # run backward chain
        step_sizes_path = self.step_sizes_folder + "backward_"
        backward_schedule = np.flip(t_schedule, axis=0)
        if hparams.IS_debug:
            note_taking(f'backward backward_schedule={backward_schedule}')

        backward_logws, logw_chains_backward, logw_tvo_backward, logw_tvo_integrated_backward = run_ais_chain_simplified(
            model,
            data_loader,
            mode='backward',
            schedule=backward_schedule,
            hparams=hparams,
            task_params=hparams.rd,
            start_state=None,
            init_weights=None,
            step_sizes_dir=step_sizes_path,
            prior_dist=prior_dist,
            rep_model=None,
            anneal_ops=anneal_ops)

        return backward_logws.cpu(), forward_logws.cpu(
        ), logw_chains_forward.cpu(), logw_chains_backward.cpu()

    def iwae_entropy(self,
                     model,
                     data_loader,
                     hparams,
                     prior_dist=None,
                     n_batch=1):
        prior_dist_obj = unit_prior(hparams.mi.batch_size,
                                    hparams.model_train.z_size, hparams)

        if prior_dist == "variational":
            batch = data_loader[0][0]
            z_exact = data_loader[0][1]
            batch = batch[2]
            sampling_dist = compute_vp_reverse(model, batch, hparams, z_exact)

        elif prior_dist == "variational_reverse_forward":
            batch = data_loader[0][0]
            batch = batch[2]
            z_exact = data_loader[0][1]
            sampling_dist = compute_vp_reverse_forward(model, batch, hparams,
                                                       z_exact)

        elif prior_dist == "compute_vp_symmetric":
            batch = data_loader[0][0]
            batch = batch[2]
            z_exact = data_loader[0][1]
            sampling_dist = compute_vp_symmetric(model, batch, hparams,
                                                 z_exact)

        elif prior_dist == "loadq":
            v_mean, std = load_q(hparams)
            sampling_dist = Normal(loc=v_mean, scale=std)
        elif prior_dist == "load_encoder":
            batch = data_loader[0][0]
            batch = batch[2]
            z_exact = data_loader[0][1]
            sampling_dist = compute_vp_rf_w_loaded_encoder(
                model, batch, hparams, z_exact)
        else:
            sampling_dist = prior_dist_obj

        with torch.no_grad():
            lower_bounds = IWAE_loglikelihood_estimator(
                model, data_loader, "lower", hparams, prior_dist,
                sampling_dist, prior_dist_obj)
            upper_bounds = IWAE_loglikelihood_estimator(
                model, data_loader, "upper", hparams, prior_dist,
                sampling_dist, prior_dist_obj)

        return upper_bounds.cpu(), lower_bounds.cpu()

    def compute_ais_mi(self):
        if self.hparams.mi.iwae:

            upper_bounds, lower_bounds = self.iwae_entropy(
                self.model, self.data_loader, self.hparams,
                self.hparams.rd.prior_dist, 1)
        else:
            upper_bounds, lower_bounds, logw_chains_forward, logw_chains_backward = self.ais_entropy(
                self.model, self.t_schedule, self.data_loader, self.hparams,
                self.hparams.rd.prior_dist)

        cond_H_1batch = conditional_H(self.hparams, self.model,
                                      self.data_loader, 1)
        u_MI = -lower_bounds - cond_H_1batch
        l_MI = -upper_bounds - cond_H_1batch
        u_MI_avg, u_MI_p, u_MI_m = compute_CI(u_MI, self.hparams.mi.CI_n_batch)
        l_MI_avg, l_MI_p, l_MI_m = compute_CI(l_MI, self.hparams.mi.CI_n_batch)
        upper_H = -lower_bounds.mean()
        lower_H = -upper_bounds.mean()
        gap_np = u_MI_avg - l_MI_avg

        self.results['u_MI'] = u_MI_avg
        self.results['u_MI_p'] = u_MI_p
        self.results['u_MI_m'] = u_MI_m
        self.results['l_MI'] = l_MI_avg
        self.results['l_MI_p'] = l_MI_p
        self.results['l_MI_m'] = l_MI_m
        # NOTE only consider one batch
        self.results['cond_H'] = cond_H_1batch.mean()
        self.results['upper_H'] = upper_H
        self.results['lower_H'] = lower_H
        self.results['gap_np'] = gap_np

        dict_to_numpy(self.results)

        return self.results
