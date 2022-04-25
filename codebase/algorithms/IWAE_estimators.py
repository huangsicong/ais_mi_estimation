import torch
from ..utils.vae_utils import log_mean_exp, log_normal_likelihood
from ..utils.experiment_utils import note_taking
from .variational_posterior import *
from tqdm import tqdm
import gc


def IWAE_loglikelihood_estimator(model, data_loader, mode, hparams, prior_dist,
                                 sampling_dist, prior_dist_obj):
    """ Estimating  marginal loglikelihood """

    across_batch_list = list()
    batch = data_loader[0][0]
    post_z = data_loader[0][1]
    batch = batch[2]

    with torch.no_grad():
        within_batch_list = list()
        note_taking(
            f"Running IWAE starting with {prior_dist} for K={hparams.mi.iwae} "
        )
        for j in tqdm(range(hparams.mi.iwae)):

            if j == 0:
                if mode == "upper":
                    z = post_z

                else:
                    z = sampling_dist.sample()
            else:
                z = sampling_dist.sample()

            x_mean, x_logvar = model.decode(z)
            x_logvar_full = torch.zeros_like(batch) + x_logvar
            conditional_likelihood = log_normal_likelihood(
                batch, x_mean, x_logvar_full)

            if prior_dist is None:
                weights = conditional_likelihood
            else:
                log_prior = prior_dist_obj.log_prob(z).sum(1)
                proposal = sampling_dist.log_prob(z).sum(1)
                weights = conditional_likelihood + log_prior - proposal
            within_batch_list.append(weights)
            if hparams.iwae_gc is None:
                hparams.iwae_gc = 1
            if j % hparams.iwae_gc == 0:
                torch.cuda.empty_cache()
                gc.collect()

        within_batch_matrix = torch.stack(within_batch_list, dim=1)

        if hparams.mi.iwae > 1000000:
            batch_mean = log_mean_exp(within_batch_matrix.cpu(),
                                      dim=1).to(hparams.device)
        else:
            batch_mean = log_mean_exp(within_batch_matrix, dim=1)

        across_batch_list.append(batch_mean)

    batch_vector = torch.cat(across_batch_list, dim=0)
    return batch_vector
