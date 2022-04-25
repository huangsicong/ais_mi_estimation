from ast import For
from copy import deepcopy
from torch.distributions import Normal
import numpy as np
import torch
from ..utils.vae_utils import log_normal_likelihood
from ..utils.experiment_utils import note_taking, save_recon
from ..registry import get_vp_model, get_hparams
from tqdm import tqdm
from torch import nn
from torch import optim
import wandb
from os.path import join as pjoin


def save_q(hparams, v_mean, std):
    q_path = pjoin(hparams.messenger.loadable_dir, "q.pt")
    torch.save((v_mean.cpu(), std.cpu()), q_path)


def save_encoder(hparams, model):
    encoder_path = pjoin(hparams.messenger.loadable_dir, "encoder.pt")
    dict_to_save = {
        "state_dict": model.state_dict(),
        "rng": torch.random.get_rng_state(),
    }
    torch.save(dict_to_save, encoder_path)


def load_encoder(hparams):
    encoder_hparam = get_hparams(hparams.load_encoder_hparam)
    encoder_dir = pjoin(encoder_hparam.output_root_dir, "result_loadable",
                        hparams.load_encoder_hparam)
    encoder_path = pjoin(encoder_dir, "encoder.pt")
    loaded_dict = torch.load(encoder_path)

    note_taking(f"loaded encoder from path {encoder_path}")
    return loaded_dict


def load_q(hparams):
    q_hparam = get_hparams(hparams.load_q_hparam)
    q_dir = pjoin(q_hparam.output_root_dir, "result_loadable",
                  hparams.load_q_hparam)

    q_path = pjoin(q_dir, "q.pt")
    # std_path = pjoin(hparams.q_hparam,"result_loadable", "std.pt")
    v_mean_cpu, std_cpu = torch.load(q_path)
    v_mean = v_mean_cpu.to(hparams.device)
    std = std_cpu.to(hparams.device)
    # std = torch.load(std_path)
    note_taking(f"loaded q from path {q_path}")
    return v_mean, std


def qz_x(hparams, temp_model, z, data):

    if "vae" in hparams.model_name:
        x_mean, x_logvar = temp_model.decode(z)
    else:
        x_mean, x_logvar = temp_model.dec.decode(z)
    std = torch.ones(x_mean.size()).to(hparams.device).mul(
        torch.exp(x_logvar * 0.5))
    x_normal_dist = Normal(loc=x_mean, scale=std)
    x = x_normal_dist.sample().to(device=hparams.device)
    if hparams.dataset.input_dims[0] != 1:
        x = x.view(-1, *hparams.dataset.input_dims)
        data = data.view(-1, *hparams.dataset.input_dims)

    z_mean, z_logvar = temp_model.enc(x)
    q_loss = -log_normal_likelihood(z, z_mean, z_logvar).mean()
    return q_loss


def compute_vp_reverse(model, data, hparams, z_exact=None):

    if "vae" in hparams.model_name:
        temp_decoder = deepcopy(model.dec)
    else:
        temp_decoder = deepcopy(model)
    temp_model = get_vp_model(hparams.mi.model_name,
                              hparams).to(hparams.device)

    temp_model.set_decoder(temp_decoder)

    temp_opt = optim.Adam(
        temp_model.enc.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay if hparams.weight_decay else 0)

    for i in range(hparams.mi.vp_num_iter):
        temp_opt.zero_grad()
        z = torch.randn([hparams.mi.batch_size, hparams.model_train.z_size
                         ]).requires_grad_().to(device=hparams.device)
        if "vae" in hparams.model_name:
            x_mean, x_logvar = temp_model.decode(z)
        else:
            x_mean, x_logvar = temp_model.dec.decode(z)
        std = torch.ones(x_mean.size()).to(hparams.device).mul(
            torch.exp(x_logvar * 0.5))
        x_normal_dist = Normal(loc=x_mean, scale=std)
        x = x_normal_dist.sample().to(device=hparams.device)
        if hparams.dataset.input_dims[0] != 1:
            x = x.view(-1, *hparams.dataset.input_dims)
            data = data.view(-1, *hparams.dataset.input_dims)

        z_mean, z_logvar = temp_model.enc(x)
        loss = -log_normal_likelihood(z, z_mean, z_logvar).mean()
        loss.backward()
        temp_opt.step()

    # This step is not necessary, but it is here for mainting the RNG states for reproducibility
    _, elbo, _, _, _, _, _ = temp_model(data, num_iwae=1, exact_kl=True)

    v_mean, v_logvar = temp_model.enc(data)

    note_taking(
        f"v_logvar.shape={v_logvar.shape}, v_logvar.sum(1).shape={v_logvar.sum(1).shape}"
    )
    wandb.log({"logvar_sum_avg": v_logvar.sum(1).mean()})

    std = torch.ones(v_mean.size()).to(hparams.device).mul(
        torch.exp(v_logvar * 0.5))
    v_normal_dist = Normal(loc=v_mean, scale=std)
    save_q(hparams, v_mean, std)
    return v_normal_dist


def compute_vp_reverse_forward(model, data, hparams, z_exact=None):

    if "vae" in hparams.model_name:
        temp_decoder = deepcopy(model.dec).to(hparams.device)
    else:
        temp_decoder = deepcopy(model).to(hparams.device)

    if hparams.dataset.input_dims[0] != 1:
        data = data.view(-1, *hparams.dataset.input_dims)

    temp_model = get_vp_model(hparams.mi.model_name,
                              hparams).to(hparams.device)
    temp_model.set_decoder(temp_decoder)

    # This step is not necessary, but it is here for mainting the RNG states for reproducibility
    _, elbo, _, _, _, _, _ = temp_model(data, num_iwae=1, exact_kl=True)

    temp_opt = optim.Adam(
        temp_model.enc.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay if hparams.weight_decay else 0)

    for i in tqdm(range(hparams.mi.vp_num_iter)):
        temp_opt.zero_grad()
        z = torch.randn([hparams.mi.batch_size, hparams.model_train.z_size
                         ]).requires_grad_().to(device=hparams.device)
        if "vae" in hparams.model_name:
            x_mean, x_logvar = temp_model.decode(z)
        else:
            x_mean, x_logvar = temp_model.dec.decode(z)
        std = torch.ones(x_mean.size()).to(hparams.device).mul(
            torch.exp(x_logvar * 0.5))
        x_normal_dist = Normal(loc=x_mean, scale=std)
        x = x_normal_dist.sample().to(device=hparams.device)
        if hparams.dataset.input_dims[0] != 1:
            x = x.view(-1, *hparams.dataset.input_dims)
            data = data.view(-1, *hparams.dataset.input_dims)

        z_mean, z_logvar = temp_model.enc(x)
        loss = -log_normal_likelihood(z, z_mean, z_logvar).mean()
        loss.backward()
        temp_opt.step()

    save_recon(data.view(-1, *hparams.dataset.input_dims),
               x_mean,
               hparams,
               epoch=666,
               best=False,
               overwrite_name=None)

    for i in range(hparams.mi.vp_num_iter):
        temp_opt.zero_grad()

        x_mean, elbo, _, _, _, _, _ = temp_model(data,
                                                 num_iwae=1,
                                                 exact_kl=True)
        loss = -elbo
        loss.backward()
        temp_opt.step()

    save_recon(data.view(-1, *hparams.dataset.input_dims),
               x_mean,
               hparams,
               epoch=667,
               best=False,
               overwrite_name=None)
    save_encoder(hparams, temp_model.enc)
    if hparams.dataset.input_dims[0] == 1:
        input_vector_length = hparams.dataset.input_vector_length
        data = data.view(-1, input_vector_length)
    v_mean, v_logvar = temp_model.enc(data)
    wandb.log({"logvar_sum_avg": v_logvar.sum(1).mean()})
    std = torch.ones(v_mean.size()).to(hparams.device).mul(
        torch.exp(v_logvar * 0.5))
    v_normal_dist = Normal(loc=v_mean, scale=std)
    save_q(hparams, v_mean, std)
    return v_normal_dist


def compute_vp_rf_w_loaded_encoder(model, data, hparams, z_exact=None):

    if "vae" in hparams.model_name:
        temp_decoder = deepcopy(model.dec).to(hparams.device)
    else:
        temp_decoder = deepcopy(model).to(hparams.device)

    if hparams.dataset.input_dims[0] != 1:
        data = data.view(-1, *hparams.dataset.input_dims)

    temp_model = get_vp_model(hparams.mi.model_name,
                              hparams).to(hparams.device)

    temp_model.set_decoder(temp_decoder)
    loaded_dict = load_encoder(hparams)
    torch.random.set_rng_state(loaded_dict["rng"])
    temp_model.enc.load_state_dict(loaded_dict["state_dict"])

    if hparams.dataset.input_dims[0] == 1:
        input_vector_length = hparams.dataset.input_vector_length
        data = data.view(-1, input_vector_length)
    v_mean, v_logvar = temp_model.enc(data)
    wandb.log({"logvar_sum_avg": v_logvar.sum(1).mean()})

    std = torch.ones(v_mean.size()).to(hparams.device).mul(
        torch.exp(v_logvar * 0.5))
    v_normal_dist = Normal(loc=v_mean, scale=std)
    save_q(hparams, v_mean, std)
    return v_normal_dist


def compute_vp_symmetric(model, data, hparams, z_exact=None):

    if "vae" in hparams.model_name:
        temp_decoder = deepcopy(model.dec).to(hparams.device)
    else:
        temp_decoder = deepcopy(model).to(hparams.device)

    if hparams.dataset.input_dims[0] != 1:
        data = data.view(-1, *hparams.dataset.input_dims)

    temp_model = get_vp_model(hparams.mi.model_name,
                              hparams).to(hparams.device)

    temp_model.set_decoder(temp_decoder)

    # This step is not necessary, but it is here for mainting the RNG states for reproducibility
    _, elbo, _, _, _, _, _ = temp_model(data, num_iwae=1, exact_kl=True)

    temp_opt = optim.Adam(
        temp_model.enc.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay if hparams.weight_decay else 0)

    for i in tqdm(range(hparams.mi.vp_num_iter)):
        temp_opt.zero_grad()
        z = torch.randn([hparams.mi.batch_size, hparams.model_train.z_size
                         ]).requires_grad_().to(device=hparams.device)
        if "vae" in hparams.model_name:
            x_mean, x_logvar = temp_model.decode(z)
        else:
            x_mean, x_logvar = temp_model.dec.decode(z)
        std = torch.ones(x_mean.size()).to(hparams.device).mul(
            torch.exp(x_logvar * 0.5))
        x_normal_dist = Normal(loc=x_mean, scale=std)
        x = x_normal_dist.sample().to(device=hparams.device)

        if hparams.dataset.input_dims[0] != 1:
            x = x.view(-1, *hparams.dataset.input_dims)
            data = data.view(-1, *hparams.dataset.input_dims)

        z_mean, z_logvar = temp_model.enc(x)
        reverse_loss = -log_normal_likelihood(z, z_mean, z_logvar).mean()
        x_mean, elbo, _, _, _, _, _ = temp_model(x, num_iwae=1, exact_kl=True)
        forward_loss = -elbo
        loss = 0.5 * forward_loss + 0.5 * reverse_loss
        loss.backward()
        temp_opt.step()

    save_recon(data.view(-1, *hparams.dataset.input_dims),
               x_mean,
               hparams,
               epoch=666,
               best=False,
               overwrite_name=None)

    save_encoder(hparams, temp_model.enc)
    if hparams.dataset.input_dims[0] == 1:
        input_vector_length = hparams.dataset.input_vector_length
        data = data.view(-1, input_vector_length)
    v_mean, v_logvar = temp_model.enc(data)
    wandb.log({"logvar_sum_avg": v_logvar.sum(1).mean()})
    std = torch.ones(v_mean.size()).to(hparams.device).mul(
        torch.exp(v_logvar * 0.5))
    v_normal_dist = Normal(loc=v_mean, scale=std)

    save_q(hparams, v_mean, std)
    return v_normal_dist
