from .experiment_utils import note_taking
from torchvision.utils import save_image
import torch

import numpy as np
# import scipy.stats
from scipy import stats

from datetime import datetime


def sample_images(hparams,
                  model,
                  epoch,
                  best=False,
                  prior_dist=None,
                  name=None,
                  sample_z=None):
    if sample_z is None:
        sample = torch.randn(64, hparams.model_train.z_size).to(
            device=hparams.device, )
    else:
        sample = sample_z
    sample, x_logvar = model.decode(sample)
    sample = sample.cpu()
    save_name = hparams.messenger.readable_dir + (
        '/best_' if best else
        '') + 'sample' + (str(epoch) if epoch is not None else '') + (
            ("_" + name) if name is not None else '')
    image_path = save_name + '.png'

    if hparams.dataset.data_name == "cifar10":
        sample = sample / 2 + 0.5
        note_taking("...cifar generation re-normalized and saved. ")
    save_image(
        sample.view(64, hparams.dataset.input_dims[0],
                    hparams.dataset.input_dims[1],
                    hparams.dataset.input_dims[2]), image_path)
    if x_logvar is not None:
        note_taking(
            "Image sampled from the {}checkpoint has the decoder variance: {}".
            format(("best " if best else ''),
                   torch.exp(x_logvar).detach().cpu().numpy().item()))
    else:
        note_taking("Image sampled from the {}checkpoint".format(
            ("best " if best else '')))


def init_rd_ais_results(data):
    rate_list = list()
    rd_const_lower_list = list()
    rd_const_dict = {"lower": rd_const_lower_list}

    if data == "simulate":
        rd_const_upper_list = list()
        rd_const_mean_list = list()
        rd_const_dict.update({"mean": rd_const_mean_list})
        rd_const_dict.update({"upper": rd_const_upper_list})

    distortion_list = list()
    L2_D_list = list()
    results = {
        "rate_list": rate_list,
        "rd_const_dict": rd_const_dict,
        "distortion_list": distortion_list,
        "L2_D_list": L2_D_list
    }
    return results


def check_batch_size(batch_size, n_batch):
    assert batch_size % n_batch == 0, f"batch_size={batch_size} should be devisible by number of batch for CI ({n_batch})"


def mean_confidence_interval(data, confidence=0.95):
    """ https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data """

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    note_taking(f"se={se}, h={h}")
    return m, m - h, m + h


def Guassian_test(gpu_tensor, dim=0):
    z_variance = torch.var(gpu_tensor, dim=dim)
    z_mean = torch.mean(gpu_tensor, dim=dim)
    z_numpy = gpu_tensor.detach().cpu().numpy()
    _, p = stats.normaltest(z_numpy, axis=dim)
    note_taking("Across {} entries, mean={}, var={}, p value={}".format(
        gpu_tensor.size(dim), z_mean, z_variance, p))


def compute_CI(data, n_batch):
    """ 
    data: [B]
    
    """

    total = data.shape[0]
    batch_size = total // n_batch
    assert n_batch * batch_size == total, "Please make sure the number of total data points is an integer multiple of the batch size for CI."
    start = 0
    end = 0 + batch_size
    indep_runs = list()
    for i in range(n_batch):
        indep_runs.append(data[start:end].mean())
        start += batch_size
        end += batch_size
    mean, minus, plus = mean_confidence_interval(indep_runs)
    return mean, plus, minus