import wandb
from .trainerbase import Trainerbase
from ..registry import get_model
from ..utils.experiment_utils import (latent_image_sample, save_recon,
                                      EvaluateModel, note_taking, get_recon)
from ..utils.vae_utils import get_vae_optimizer


class VAETrainer(Trainerbase):
    """
    A VAE trainer that inherits from the trainerbase,
        it over writes the run_batch function with the ELBO loss
    """

    def __init__(self, hparams):
        model = get_model(hparams).to(hparams.device)
        optimizer = get_vae_optimizer(hparams, model)
        super().__init__(model, optimizer, hparams)
        self.exact_kl = (self.hparams.exact_kl
                         if self.hparams.exact_kl else False)

    def sample_images(self, best):
        """Sample images using the latent space of the model

        Arguments:
            best (boolean): True if the model is currently at the lowest
                validation loss, False otherwise
        Returns None
        """
        latent_image_sample(self.hparams,
                            self.model,
                            self.epoch,
                            best,
                            use_wandb=self.hparams.use_wandb)

    def run_batch(self, data, istrain, to_save=False):
        """
        Runs a batch of training or test data and plot the reconstruction
            in between
        """
        if istrain:
            self.optimizer.zero_grad()
            recon_batch, elbo, _, _, _, _, _ = self.model(
                data, exact_kl=self.exact_kl)
            loss = -elbo
            loss.backward()
            self.optimizer.step()
        else:
            with EvaluateModel(self.model):
                recon_batch, elbo, _, _, _, _, _ = self.model(
                    data, exact_kl=self.exact_kl)
                loss = -elbo
        if to_save:
            recon_batch = recon_batch.detach()
            comparison = get_recon(data, recon_batch, self.hparams)
            if self.use_wandb:
                wandb.log({f"epoch_{self.epoch}": wandb.Image(comparison)})

            save_recon(data, recon_batch, self.hparams, self.epoch, False)

        return loss.item()
