import os
import torch
from tqdm import tqdm
import wandb
from ..utils.experiment_utils import note_taking, get_checkpoint_path
from ..utils.load_utils import load_checkpoint
from ..utils.save_utils import save_checkpoint
from ..utils.computation_utils import bits_per_dimension
from ..data.loaderbase import LoaderBase
from ..utils.save_utils import save_input_batch
from ..registry import get_augment


class Trainerbase:
    """Abstract class for the various trainers in the codebase

    This serves to be class to be inherited by other algorithms,
    it will work out of the box as long as run_batch is implemented
    Note that to the child can sample reconstructions in run_batch

    Arguments:
        model: a PyTorch model that is callable
        optimizer: a PyTorch optimizer
        hparams: an instance of the Hparam object
    """

    def __init__(self,
                 model,
                 optimizer,
                 hparams,
                 to_print=True,
                 save_first_batch=True):
        self.epoch, self.validation_loss_list = 0, []
        self.optimizer = optimizer
        self.hparams = hparams
        self.model = model
        self.loader = LoaderBase(
            hparams, [get_augment(hparams.dataset.augmentation.type, hparams)]
            if hparams.dataset.augmentation is not None else None)
        self.use_wandb = hparams.use_wandb

        if not self.use_wandb and to_print:
            note_taking("Model Architecture")
            note_taking(str(model))

        if save_first_batch:
            save_input_batch(hparams)

        if self.use_wandb:
            wandb.watch(model)

    def get_model(self):
        """Returns the model that the trainer is currently managing
        """
        return self.model

    def sample_images(self, best):
        """For children to optionally implement to sample images from the model

        Arguments:
            best (boolean): True if the model is currently at the lowest
                validation loss, False otherwise
        Returns None
        """
        return None

    def run_batch(self, data, istrain, to_save=False):
        """For children to implement, runs training on a batch of samples

        Arguments:
            data (Tensor): a PyTorch tensor
            istrain (boolean): indicates if it's training or not training(test/val)
            to_save (boolean): to save the reconstructed image or not
        """
        raise NotImplementedError

    def save(self, checkpoint_path):
        """Save the current model, optimizer, epoch and
            validation_loss_list

        Arguments:
            checkpoint_path (str): place to save the checkpoint

        NOTE:
            If you write your own save function, you also must
                write your own load function!!!

            **When overriding, you are only allowed to
                extend the dictionary but should provide
                these keys still**
        """
        dict_to_save = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_epoch": self.epoch,
            "validation_loss_list": self.validation_loss_list,
            "rng": torch.random.get_rng_state(),
        }
        save_checkpoint(checkpoint_path, dict_to_save)

    def load(self, checkpoint_path, load_optimizer=True):
        """Loads the model and optimizer while setting self.epoch and
            self.validation_loss_list to the checkpoint

        Arguments:
            checkpoint_path (str): place to load the checkpoint
            optimizer: the PyTorch optimizer to mutate, can be None
        """
        loaded_dict = load_checkpoint(checkpoint_path)

        torch.random.set_rng_state(loaded_dict["rng"])

        self.model.load_state_dict(loaded_dict["state_dict"])

        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict["optimizer"])

        self.epoch = loaded_dict["global_epoch"]
        self.validation_loss_list = loaded_dict["validation_loss_list"]

        note_taking(
            "Model loaded in trainer from {} at epoch {} with validation_loss {}"
            .format(checkpoint_path, self.epoch,
                    self.validation_loss_list[-1]))

    def run(self):
        """
        Entry point for all the trainers, call this function to start the run
            it will load the checkpoint first if path is specified
        """
        hparams = self.hparams
        if hparams.checkpoint_path:
            load_optimizer = not (hparams.freeze_encoder
                                  or hparams.freeze_decoder)
            self.load(hparams.checkpoint_path, load_optimizer=load_optimizer)
        self.train_and_val()
        return self.model

    def done_training(self):
        """A call back for when training is done to save extra info
        """
        pass

    def get_save_path(self, name):
        checkpoint_path = os.path.join(self.hparams.messenger.checkpoint_dir,
                                       name)
        return checkpoint_path

    def run_epoch(self, loader, istrain=True, term=None, to_save=False):
        """Runs one full pass on the given dataloader

        Arguments:
            epoch (int): which epoch it's currently going to run
            loader: a PyTorch dataloader
            istrain (boolean): indicates if it's running training or
                not training(test/val)
            term (int or None): how many batches to run before terminating
            to_save (boolean): to save or not
        """
        running_loss, count = 0.0, 0
        for i, (data, _) in enumerate(loader):

            if term == count:
                break
            if self.hparams.clamp_0_1 is True:
                data.clamp_(0, 1)
            data = data.to(device=self.hparams.device,
                           dtype=self.hparams.tensor_type)
            save_bool = (i == 0 and to_save)  #saves on the first batch
            loss_cpu = self.run_batch(data, istrain, to_save=save_bool)
            running_loss += loss_cpu
            count += 1
        running_loss = running_loss / count if count else 0
        return running_loss

    def post_epoch_hook(self):
        """Override this function to do things after a training epoch is done
        """
        pass

    def get_train_and_val_loader(self):
        train_loader, validation_loader = self.loader.get_train_and_val(
            shuffle_train=True, shuffle_val=False)
        return train_loader, validation_loader

    def train_and_val(self):
        """
        Trains starting from scratch or a checkpoint and runs
            validation in between
        """
        hparams = self.hparams
        num_epochs = hparams.model_train.epochs
        train_loader, validation_loader = self.get_train_and_val_loader()
        start = self.epoch + 1
        for epoch in tqdm(range(start, num_epochs + 1)):
            self.epoch = epoch
            train_loss = self.run_epoch(train_loader, istrain=True)
            self.post_epoch_hook()
            note_taking(f'Training Epoch {epoch} Train loss is {train_loss}')
            to_save = (epoch % hparams.train_print_freq) == 0
            save_ckpt = (epoch % hparams.checkpointing_freq) == 0
            validation_loss = self.run_epoch(validation_loader,
                                             istrain=False,
                                             term=hparams.n_val_batch,
                                             to_save=to_save)
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss
                })

            note_taking(
                f'Training Epoch {epoch} Validation loss is {validation_loss}')
            best = False
            if not self.validation_loss_list or validation_loss < min(
                    self.validation_loss_list):
                best = True
                note_taking(
                    f'new min val loss is {validation_loss} at epoch {epoch}')
            if to_save:
                self.sample_images(best=False)
            self.validation_loss_list.append(validation_loss)
            if best:
                self.sample_images(best=best)
                self.save(self.get_save_path('best.pth'))

            if save_ckpt or (epoch == num_epochs):
                self.save(self.get_save_path(f'checkpoint_epoch{epoch}.pth'))
        self.sample_images(best=False)  # sample when done
        self.done_training()
