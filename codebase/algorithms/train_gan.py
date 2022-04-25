import torch
from torch import optim
from torch.optim import lr_scheduler
import wandb
from .trainerbase import Trainerbase
from ..registry import get_G, get_D
from ..utils.experiment_utils import (EvaluateModel, note_taking)
from ..utils.load_utils import load_checkpoint
from ..utils.save_utils import save_checkpoint
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from ..data.loaderbase import LoaderBase
from ..registry import get_augment
from ..utils.save_utils import save_input_batch
from torchvision.utils import save_image
import os


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GANTrainer:
    """
    # A GAN trainer that starts from scrath. 
    # Some code are modified from  https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    def __init__(self, hparams):
        self.input_dims = hparams.dataset.input_dims
        # Create the generator
        self.netG = get_G(hparams).to(hparams.device)
        self.netD = get_D(hparams).to(hparams.device)
        self.netD.apply(weights_init)

        # # Print the model
        # print(netD)
        self.netG.apply(weights_init)

        # # Print the mode
        # print(netG)
        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64,
                                       hparams.model_train.z_size,
                                       1,
                                       1,
                                       device=hparams.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(),
                                     lr=hparams.learning_rate,
                                     betas=(hparams.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(),
                                     lr=hparams.learning_rate,
                                     betas=(hparams.beta1, 0.999))

        self.epoch, self.validation_loss_list = 0, []
        self.hparams = hparams
        self.loader = LoaderBase(
            hparams, [get_augment(hparams.dataset.augmentation.type, hparams)]
            if hparams.dataset.augmentation is not None else None)
        self.use_wandb = hparams.use_wandb

        save_input_batch(hparams)

    def sample_images(self, best):
        self.latent_image_sample(self.hparams,
                                 self.epoch,
                                 best,
                                 sample_z=self.fixed_noise,
                                 use_wandb=self.use_wandb)

    def latent_image_sample(self,
                            hparams,
                            epoch,
                            best=False,
                            name=None,
                            sample_z=None,
                            use_wandb=False):
        model = self.netG
        with EvaluateModel(model):
            sample = model(self.fixed_noise).detach().cpu()
        sample = sample.cpu()
        num_samples = sample.size(0)
        if best:
            toadd = "best_sample"
        else:
            toadd = f"epoch{epoch}_sample"
        toadd += f"{name if name else ''}"
        save_name = os.path.join(hparams.messenger.readable_dir, toadd)

        image_path = save_name + '.png'

        if hparams.tanh_output:
            sample = sample / 2 + 0.5

        save_image(
            sample.view(num_samples, hparams.dataset.input_dims[0],
                        hparams.dataset.input_dims[1],
                        hparams.dataset.input_dims[2]), image_path)

        note_taking("Image sampled from the {}checkpoint".format(
            ("best " if best else '')))

        if use_wandb:
            wandb.log({toadd: wandb.Image(sample)})

    def get_save_path(self, name):
        checkpoint_path = os.path.join(self.hparams.messenger.checkpoint_dir,
                                       name)
        return checkpoint_path

    def get_train_and_val_loader(self):
        train_loader, validation_loader = self.loader.get_train_and_val(
            shuffle_train=True, shuffle_val=False)
        return train_loader, validation_loader

    # def post_epoch_hook(self):
    #     self.scheduler.step()

    def load(self, checkpoint_path, load_optimizer=True):
        loaded_dict = load_checkpoint(checkpoint_path)

        torch.random.set_rng_state(loaded_dict["rng"])

        self.netD.load_state_dict(loaded_dict["D_weights"])
        self.netG.load_state_dict(loaded_dict["G_weights"])

        if load_optimizer:
            self.optimizerD.load_state_dict(loaded_dict["D_optimizer"])
            self.optimizerG.load_state_dict(loaded_dict["G_optimizer"])
        # self.scheduler.load_state_dict(loaded_dict["scheduler"])

        self.epoch = loaded_dict["epoch"]
        self.validation_loss_list = loaded_dict["validation_loss_list"]

        note_taking(
            "Model loaded in trainer from {} at epoch {} with validation_loss {}"
            .format(checkpoint_path, self.epoch,
                    self.validation_loss_list[-1]))

    def save(self, checkpoint_path):
        dict_to_save = {
            "D_weights": self.netD.state_dict(),
            "G_weights": self.netG.state_dict(),
            "D_optimizer": self.optimizerD.state_dict(),
            "G_optimizer": self.optimizerG.state_dict(),
            "epoch": self.epoch,
            "validation_loss_list": self.validation_loss_list,
            "rng": torch.random.get_rng_state(),
        }
        save_checkpoint(checkpoint_path, dict_to_save)

    def run_batch(self, data, istrain, to_save=False):
        """
        Runs a batch of training or test data and plot the reconstruction
        in between
        """

        self.netD.zero_grad()
        b_size = data.size(0)

        # Forward pass real batch through D
        output = self.netD(data).view(-1)
        label_real = torch.zeros_like(output) + self.real_label
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label_real)
        # Calculate gradients for D in backward pass
        errD_real.backward()

        # D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size,
                            self.hparams.model_train.z_size,
                            1,
                            1,
                            device=self.hparams.device)
        # Generate fake image batch with G
        fake = self.netG(noise)

        # label.fill_(self.fake_label)
        # Classify all fake batch with D
        D_fake = self.netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch

        label_fake = torch.zeros_like(D_fake) + self.fake_label
        errD_fake = self.criterion(D_fake, label_fake)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()

        # D_G_z1 = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update D
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        # label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        GD_updated = self.netD(fake).view(-1)
        # Calculate G's loss based on this output

        errG = self.criterion(GD_updated, label_real)
        # Calculate gradients for G
        errG.backward()
        self.optimizerG.step()

        return errD.item(), errG.item()

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

        for i, (data, _) in enumerate(loader):
            data = data.to(device=self.hparams.device,
                           dtype=self.hparams.tensor_type)
            save_bool = (i == 0 and to_save)  # saves on the first batch
            errD, errG = self.run_batch(data, istrain, to_save=save_bool)

        return errD, errG

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
            Dloss_train, Gloss_train = self.run_epoch(train_loader,
                                                      istrain=True)
            note_taking(
                f'Training Epoch {epoch} D loss is {Dloss_train}, Gloss={Gloss_train}'
            )
            to_save = (epoch % hparams.train_print_freq) == 0
            save_ckpt = (epoch % hparams.checkpointing_freq) == 0

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "Dloss_train": Dloss_train,
                    "Gloss_train": Gloss_train,
                })

            note_taking(
                f'Training Epoch {epoch} Gloss_train loss is {Gloss_train},Dloss_train loss is {Dloss_train}'
            )
            best = False
            if not self.validation_loss_list or Gloss_train < min(
                    self.validation_loss_list):
                best = True
                note_taking(
                    f'new min val loss is {Gloss_train} at epoch {epoch}')
            if to_save:
                self.sample_images(best=False)
            self.validation_loss_list.append(Gloss_train)
            if best:
                self.sample_images(best=best)
                self.save(self.get_save_path('best.pth'))

            if save_ckpt or (epoch == num_epochs):
                self.save(self.get_save_path(f'checkpoint_epoch{epoch}.pth'))
        self.sample_images(best=False)  # sample when done

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
        # return self.model
