import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from cosmo_data import CosmoData
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from vis import plt_slices, plt_power


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.current_step = 0
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        self.current_step += 1

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        if self.current_step % 1000 == 0:
            self.sample_images()

        return val_loss

    def on_validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data[:24],
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch:04d}.png",
                          normalize=True,
                          nrow=12)

        vutils.save_image(test_input.data[:24],
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"real_img_{self.logger.name}_{self.current_epoch:04d}.png",
                          normalize=True,
                          nrow=12)

        f = plt_slices(test_input.data[:6], recons.data[:6], title=['original img', 'recon'])
        self.logger.experiment.add_figure(f'recon visuals', f, global_step=self.current_step)
        f = plt_power(test_input.data[:6], recons.data[:6], label=['original img', 'recon'])
        self.logger.experiment.add_figure(f'ps visuals', f, global_step=self.current_step)

        # try:
        #     samples = self.model.sample(144,
        #                                 self.curr_device,
        #                                 labels=test_label)
        #     vutils.save_image(samples.cpu().data,
        #                       f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                       f"{self.logger.name}_{self.current_epoch}.png",
        #                       normalize=True,
        #                       nrow=12)
        # except:
        #     pass

        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        scheduler = {'scheduler': OneCycleLR(optimizer, max_lr=self.params['LR'],
                                             steps_per_epoch=len(self.train_dataloader()),
                                             epochs=self.params['max_epochs'], final_div_factor=1e4),
                     'interval': 'step'}
        scheds.append(scheduler)

        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims, scheds

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root=self.params['data_path'],
                             split="train",
                             transform=transform,
                             download=False)
        elif self.params['dataset'] == 'cosmo':
            dataset = CosmoData(train='train')

        ### Create dummy dataset for test ###
        elif self.params['dataset'] == 'dummy':
            dataset = DummyData()

        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          num_workers=self.params['num_workers'],
                          shuffle=True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader = DataLoader(CelebA(root=self.params['data_path'],
                                                       split="test",
                                                       transform=transform,
                                                       download=False),
                                                batch_size=144,
                                                shuffle=True,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)

        elif self.params['dataset'] == 'cosmo':
            self.sample_dataloader = DataLoader(CosmoData(train='val'),
                                                batch_size=self.params['batch_size'],
                                                num_workers=self.params['num_workers'],
                                                shuffle=False)
            self.num_val_imgs = len(self.sample_dataloader)
        
        ### Create dummy dataset for test ###
        elif self.params['dataset'] == 'dummy':
            self.sample_dataloader = DataLoader(DummyData(),
                                                batch_size=self.params['batch_size'],
                                                num_workers=self.params['num_workers'],
                                                shuffle=False)
            self.num_val_imgs = len(self.sample_dataloader)
        
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'cosmo':
            transform = None
        else:
            raise ValueError('Undefined dataset type')
        return transform
