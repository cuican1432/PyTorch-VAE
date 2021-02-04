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
from cosmo_data import CosmoData, DummyData, batchify
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from vis import plt_slices, plt_power


class RegExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 reg_model: BaseVAE,
                 params: dict) -> None:
        super(RegExperiment, self).__init__()

        self.vae_model = vae_model
        self.model = reg_model
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
        self.current_step += 1
        input_imgs, labels = batch
        self.curr_device = input_imgs.device

        results = self.forward(input_imgs, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        input_imgs, labels = batch
        self.curr_device = input_imgs.device

        results = self.forward(input_imgs, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        return val_loss

    def on_validation_end(self, outputs):
        avg_loss = torch.stack([x['R_squared_score'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

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
            dataset = CosmoData(train='train', load_every=self.params['load_every'])
            self.num_train_imgs = len(dataset)
            return DataLoader(dataset,
                              batch_size=self.params['batch_size'],
                              num_workers=self.params['num_workers'],
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=batchify)

        # Create dummy dataset for test
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
            self.sample_dataloader = DataLoader(CosmoData(train='val', load_every=self.params['load_every']),
                                                batch_size=self.params['batch_size'],
                                                num_workers=self.params['num_workers'],
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=batchify
                                                )
            self.num_val_imgs = len(self.sample_dataloader)

        # Create dummy dataset for test
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
        elif self.params['dataset'] == 'dummy':
            transform = None
        else:
            raise ValueError('Undefined dataset type')
        return transform
