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

        for weight in self.vae_model.parameters():
            weight.requires_grad = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        input_imgs, labels = batch
        vecs = self.vae_model.encode(input_imgs)[0]
        results = self.forward(vecs.detach())
        train_loss = self.model.train_loss_function(results, labels)

        self.logger.experiment.log(
            {(key if key == 'R_squared_score' else 'RMSE_loss'): val for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        input_imgs, labels = batch
        vecs = self.vae_model.encode(input_imgs)[0]
        results = self.forward(vecs.detach())
        val_loss = self.model.valid_loss_function(results, labels)
        return val_loss

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        input_imgs, labels = batch
        vecs = self.vae_model.encode(input_imgs)[0]
        results = self.forward(vecs.detach())
        test_loss = self.model.valid_loss_function(results, labels)
        return test_loss

    def on_validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
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
        return optims, scheds

    @data_loader
    def train_dataloader(self):
        if self.params['dataset'] == 'cosmo':
            dataset = CosmoData(train='train', load_every=self.params['load_every'], exp=self.params['exp'])
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
            return DataLoader(dataset,
                              batch_size=self.params['batch_size'],
                              num_workers=self.params['num_workers'],
                              shuffle=True,
                              drop_last=True)
        else:
            raise ValueError('Undefined dataset type')


    @data_loader
    def val_dataloader(self):
        if self.params['dataset'] == 'cosmo':
            self.sample_dataloader = DataLoader(
                CosmoData(train='val', load_every=self.params['load_every'], exp=self.params['exp']),
                batch_size=self.params['batch_size'],
                num_workers=self.params['num_workers'],
                shuffle=False,
                pin_memory=True,
                collate_fn=batchify
            )

        # Create dummy dataset for test
        elif self.params['dataset'] == 'dummy':
            self.sample_dataloader = DataLoader(DummyData(),
                                                batch_size=self.params['batch_size'],
                                                num_workers=self.params['num_workers'],
                                                shuffle=False)

        else:
            raise ValueError('Undefined dataset type')
        return self.sample_dataloader

    @data_loader
    def test_dataloader(self):
        if self.params['dataset'] == 'cosmo':
            self.sample_dataloader = DataLoader(
                CosmoData(train='val', load_every=self.params['load_every'], exp=self.params['exp']),
                batch_size=self.params['batch_size'],
                num_workers=self.params['num_workers'],
                shuffle=False,
                pin_memory=True,

                collate_fn=batchify
            )
        # Create dummy dataset for test
        elif self.params['dataset'] == 'dummy':
            self.sample_dataloader = DataLoader(DummyData(),
                                                batch_size=self.params['batch_size'],
                                                num_workers=self.params['num_workers'],
                                                shuffle=False)

        else:
            raise ValueError('Undefined dataset type')
        return self.sample_dataloader

    def data_transforms(self):
        pass
