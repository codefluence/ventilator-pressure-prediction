
import os
from os import walk
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule



class PatternFinder(LightningModule):

    def __init__(self, in_channels, multf=2):

        super(PatternFinder, self).__init__()

        self.ks = 3
        self.st = 1

        self.batch_norm_1d = nn.BatchNorm1d(in_channels)

        self.conv1 = nn.Conv1d(in_channels, in_channels*multf, kernel_size=self.ks, stride=self.st, padding=1)
        self.conv2 = nn.Conv1d(in_channels*multf, in_channels*multf**2, kernel_size=self.ks, stride=self.st, padding=1)
        self.conv3 = nn.Conv1d(in_channels*multf**2, in_channels*multf**3, kernel_size=self.ks, stride=self.st, padding=1)
        self.conv4 = nn.Conv1d(in_channels*multf**3, in_channels*multf**4, kernel_size=self.ks, stride=self.st, padding=1)

        self.conv4r = nn.Conv1d(in_channels*multf**4, in_channels*multf**3, kernel_size=self.ks, stride=self.st, padding=1)
        self.conv3r = nn.Conv1d(in_channels*multf**3, in_channels*multf**2, kernel_size=self.ks, stride=self.st, padding=1)
        self.conv2r = nn.Conv1d(in_channels*multf**2, in_channels*multf, kernel_size=self.ks, stride=self.st, padding=1)
        self.conv1r = nn.Conv1d(in_channels*multf, in_channels, kernel_size=self.ks, stride=self.st, padding=1)

        self.convf = nn.Conv1d(in_channels, in_channels//2, kernel_size=self.ks, stride=self.st, padding=1)

    def forward(self, series):

        x = series

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = F.leaky_relu(x)

        x = self.conv4r(x)
        x = F.leaky_relu(x)

        x = self.conv3r(x)
        x = F.leaky_relu(x)

        x = self.conv2r(x)
        x = F.leaky_relu(x)

        x = self.conv1r(x)
        x = F.leaky_relu(x)

        x = self.convf(x)
        x = F.leaky_relu(x)

        return x

    def training_step(self, train_batch, batch_idx):

        series_input, features, series_target = train_batch

        series_output = self.forward(series_input).squeeze(1)

        loss = F.mse_loss(series_target, series_output, reduction='sum') 

        self.log('train_loss', loss.cpu().item())

        return loss

    def validation_step(self, val_batch, batch_idx):

        series_input, features, series_target = val_batch

        series_output = self.forward(series_input).squeeze(1)

        loss = F.mse_loss(series_target, series_output, reduction='sum') 

        val_mae = loss.cpu().item()

        self.log('val_loss', loss.cpu().item())
        self.log('val_mae', val_mae)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/3, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_mae'}


