
import os
from os import walk
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule



class LSTM(LightningModule):

    def __init__(self, n_channels, hidden_size=150, num_layers=4):

        super(LSTM, self).__init__()

        self.name = 'LSTM'

        self.lstm = nn.LSTM(    input_size=n_channels, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_size*2, 1)

        self.mae_loss = torch.nn.L1Loss()

    def forward(self, series):

        output, (hn, cn) = self.lstm(torch.swapaxes(series, 1, 2))

        x = self.dense(output)
        x = F.leaky_relu(x)

        return torch.swapaxes(x, 1, 2).squeeze(1)

    def training_step(self, train_batch, batch_idx):

        series_input,series_target = train_batch

        series_output = self.forward(series_input)

        loss = self.mae_loss(series_target, series_output) #, reduction='sum'

        with torch.no_grad():
            train_mae = torch.mean(torch.abs(series_target.flatten() - series_output.flatten())).cpu().item()

        self.log('train_loss', loss.cpu().item())
        self.log('train_mae', train_mae)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series_input, series_target = val_batch

        series_output = self.forward(series_input)

        loss = self.mae_loss(series_target, series_output) #, reduction='sum'

        #TODO: nograd needed?
        val_mae = torch.mean(torch.abs(series_target.flatten() - series_output.flatten())).cpu().item()

        self.log('val_loss', loss.cpu().item())
        self.log('val_mae', val_mae)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        sccheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_mae'}


