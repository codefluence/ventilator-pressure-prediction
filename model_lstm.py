
import os
from os import walk
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule



class LSTMfinder(LightningModule):

    def __init__(self, in_channels, hidden_size=100, num_layers=3):

        super(LSTMfinder, self).__init__()

        self.name = 'LSTM'

        self.lstm   = nn.LSTM(  input_size=in_channels, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2 + 6, 1)

    def forward(self, series, features, logits):

        x = torch.cat((series, logits),dim=1)

        output, (hn, cn) = self.lstm(torch.swapaxes(x, 1, 2))

        x = self.linear(torch.cat((output, torch.swapaxes(series, 1, 2)),dim=2))
        x = F.leaky_relu(x)

        return torch.swapaxes(x, 1, 2).squeeze(1)

    def training_step(self, train_batch, batch_idx):

        series_input, features, logits, series_target = train_batch

        series_output = self.forward(series_input, features, logits)

        #loss = torch.mean(torch.square(series_target - series_output))
        loss = F.mse_loss(series_target, series_output, reduction='sum')

        with torch.no_grad():
            train_mae = torch.mean(torch.abs(series_target.flatten() - series_output.flatten())).cpu().item()

        self.log('train_loss', loss.cpu().item())
        self.log('train_mae', train_mae)

        return loss

    def validation_step(self, val_batch, batch_idx):

        series_input, features, logits, series_target = val_batch

        series_output = self.forward(series_input, features, logits)

        #loss = torch.mean(torch.square(series_target - series_output))
        loss = F.mse_loss(series_target, series_output, reduction='sum') 

        #TODO: nograd needed?
        val_mae = torch.mean(torch.abs(series_target.flatten() - series_output.flatten())).cpu().item()

        self.log('val_loss', loss.cpu().item())
        self.log('val_mae', val_mae)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/4, verbose=True, min_lr=1e-5)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_mae'}


