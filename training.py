import math
from datetime import datetime
import os
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiHorizonMetric
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from data import VantilatorDataModule
from lstm import LSTM
from unet import UNet


def fit_data(model_name, CV_split):

    torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    pl.utilities.seed.seed_everything(0)

    # torch.backends.cudnn.benchmark = False
    # pl.utilities.seed.seed_everything(0)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    data = VantilatorDataModule(CV_split=CV_split)

    if model_name == 'LSTM':
        model = LSTM(n_channels = data.series_input.shape[1])
    elif model_name == 'UNET':
        model = UNet(n_channels=data.series_input.shape[1])

    filename = model.name + '_CV5'+str(CV_split)
    dirpath='./checkpoints/'

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=11,
        verbose=True,
        mode='min',
        min_delta=0.0001
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        save_top_k=1,
        verbose=True,
        monitor='val_mae',
        mode='min'
    )

    trainer = pl.Trainer(   logger=pl_loggers.TensorBoardLogger('./logs/'),
                            gpus=1,
                            max_epochs=1000,
                            checkpoint_callback=True,
                            callbacks=[early_stop_callback,checkpoint_callback] )

    trainer.fit(model, data)


def eval_models(device='cuda'):

    preds = []
    model_names = ('LSTM',)#'UNET', 
    NUM_CV_SPLITS = 1

    for i in range(NUM_CV_SPLITS):

        data = VantilatorDataModule(dataset='test', CV_split=i)

        for model_name in model_names:

            print(model_name,i)

            if model_name == 'UNET':
                model = UNet.load_from_checkpoint('./checkpoints/UNET_CV5{}.ckpt'.format(i), n_channels=data.series_input.shape[1])
            else:
                model = LSTM.load_from_checkpoint('./checkpoints/LSTM_CV5{}.ckpt'.format(i), n_channels=data.series_input.shape[1])

            model.to(device)
            model.eval()

            output = []

            for idx, batch in enumerate(data.data_loader_test):

                output.append( model(batch[0].to(device)).detach().cpu().numpy().squeeze() / (NUM_CV_SPLITS*len(model_names)) )

            preds.append(np.vstack(output))

    preds = np.add(preds)

    return preds.flatten()[data.indices.astype(bool).flatten()]




if __name__ == '__main__':

    for model_name in 'LSTM','UNET':
        for i in range(5):
            print('\n',model_name,i)
            fit_data(model_name, i)

    preds = eval_models()

    submission = pd.DataFrame({ 'id': np.arange(len(preds))+1, 'pressure': preds })
    submission.to_csv('submission.csv', index=False)

