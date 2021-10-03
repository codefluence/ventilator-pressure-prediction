import math
from datetime import datetime
import os
import json
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from data import VantilatorDataModule
from model import PatternFinder, PatternFinderDilated
from model_lstm import LSTMfinder

from unet import UNet

def fit_model(CV_split):

    torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    pl.utilities.seed.seed_everything(0)

    # torch.backends.cudnn.benchmark = False
    # pl.utilities.seed.seed_everything(0)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    data = VantilatorDataModule(CV_split=CV_split)

    model = UNet(n_channels=data.series_input.shape[1], n_classes=1)
    #model = LSTMfinder(in_channels = data.series_input.shape[1]+33)

    #model = PatternFinder(in_channels = data.series_input.shape[1], series_length=data.series_input.shape[2])    

    filename = model.name + '_CV5'+str(CV_split)
    dirpath='./checkpoints/'

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=8,
        verbose=True,
        mode='min',
        min_delta=0.001
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
                            max_epochs=10000,
                            checkpoint_callback=True,
                            callbacks=[early_stop_callback,checkpoint_callback] )
    trainer.fit(model, data)

    return data




def eval_models(data, settings_path='./settings.json', device='cuda'):

    inputs = []
    output = []
    targets = []
    features = []

    NUM_MODELS = 1

    for i in range(NUM_MODELS):

        print('model:',i)

        model = PatternFinder.load_from_checkpoint('./checkpoints/CNN_CV5{}.ckpt'.format(i), in_channels=data.series_input.shape[1])

        model.to(device)
        model.eval()

        for idx, batch in enumerate(data.data_loader_val):

            inputs.append( batch[0].detach().cpu().numpy().squeeze() )
            features.append( batch[1].detach().cpu().numpy().squeeze() )
            output.append( model(batch[0].to(device),batch[1].to(device)).detach().cpu().numpy().squeeze() /NUM_MODELS )
            targets.append( batch[2].detach().cpu().numpy().squeeze() )

    return np.vstack(output), np.vstack(targets), np.vstack(inputs), np.vstack(features)

def eval_models_test(data, settings_path='./settings.json', device='cuda'):

    output = []

    NUM_MODELS = 1

    for i in range(NUM_MODELS):

        print('model:',i)

        model = PatternFinder.load_from_checkpoint('./checkpoints/CNN_CV5{}.ckpt'.format(i), in_channels=data.series_input.shape[1])

        model.to(device)
        model.eval()

        for idx, batch in enumerate(data.data_loader_test):

            output.append( model(batch[0].to(device),batch[1].to(device)).detach().cpu().numpy().squeeze() /NUM_MODELS )

    return np.vstack(output)





if __name__ == '__main__':
    

    # for i in range(5):
    #     print('model:',i)
    #     fit_model(data = VantilatorDataModule(CV_split=i))


    #EVAL VAL
    data = fit_model(1)
    #data = VantilatorDataModule(CV_split=0)
    output, targets, _, _ = eval_models(data=data)
    print('mae:',round(np.mean(np.abs(targets.flatten() - output.flatten())),4))



    # EVAL TEST
    # output = eval_models_test(data=VantilatorDataModule(CV_split=-1)).flatten() 

    # submission = pd.DataFrame({ 'id': np.arange(len(output))+1, 'pressure': output})
    # submission.to_csv('submission.csv', index=False)

