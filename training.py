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

from model import PatternFinder, PatternFinderDilated
from lstm import LSTMfinder
from unet import UNet


class MAE2(MultiHorizonMetric):

    def loss(self, y_pred, target):
        return F.mse_loss(target, self.to_prediction(y_pred), reduction='mean')

def fit_model(CV_split):

    torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    pl.utilities.seed.seed_everything(0)

    # torch.backends.cudnn.benchmark = False
    # pl.utilities.seed.seed_everything(0)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    model_name = 'TFTRAN'
    monitor = 'val_mae'

    if model_name == 'LSTM':
        data = VantilatorDataModule(CV_split=CV_split)#, fixed_length=80, scale=False
        model = LSTMfinder(in_channels = data.series_input.shape[1])   #12 para el de kaggle
    elif model_name == 'UNET':
        data = VantilatorDataModule(CV_split=CV_split)
        model = UNet(n_channels=data.series_input.shape[1])
    elif model_name == 'TFTRAN':

        monitor = 'val_loss'

        DATA_PATH = 'D:/data/ventilator-pressure-prediction/'
        df = pd.read_csv(DATA_PATH + 'train.csv', index_col=0) #, nrows=5000
        df['area'] = df['time_step'] * df['u_in']
        df['area'] = df.groupby('breath_id')['area'].cumsum()

        df['time_step'] = df.groupby('breath_id')['time_step'].cumcount()

        df['R'] = df['R'].astype(str)
        df['C'] = df['C'].astype(str)
        df = pd.get_dummies(df, dtype=str)

        ids = np.array(range(len(df)))
        df_train = df[df.breath_id % 5 != CV_split]
        df_val = df[df.breath_id % 5 == CV_split]

        training = TimeSeriesDataSet(
            df_train,
            time_idx='time_step',
            target="pressure",
            group_ids=["breath_id"],
            min_encoder_length=0,  
            max_encoder_length=30,
            min_prediction_length=80,
            max_prediction_length=80,
            static_categoricals=['R_20', 'R_5', 'R_50', 'C_10', 'C_20', 'C_50'],
            static_reals=["u_in", "u_out", "area"],
            time_varying_known_categoricals=[],  
            time_varying_known_reals=['u_out'],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=["u_in", "area"],
            categorical_encoders={'breath_id': NaNLabelEncoder(add_nan=True)},
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=100,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=1,
            loss=MAE2(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        model.name = 'TFTRAN'

        validation = TimeSeriesDataSet.from_dataset(training, df_val, predict=True, stop_randomization=True)

        train_dataloader = training.to_dataloader(train=True, batch_size=1024, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=2048, num_workers=0)

    else:
        data = VantilatorDataModule(CV_split=CV_split)
        model = PatternFinder(in_channels = data.series_input.shape[1], series_length=data.series_input.shape[2])

    filename = model.name + '_CV5'+str(CV_split)
    dirpath='./checkpoints/'

    early_stop_callback = EarlyStopping(
        monitor=monitor,
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
        monitor=monitor,
        mode='min'
    )

    trainer = pl.Trainer(   logger=pl_loggers.TensorBoardLogger('./logs/'),
                            gpus=1,
                            max_epochs=10000,
                            checkpoint_callback=True,
                            callbacks=[early_stop_callback,checkpoint_callback] )

    if model_name == 'TFTRAN':

        trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
        )
    else:
        trainer.fit(model, data)





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

