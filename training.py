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
from model import PatternFinder

def fit_model(CV_split):

    torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    pl.utilities.seed.seed_everything(0)

    # torch.backends.cudnn.benchmark = False
    # pl.utilities.seed.seed_everything(0)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    data = VantilatorDataModule(CV_split=CV_split)

    model = PatternFinder(in_channels = data.series_input.shape[1])

    filename = 'CNN_CV5'+str(CV_split)
    dirpath='./checkpoints/'

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=8,
        verbose=True,
        mode='min',
        #min_delta=0.0003
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
                            max_epochs=100,
                            checkpoint_callback=True,
                            callbacks=[early_stop_callback,checkpoint_callback] )
    trainer.fit(model, data)




def eval_models(data, settings_path='./settings.json', device='cuda'):

    output = np.zeros((len(data.series_input),80))

    NUM_MODELS = 1

    for i in range(NUM_MODELS):

        print('model:',i)

        model = PatternFinder.load_from_checkpoint('./checkpoints/CNN_CV5{}.ckpt'.format(i), in_channels=data.series_input.shape[1])

        model.to(device)
        model.eval()

        batch_size = 2**14
        num_batches = math.ceil(len(data.series_input) / batch_size)

        for bidx in tqdm(range(num_batches)):

            start = bidx*batch_size
            end   = start + min(batch_size, len(data.series_input) - start)

            if start == end:
                break

            mminput = data.series_input[start:end]

            output[start:end] += model(torch.tensor(mminput, dtype=torch.float32, device=device)).detach().cpu().numpy().squeeze()

    return output/NUM_MODELS


def create_submission_file():

    output = eval_models(data=VantilatorDataModule(CV_split=-1)).flatten() 

    submission = pd.DataFrame({ 'id': np.arange(len(output))+1, 'pressure': output})
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    
    #fit_model(data = VantilatorDataModule())

    # for i in range(5):
    #     print('model:',i)
    #     fit_model(data = VantilatorDataModule(CV_split=i))

    data = VantilatorDataModule(CV_split=1)
    target = data.series_target.flatten()
    output = eval_models(data=data).flatten()

    print('mae:',round(np.mean(np.abs(target - output)),4))
    print('done')

