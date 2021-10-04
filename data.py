import os
import gc
import json
import math
import pickle
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from unet import UNet

# from sklearn.preprocessing import MinMaxScaler
# from scipy.signal import find_peaks
# from scipy.interpolate import interp1d
# from scipy.signal import argrelmax
#from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=2000, linewidth=110, precision=4, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 600)



class VantilatorDataModule(pl.LightningDataModule):

    def __init__(self, settings_path='./settings.json', CV_split=0, fixed_length=128, scale=True):

        super(VantilatorDataModule, self).__init__()

        DATA_PATH = 'D:/data/ventilator-pressure-prediction/'

        file_name = 'test' if CV_split==-1 else 'train'

        df = pd.read_csv(DATA_PATH + file_name + '.csv',index_col=0)
        df['area'] = df['time_step'] * df['u_in']
        df['area'] = df.groupby('breath_id')['area'].cumsum()

        cols = list(df.columns)

        if 'pressure' in cols:
            df = df[cols[:6] + cols[7:] + cols[6:7]]

        data = df.to_numpy(dtype=np.float32)

        # data[data[:,0]==8522][40,3]
        # data[data[:,0]==44245][40,3]

        # np.mean(np.diff(data[data[:,0]==44245][:,3]))
        # np.mean(np.diff(data[data[:,0]==8522][:,3]))

        breaths = np.split(data.T[1:], np.unique(data.T[0], return_index=True)[1][1:], axis=1)
        breaths = np.stack(breaths)

        num_channels = breaths.shape[1] - 3
        series = np.zeros((len(breaths), num_channels, fixed_length+1))
        series[:] = np.nan

        trange = 3/(fixed_length+1)

        for i in tqdm(range(fixed_length+1)):

            time_step = breaths[:,2]
            info_in_range = ((trange*i - trange/2) < time_step) & (time_step <= (trange*(i+1) - trange/2))

            # print(i, (trange*i - trange/2), (trange*(i+1) - trange/2), info_in_range.sum())

            # 8522 o 44245, no recuerdo
            # if i == 6:
            #     print('lol')

            idx = (info_in_range).any(axis=1)
            info_in_range = np.expand_dims(info_in_range,1)[idx]
            series[idx,:,i] = (breaths[idx,-num_channels:]*info_in_range).sum(axis=2) / info_in_range.sum(axis=2)

            # if idx.sum()>0:
            #     print(np.min(info_in_range.sum(axis=2)), np.max(info_in_range.sum(axis=2)))

        # linear iterpolation
        ret = np.cumsum(np.nan_to_num(series), axis=-1)
        ret[:,:,3:] = ret[:,:,3:] - ret[:,:,:-3]
        ret[:,:,2:] = ret[:,:,2:] / 2

        #34
        series = np.nan_to_num(series) + np.isnan(series) * np.roll(ret, -1, axis=-1)
        series = series[:,:,:-1]

        assert((~np.isfinite(series)).sum() == 0)

        if 'pressure' in cols:
            self.series_target = series[:,-1].copy().astype(np.float32)

        series = np.concatenate((series[:,:-1],series[:,:2].copy(),series[:,:2].copy()), axis=1)
        series[:,-4] = np.expand_dims(breaths[:,0,0],axis=1)
        series[:,-3] = np.expand_dims(breaths[:,1,0],axis=1)
        series[:,-2] = np.cumsum(series[:,0], axis=-1)
        series[:,-1] = np.roll(series[:,0], 2, axis=-1)
        series[:,-1,:2] = np.expand_dims(series[:,0,0],axis=1)

        if scale and CV_split != -1:

            ids = np.array(range(len(series)))
            idx_train = ids % 5 != CV_split
            series = series - np.mean(series[idx_train],axis=(0,2)).reshape(-1,1)
            series = series / np.std(series[idx_train],axis=(0,2)).reshape(-1,1)

        self.series_input = series.astype(np.float32)

        self.features = np.hstack((np.unique(breaths[:,0],axis=1), np.unique(breaths[:,1],axis=1))).astype(np.float32)

        #TODO: revisar el ultimo numero, para series cortas este numero podria ser nan reemplazado con el primer numero

        print('series_input shape:',self.series_input.shape)
        #print('series_target shape:',self.series_target.shape)
        print('features shape:',self.features.shape)

        #TODO: sorting

        if CV_split == -1:

            self.data_loader_test = DataLoader(SeriesDataSet(   self.series_input,
                                                                self.features ), batch_size=2096, shuffle=False)
        else:
            ids = np.array(range(len(self.series_input)))

            idx_train = ids % 5 != CV_split
            self.data_loader_train = DataLoader(SeriesDataSet(  self.series_input[idx_train],
                                                                self.features[idx_train],
                                                                self.series_target[idx_train] ), batch_size=1024, shuffle=True)

            idx_val = ids % 5 == CV_split
            self.data_loader_val = DataLoader(SeriesDataSet(    self.series_input[idx_val],
                                                                self.features[idx_val],
                                                                self.series_target[idx_val] ), batch_size=2096, shuffle=False)

    def train_dataloader(self):

        return self.data_loader_train

    def val_dataloader(self):

        return self.data_loader_val



class SeriesDataSet(Dataset):
    
    def __init__(self, series_input, features, series_target=None):

        super(SeriesDataSet, self).__init__()

        self.series_input = series_input
        self.features = features
        self.series_target = series_target

        fname = './checkpoints/UNET_CV51.ckpt'
        if os.path.isfile(fname):

            self.UNET_logits = np.zeros((len(series_input),33,series_input.shape[2])).astype(np.float32)

            model = UNet.load_from_checkpoint(fname, n_channels=series_input.shape[1], n_classes=1)
            model.to('cuda')
            model.eval()

            batch_size = 2**12
            num_batches = math.ceil(len(series_input) / batch_size)

            for bidx in tqdm(range(num_batches)):

                start = bidx*batch_size
                end   = start + min(batch_size, len(series_input) - start)

                if start == end:
                    break

                mminput = series_input[start:end]

                output = model(torch.tensor(mminput, dtype=torch.float32, device='cuda'),
                            torch.tensor(features[start:end], dtype=torch.float32, device='cuda'))

                o0 = output[0].detach().cpu().numpy()
                o1 = output[1].detach().cpu().numpy()

                self.UNET_logits[start:end] = np.concatenate((np.expand_dims(o0,1),o1),axis=1)

    def __len__(self):

        return len(self.series_input)

    def __getitem__(self, idx):

        batch_row = self.series_input[idx], self.features[idx]

        if hasattr(self, 'UNET_logits'):
            batch_row += (self.UNET_logits[idx],)
        else:
            batch_row += (np.array(np.nan),)

        if not self.series_target is None:
            batch_row += (self.series_target[idx],)

        return batch_row


if __name__ == '__main__':

    data = VantilatorDataModule()
