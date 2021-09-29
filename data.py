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

# from sklearn.preprocessing import MinMaxScaler
# from scipy.signal import find_peaks
# from scipy.interpolate import interp1d
# from scipy.signal import argrelmax
#from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=2000, linewidth=110, precision=4, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 600)



class VantilatorDataModule(pl.LightningDataModule):

    def __init__(self, settings_path='./settings.json', CV_split=0, fixed_length=80):

        super(VantilatorDataModule, self).__init__()

        self.CV_split = CV_split

        DATA_PATH = 'D:/data/ventilator-pressure-prediction/'

        file_name = 'test' if CV_split==-1 else 'train'
        data = pd.read_csv(DATA_PATH + file_name + '.csv',index_col=0).to_numpy()

        # data[data[:,0]==8522][40,3]
        # data[data[:,0]==44245][40,3]

        # np.mean(np.diff(data[data[:,0]==44245][:,3]))
        # np.mean(np.diff(data[data[:,0]==8522][:,3]))

        breaths = np.split(data.T[1:], np.unique(data.T[0], return_index=True)[1][1:], axis=1)
        breaths = np.stack(breaths)

        num_channels = breaths.shape[1] - 3
        series = np.zeros((len(breaths), num_channels, fixed_length+1))
        series[:] = np.nan

        trange = 2.5/(fixed_length+1)

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

            pass

        ret = np.cumsum(np.nan_to_num(series), axis=-1)
        ret[:,:,3:] = ret[:,:,3:] - ret[:,:,:-3]
        ret[:,:,2:] = ret[:,:,2:] / 2

        #34
        series = np.nan_to_num(series) + np.isnan(series) * np.roll(ret, -1, axis=-1)
        series = series[:,:,:-1]

        assert((~np.isfinite(series)).sum() == 0)

        self.series_input = series[:,:2].astype(np.float32)

        if series.shape[1]>2:
            self.series_target = series[:,-1].astype(np.float32)

        self.features = np.hstack((np.unique(breaths[:,0],axis=1), np.unique(breaths[:,1],axis=1))).astype(np.float32)

        #TODO: revisar el ultimo numero, para series cortas este numero podria ser nan reemplazado con el primer numero

        print('series_input shape:',self.series_input.shape)
        #print('series_target shape:',self.series_target.shape)
        print('features shape:',self.features.shape)

        #TODO: sorting

        if CV_spli == -1:

            self.data_loader_test = DataLoader(SeriesDataSet(   self.series_input,
                                                                self.features,
                                                                self.series_target ), batch_size=2096, shuffle=False)
        else:
            ids = np.array(range(len(self.series_input)))

            idx_train = ids % 5 != self.CV_split
            self.data_loader_train = DataLoader(SeriesDataSet(  self.series_input[idx_train],
                                                                self.features[idx_train],
                                                                self.series_target[idx_train] ), batch_size=1024, shuffle=True)

            idx_val = ids % 5 == self.CV_split
            self.data_loader_val = DataLoader(SeriesDataSet(    self.series_input[idx_val],
                                                                self.features[idx_val],
                                                                self.series_target[idx_val] ), batch_size=2096, shuffle=False)

    def train_dataloader(self):

        return self.data_loader_train

    def val_dataloader(self):

        return data_loader_val



class SeriesDataSet(Dataset):
    
    def __init__(self, series_input, features, series_target):

        super(SeriesDataSet, self).__init__()

        self.series_input = series_input
        self.features = features
        self.series_target = series_target

    def __len__(self):

        return len(self.series_input)

    def __getitem__(self, idx):

        return self.series_input[idx], self.features[idx], self.series_target[idx]


if __name__ == '__main__':

    data = VantilatorDataModule(CV_split=0)
