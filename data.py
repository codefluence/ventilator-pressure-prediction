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

from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=2000, linewidth=110, precision=4, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 600)

DATA_PATH = 'D:/data/ventilator-pressure-prediction/'


class VantilatorDataModule(pl.LightningDataModule):

    def __init__(self, dataset='train', CV_split=0, span=0.5):

        super(VantilatorDataModule, self).__init__()

        fixed_length = int(128 * span)

        df = pd.read_csv(DATA_PATH + dataset + '.csv',index_col=0)

        ################## picked from https://www.kaggle.com/dmitryuarov/ventilator-pressure-eda-lstm-0-189 ##################
        df['ewm_u_in_mean'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).mean().reset_index(level=0,drop=True)
        df['ewm_u_in_std'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0,drop=True)
        df['ewm_u_in_corr'] = df.groupby('breath_id')['u_in'].ewm(halflife=10).corr().reset_index(level=0,drop=True)

        df['rolling_10_mean'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).mean().reset_index(level=0,drop=True)
        df['rolling_10_max'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).max().reset_index(level=0,drop=True)
        df['rolling_10_std'] = df.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).std().reset_index(level=0,drop=True)

        df['expand_mean'] = df.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0,drop=True)
        df['expand_max'] = df.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0,drop=True)
        df['expand_std'] = df.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0,drop=True)
        #######################################################################################################################

        ################## picked from https://www.kaggle.com/carlmcbrideellis/tpu-ventilator-lstm ##################
        # my features
        df['u_in_lag_1'] = df.groupby('breath_id')['u_in'].shift(1).fillna(method="backfill")
        df['u_in_delta_1'] = df['u_in'] - df['u_in_lag_1']
        df['u_in_lag_2'] = df.groupby('breath_id')['u_in'].shift(2).fillna(method="backfill")
        df['u_in_delta_02'] = df['u_in']       - df['u_in_lag_2']
        df['u_in_delta_12'] = df['u_in_lag_1'] - df['u_in_lag_2']
        df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
        
        # descriptive features
        df['u_in_mean'] = df.groupby('breath_id')['u_in'].transform('mean')
        df['u_in_max'] = df.groupby('breath_id')['u_in'].transform('max')
        df['u_in_min'] = df.groupby('breath_id')['u_in'].transform('min')
        df['u_in_median'] = df.groupby('breath_id')['u_in'].transform('median')
        df['first_value_u_in'] = df.groupby('breath_id')['u_in'].transform('first')
        df["u_in_sum"] = df.groupby("breath_id")["u_in"].transform("sum")
        df["u_in_std"] = df.groupby("breath_id")["u_in"].transform("std")
        # this is a guide to the initial pressure 
        df['last_value_u_in'] = df.groupby('breath_id')['u_in'].transform('last')

        # Lukasz Borecki features
        df['u_in_change']   = df['u_in'].shift(-1, fill_value=0) - df['u_in']
        df['delta_time']    = df['time_step'].shift(-1, fill_value=0) - df['time_step']
        df['uin_in_time']   = df['u_in_change']/df['delta_time']
        
        # categoricals
        df['R__C'] = df['R'].astype(str) + '__' + df['C'].astype(str)
        df = pd.get_dummies(df)
        #######################################################################################################################

        df = df.fillna(0)

        cols = list(df.columns)

        #placing pressure column as the last column
        if 'pressure' in cols:
            pressure = df['pressure']
            df = df.drop(['pressure'], axis=1)
            df['pressure'] = pressure

        data = df.to_numpy(dtype=np.float32)

        # from tabular data to tensors
        breaths = np.split(data.T[1:], np.unique(data.T[0], return_index=True)[1][1:], axis=1)
        breaths = np.stack(breaths)

        # tensors with time series of length "fixed_length"
        series = np.zeros((breaths.shape[0], breaths.shape[1], fixed_length+1))
        series[:] = np.nan

        # indices to the original times, to use for the final submission
        indices = np.zeros((breaths.shape[0], fixed_length+1))
        indices[:] = np.nan

        # one time step added to the end for interpolation
        trange = (3*span) / (fixed_length+1)

        # scaling all the time series to a [0,3] interval
        for i in tqdm(range(fixed_length+1)):

            time_step = breaths[:,2]
            info_in_range = ((trange*i - trange/2) < time_step) & (time_step <= (trange*(i+1) - trange/2))

            idx = (info_in_range).any(axis=1)
            indices[:,i] = idx

            info_in_range = np.expand_dims(info_in_range,1)[idx]
            series[idx,:,i] = (breaths[idx]*info_in_range).sum(axis=2) / info_in_range.sum(axis=2)

        self.indices = indices[:,:-1]

        # linear interpolation
        ret = np.cumsum(np.nan_to_num(series), axis=-1)
        ret[:,:,3:] = ret[:,:,3:] - ret[:,:,:-3]
        ret[:,:,2:] = ret[:,:,2:] / 2

        series = np.nan_to_num(series) + np.isnan(series) * np.roll(ret, -1, axis=-1)
        series = series[:,:,:-1]

        assert((~np.isfinite(series)).sum() == 0)

        if 'pressure' in cols:

            self.orig_pressure = breaths[:,-1]
            self.series_target = (series[:,-1]).astype(np.float32)
            series = series[:,:-1]

        else:
            self.series_target[:] = np.nan

        # u_in = series[:,3]
        # cum_integral = np.cumsum(u_in, axis=-1)
        # first_derivative = np.diff(u_in, prepend=0)
        # second_derivative = np.diff(first_derivative, prepend=0)

        # series = np.concatenate((   series,
        #                             np.expand_dims(cum_integral, axis=1),
        #                             np.expand_dims(first_derivative, axis=1),
        #                             np.expand_dims(second_derivative, axis=1)), axis=1)

        assert((~np.isfinite(series)).sum() == 0)

        semf = './checkpoints/series_means_5{}.npy'.format(CV_split)
        sesf = './checkpoints/series_stds_5{}.npy'.format(CV_split)

        ids = np.array(range(len(series)))
        idx_train = ids % 5 != CV_split
        idx_val = ids % 5 == CV_split

        if dataset == 'train':

            series_means = np.mean(series[idx_train],axis=(0,2)).reshape(-1,1)
            np.save(semf, series_means)

            series_stds = np.std(series[idx_train],axis=(0,2)).reshape(-1,1)
            np.save(sesf, series_stds)

            self.series_input = (series - series_means) / series_stds
        else:

            self.series_input = (series - np.load(semf)) / np.load(sesf)

        self.series_input[:,-9:] = series[:,-9:]
        self.series_input = self.series_input.astype(np.float32)

        # 'R' and 'C' as tabular data (already fed as channels)
        #self.features = np.hstack((np.unique(breaths[:,0],axis=1), np.unique(breaths[:,1],axis=1))).astype(np.float32)
        
        self.data_loader_train = DataLoader(SeriesDataSet(  self.series_input[idx_train],
                                                            self.series_target[idx_train] ), batch_size=512, shuffle=True)

        self.data_loader_val = DataLoader(SeriesDataSet(    self.series_input[idx_val],
                                                            self.series_target[idx_val] ), batch_size=4192, shuffle=False)

        self.data_loader_test = DataLoader(SeriesDataSet(   self.series_input, self.series_target ), batch_size=1024, shuffle=False)

    def train_dataloader(self):

        return self.data_loader_train

    def val_dataloader(self):

        return self.data_loader_val



class SeriesDataSet(Dataset):
    
    def __init__(self, series_input, series_target):

        super(SeriesDataSet, self).__init__()

        self.series_input = series_input
        self.series_target = series_target

    def __len__(self):

        return len(self.series_input)

    def __getitem__(self, idx):

        return self.series_input[idx], self.series_target[idx]


