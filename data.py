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

    def __init__(self, settings_path='./settings.json', device='cuda', batch_size=20000, scale=True,
                 kernel_size=6, stride=4, mks=30, CV_split=0, fixed_length=150):

        super(VantilatorDataModule, self).__init__()

        DATA_PATH = 'D:/data/ventilator-pressure-prediction/'

        train_data = pd.read_csv(DATA_PATH+'train.csv',index_col=0).to_numpy(np.float32)
        test_data  = pd.read_csv(DATA_PATH+'test.csv', index_col=0).to_numpy(np.float32)

        # train_data[train_data[:,0]==8522][40,3]
        # train_data[train_data[:,0]==44245][40,3]

        # np.mean(np.diff(train_data[train_data[:,0]==44245][:,3]))
        # np.mean(np.diff(train_data[train_data[:,0]==8522][:,3]))

        a = np.split(train_data[:,1:], np.unique(train_data[:, 0], return_index=True)[1][1:])
        b = np.stack(a)

        finalcrap = np.zeros((len(b),80))
        finalcrap[:] = np.nan

        ii = 2.5/80

        for i in range(80):

            acagar = b[:,:,2]
            tot = ((ii*i - ii/2) < acagar) & (acagar <= (ii*(i+1) - ii/2))
            
            #print(i,(ii*i - ii/2), (ii*(i+1) - ii/2), tot.sum())

            # if i == 6:
            #     print('lol')

            idx = (tot).any(axis=1)
            #finalcrap[idx,i] = (acagar*tot)[idx].sum(axis=1)
            finalcrap[idx,i] = (b[:,:,5]*tot)[idx].sum(axis=1) / tot[idx].sum(axis=1)
            #print(np.min(tot[idx].sum(axis=1)), np.max(tot[idx].sum(axis=1)))

            pass

        ret = np.cumsum(np.nan_to_num(finalcrap), dtype=np.float32, axis=1)
        ret[:,3:] = ret[:,3:] - ret[:,:-3]
        ret[:,2:] = ret[:,2:] / 2

        #34
        finalcrap2 = np.nan_to_num(finalcrap) + np.isnan(finalcrap) * np.roll(ret, -1, axis=-1)

        assert((~np.isfinite(finalcrap2)).sum() == 0)


        print('done')

    def train_dataloader(self):

        #TODO: sorting por volatilidad
        idx = self.targets[:,0] % 5 != self.CV_split
        return DataLoader(SeriesDataSet(self.series[idx], self.stats[idx], self.targets[idx]), batch_size=512, shuffle=True)

    def val_dataloader(self):

        idx = self.targets[:,0] % 5 == self.CV_split
        return DataLoader(SeriesDataSet(self.series[idx], self.stats[idx], self.targets[idx]), batch_size=4096, shuffle=False)



class SeriesDataSet(Dataset):
    
    def __init__(self, series, stats, targets):

        super(SeriesDataSet, self).__init__()

        self.series = series
        self.stats = stats
        self.targets = targets

    def __len__(self):

        return len(self.series)

    def __getitem__(self, idx):

        return self.series[idx], self.stats[idx], self.targets[idx]


if __name__ == '__main__':

    data = VantilatorDataModule()
