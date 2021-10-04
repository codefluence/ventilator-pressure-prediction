import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

np.set_printoptions(threshold=2000, linewidth=110, precision=4, edgeitems=20, suppress=1)
pd.set_option('display.max_rows', 600)



class VantilatorDataModule2(pl.LightningDataModule):

    def __init__(self, settings_path='./settings.json', CV_split=0):

        super(VantilatorDataModule2, self).__init__()

        DATA_PATH = 'D:/data/ventilator-pressure-prediction/'

        train = pd.read_csv(DATA_PATH+'train.csv')
        test = pd.read_csv(DATA_PATH+'test.csv')
        submission = pd.read_csv(DATA_PATH+'sample_submission.csv')

        def add_features(df):
            df['area'] = df['time_step'] * df['u_in']
            df['area'] = df.groupby('breath_id')['area'].cumsum()
            
            df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
            
            df['u_in_lag'] = df['u_in'].shift(2).fillna(0)
            
            df['R'] = df['R'].astype(str)
            df['C'] = df['C'].astype(str)
            df = pd.get_dummies(df)
            return df

        train = add_features(train)
        test = add_features(test)

        targets = train[['pressure']].to_numpy().reshape(-1, 80)
        train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
        test = test.drop(['id', 'breath_id'], axis=1)

        RS = RobustScaler()
        train = RS.fit_transform(train)
        test = RS.transform(test)

        train = train.reshape(-1, 80, train.shape[-1]).astype(np.float32)
        test = test.reshape(-1, 80, train.shape[-1]).astype(np.float32)

        ids = np.array(range(len(train)))

        idx_train = ids % 5 != CV_split
        self.data_loader_train = DataLoader(SeriesDataSet(  train[idx_train],
                                                            targets[idx_train] ), batch_size=1024, shuffle=True)

        idx_val = ids % 5 == CV_split
        self.data_loader_val = DataLoader(SeriesDataSet(    train[idx_val],
                                                            targets[idx_val] ), batch_size=2096, shuffle=False)

    def train_dataloader(self):

        return self.data_loader_train

    def val_dataloader(self):

        return self.data_loader_val



class SeriesDataSet(Dataset):
    
    def __init__(self, series_input,  series_target=None):

        super(SeriesDataSet, self).__init__()

        self.series_input = series_input
        self.series_target = series_target

    def __len__(self):

        return len(self.series_input)

    def __getitem__(self, idx):

        return self.series_input[idx], np.array(np.nan), np.array(np.nan), self.series_target[idx]


if __name__ == '__main__':

    data = VantilatorDataModule2()
