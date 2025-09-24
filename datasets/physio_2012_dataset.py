###############
#   Package   #
###############
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Tuple, List, Optional

#################
#   Functions   #
#################
def GetDataframe(data_dir: str = None, mode: str = "train") -> pd.DataFrame:
    # define variables and check it
    assert os.path.join(data_dir), ValueError("data_dir doesn't exist.")
    assert mode in ['train', 'val', 'test'], ValueError("mode should be 'train', 'val', or 'test'.")
    file_path = os.path.join(data_dir, f"physionet_2012_{mode}.csv")
    assert os.path.isfile(file_path), LookupError("file doesn't exist.")
    return pd.read_csv(file_path)


def GetDataset(data_dir: str = None, mode: str = "train", do_normalization: bool = True, do_augmentation: bool = False, base_statistic_info_kwargs: dict = {}):
    '''
        the returned list is like
            (mode=='train', do_kfold=Flase): [[training dataset, validation dataset]]
            (mode=='test', do_kfold=Flase): [[testing dataset]]
    '''
    assert mode in ['train', 'test'], ValueError("mode should be train or test")
    dataset_list = []
    if mode == 'train':
        print("data_dir",data_dir)
        training_df = GetDataframe(data_dir=data_dir, mode=mode)
        validation_df = GetDataframe(data_dir=data_dir, mode='val')
        
        print("training_df", training_df.shape)

        base_statistic_info = BaseStatisticInfo(training_df, **base_statistic_info_kwargs)
        training_dataset = PhysioNet2012Dataset(training_df, base_statistic_info, do_normalization=do_normalization)
        validation_dataset = PhysioNet2012Dataset(validation_df, base_statistic_info, do_normalization=do_normalization)
            
        dataset_list.append([training_dataset, validation_dataset])        

    if mode == 'test':
        print("data_dir",data_dir)
        testing_df = GetDataframe(data_dir=data_dir, mode=mode)
        # to get the PID in validation set
        df_base = GetDataframe(data_dir=data_dir, mode='train') 

        base_statistic_info = BaseStatisticInfo(df_base, **base_statistic_info_kwargs)

        test_dataset = PhysioNet2012Dataset(testing_df, base_statistic_info, do_normalization=do_normalization, do_augmentation=do_augmentation)
        dataset_list.append([test_dataset])

    return dataset_list

################
#   Datasets   #
################
class PhysioNet2012ColInfo():
    # numerical value
    NUM_COLS = [
        'Weight', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
        'Urine', 'WBC', 'pH', 'Age', 'Height', 'seg_entry_cnt'
    ]
    NUM_LEN = len(NUM_COLS)
    # categorical value
    CAT_COLS = [
        'Gender', 'ICUType'
    ]
    CAT_LEN = len(CAT_COLS)
    # seq info
    seq_len = 48

class BaseStatisticInfo():
    def __init__(self,
                base_dataframe: pd.DataFrame,
                drop_outlier: bool = False,
                outlier_fence: float = None,
                statistic_plot: bool = False,
                plot_output_dir: str = './statistic_plot'
                ):
        if statistic_plot:
            self.plot(base_dataframe, plot_output_dir)
        self.num_mean, self.num_median, self.num_std, self.cat_mode = self.ComputeStatisticInfo(base_dataframe, drop_outlier, outlier_fence)

    @staticmethod
    def ComputeStatisticInfo(base_data: pd.DataFrame, drop_outlier: bool = False, m: float = None) -> tuple:
        def reject_outlier(arr: np.ndarray, m: float = 3.) -> np.ndarray:
            # REF: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
            # REF: https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
            assert m is not None, ValueError("outlier should be positive number.")
            na_drop_arr = arr[~np.isnan(arr)]
            d = np.abs(na_drop_arr - np.median(na_drop_arr))
            mdev = np.quantile(d, .75) - np.quantile(d, .25)
            s = d/mdev if mdev else np.zeros(len(d))
            return na_drop_arr[s<m]

        # compute statistic information from base data (training set) by entry.
        # initialize np array
        num_mean = np.zeros(PhysioNet2012ColInfo.NUM_LEN)
        num_median = np.zeros(PhysioNet2012ColInfo.NUM_LEN)
        num_std = np.zeros(PhysioNet2012ColInfo.NUM_LEN)
        cat_mode = np.zeros(PhysioNet2012ColInfo.CAT_LEN)
        # compute the statistic info.
        for idx in range(PhysioNet2012ColInfo.NUM_LEN):
            sub_data = base_data[PhysioNet2012ColInfo.NUM_COLS[idx]].values
            if drop_outlier:
                sub_data = reject_outlier(sub_data, m=3.)
            num_mean[idx] = np.nanmean(sub_data)
            num_median[idx] = np.nanmedian(sub_data)
            num_std[idx] = np.nanstd(sub_data)

        for idx in range(PhysioNet2012ColInfo.CAT_LEN):
            cat_mode[idx] = stats.mode(base_data[PhysioNet2012ColInfo.CAT_COLS[idx]].values, nan_policy="omit")[0]

        return (num_mean, num_median, num_std, cat_mode)
            
    def plot(self, base_data: pd.DataFrame, output_dir: str = "./figure", drop_outlier: bool = False, m: float = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        def reject_outlier(arr: np.ndarray, m: float = 3.) -> np.ndarray:
            # REF: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
            # REF: https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
            assert m is not None, "outlier bound should be positive number."
            na_drop_arr = arr[~np.isnan(arr)]
            d = np.abs(na_drop_arr - np.median(na_drop_arr))
            mdev = np.quantile(d, .75) - np.quantile(d, .25)
            s = d/mdev if mdev else np.zeros(len(d))
            return na_drop_arr[s<m]

        for col in PhysioNet2012ColInfo.NUM_COLS:
            sub_na_drop_arr = reject_outlier(base_data[col].values, m) if drop_outlier else base_data[col].values[~np.isnan(base_data[col].values)]
            fig = plt.figure()
            sns.set(style = "whitegrid")
            plt.title(col)
            sns.distplot(sub_na_drop_arr)
            fig.savefig(os.path.join(output_dir, col))

class PhysioNet2012Dataset(Dataset):
    def __init__(self,
                data: pd.DataFrame,
                base_statistic_info: BaseStatisticInfo,
                do_normalization: bool = True,
                do_augmentation: bool = False
                ):
        # define variable and check it
        x = data
        self.normalize_num_mean, self.normalize_num_median, self.normalize_num_std, self.normalize_cat_mode = base_statistic_info.num_mean, base_statistic_info.num_median, base_statistic_info.num_std, base_statistic_info.cat_mode # for normalization

        # seperate numerical and categorical data
        x_num = x[["pid", "timestamp"] + PhysioNet2012ColInfo.NUM_COLS]
        x_cat = x[["pid", "timestamp"] + PhysioNet2012ColInfo.CAT_COLS]

        # replace the categorical value from 0/1 (binary) to -1/+1 (pos/neg)
        x_cat = x_cat.replace(to_replace=0, value=-1)

        # concatenate numerical and categorical data
        x_num = x_num.sort_values(by=["pid", "timestamp"])
        x_cat = x_cat.sort_values(by=["pid", "timestamp"])
        self.x = pd.concat([x_num, x_cat.drop(columns=["pid", "timestamp"])], axis=1)

        # define "len" of the dataset and "pid" list
        self.number_of_patient = self.x['pid'].nunique()
        self.PID_list = self.x['pid'].unique().tolist()

        # define y
        self.y = x[["pid", "target"]]

        # create mask column
        self.CreateMaskCol()

        # impute missing value
        self.impute(drop_outlier=True, m=3.)
        
        # normalization
        if do_normalization:
            self.normalization()
        
        self.do_augmentation = do_augmentation

    def CreateMaskCol(self):
        # create mask
        for col in PhysioNet2012ColInfo.NUM_COLS:
            self.x['masked_' + col] = 1 * ~self.x[col].isna()
        for col in PhysioNet2012ColInfo.CAT_COLS:
            self.x['masked_' + col] = 1 * ~self.x[col].isna()

    def impute(self, drop_outlier: bool = False, m: float = 3.) -> None:
        '''
            This function must be applied after CreateMaskCol.
        '''
        # get mean, median, std, and mode for imputation from this dataset.
        impute_num_mean, impute_num_median, impute_num_std, impute_cat_mode = self.normalize_num_mean, self.normalize_num_median, self.normalize_num_std, self.normalize_cat_mode

        # impute the missing value with mean and mode for numerical and categorical data respectively.
        for col in PhysioNet2012ColInfo.NUM_COLS:
            self.x[col] = self.x[col].fillna(value=impute_num_mean[PhysioNet2012ColInfo.NUM_COLS.index(col)])
        for col in PhysioNet2012ColInfo.CAT_COLS:
            self.x[col] = self.x[col].fillna(value=impute_cat_mode[PhysioNet2012ColInfo.CAT_COLS.index(col)])

    def normalization(self):
        # normalize all numerical value "by_entry".
        for col in PhysioNet2012ColInfo.NUM_COLS:
            self.x[col] = (self.x[col] - self.normalize_num_mean[PhysioNet2012ColInfo.NUM_COLS.index(col)]) / (self.normalize_num_std[PhysioNet2012ColInfo.NUM_COLS.index(col)] + 1e-8)

    def augmentation(self, value_idx_1: np.ndarray, value_1: np.ndarray, value_mask_1: np.ndarray, value_idx_2: np.ndarray, value_2: np.ndarray, value_mask_2: np.ndarray):
        '''
            Apply random shuffle for EHR data.
        '''
        # REF: https://stackoverflow.com/questions/26194389/how-to-rearrange-array-based-upon-index-array
        shuffled_timestamp = np.arange(value_1.shape[0])
        np.random.shuffle(shuffled_timestamp)
        return value_idx_1, value_1[shuffled_timestamp], value_mask_1[shuffled_timestamp], value_idx_2, value_2[shuffled_timestamp], value_mask_2[shuffled_timestamp]
    def __len__(self):
        return self.number_of_patient

    def __getitem__(self, idx):
        # get (x_num, x_num_mask, x_cat, x_cat_mask, y) pair whose 'pid' are identical.
        # change x to 3D array (this should be imputed data)
        pid = self.PID_list[idx]
        x_num = self.x[self.x['pid'] == pid][PhysioNet2012ColInfo.NUM_COLS].values
        x_num_mask = self.x[self.x['pid'] == pid][['masked_' + col for col in PhysioNet2012ColInfo.NUM_COLS]].values
        x_num_idx = np.arange(1, PhysioNet2012ColInfo.NUM_LEN+1)
        x_cat = self.x[self.x['pid'] == pid][PhysioNet2012ColInfo.CAT_COLS].values
        x_cat_mask = self.x[self.x['pid'] == pid][['masked_' + col for col in PhysioNet2012ColInfo.CAT_COLS]].values
        x_cat_idx = np.arange(1, PhysioNet2012ColInfo.CAT_LEN+1)

        # tile the idx_vector
        x_num_idx = np.tile(x_num_idx, (x_num.shape[0], 1))
        x_cat_idx = np.tile(x_cat_idx, (x_cat.shape[0], 1))

        if self.do_augmentation:
            x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask = self.augmentation(x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask)

        # y (target)
        y = self.y[self.y['pid'] == pid]['target'].values[0]
        y = np.array([y])

        # convert all numpy array to tensor
        x_num_idx = torch.from_numpy(x_num_idx).long()
        x_num = torch.from_numpy(x_num).float()
        x_num_mask = torch.from_numpy(x_num_mask).long()
        x_cat_idx = torch.from_numpy(x_cat_idx).long()
        x_cat = torch.from_numpy(x_cat).float()
        x_cat_mask = torch.from_numpy(x_cat_mask).long()

        # for ablation study
        #x_num_mask = torch.ones_like(x_num_mask)
        #x_cat_mask = torch.ones_like(x_cat_mask)
        
        # return preprocessed (x, y) pair
        return (x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask, y)

if __name__ == '__main__':
    path = './data/physionet_2012'
    tmp_train_validation = GetDataset(data_dir=path, mode='train', do_normalization=True)
    train_dataset = tmp_train_validation[0][0]
    val_dataset = tmp_train_validation[0][1]

    
    ttmp = DataLoader(train_dataset, batch_size=2, shuffle=False)
    for step, (x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask, y) in enumerate(ttmp):
        breakpoint()

