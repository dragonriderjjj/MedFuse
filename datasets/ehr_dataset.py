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
from typing import Tuple, List

#################
#   Functions   #
#################
def GetDataframe(data_dir: str = None, mode: str = "train") -> Tuple[pd.DataFrame, pd.DataFrame]:
    # define variable and check it.
    assert os.path.isdir(data_dir), LookupError("data_dir doesn't exist.")
    x_path = os.path.join(data_dir, mode, "x_s.csv") if mode not in data_dir else os.path.join(data_dir, "x_s.csv")
    y_path = os.path.join(data_dir, mode, "y.csv") if mode not in data_dir else os.path.join(data_dir, "y.csv")
    assert os.path.isfile(x_path), LookupError("\"x_s.csv\" doesn't exist.")
    assert os.path.isfile(y_path), LookupError("\"y.csv\" doesn't exist.")
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    return (x, y)

def GetKFoldPID(data: pd.DataFrame, n_splits: int = 5, seed: int = 123) -> list:
    """
        params:
            data: this dataframe should contain "pid" and "group". we put the dataframe read from "y.csv" here.
            n_splits: the number of split.
            seed: random seed.
        output:
            list of training and testing set in each splits.
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x, y = data['pid'].values, data['group'].values
    idx_list = list(kfold.split(x, y))
    pid_list = []
    for i in range(n_splits):
        pid_list.append((data.loc[idx_list[i][0]]['pid'].values, data.loc[idx_list[i][1]]['pid'].values))
    return pid_list

def GetValidationPID(data: pd.DataFrame, validation_ratio: float = .2, seed: int = 0) -> tuple:
    """
        params:
            data: this dataframe should contain "pid" and "group". we put the dataframe read from "y.csv" here.
            validation_ratio: the ratio for valiadation data from the training set.
            seed : random seed.
        output:
            list of training PID and validation PID.
    """
    x, y = data['pid'].values, data['group'].values
    x_training, x_validation, y_training, y_validation = train_test_split(x, y, test_size=validation_ratio, random_state=seed, shuffle=True, stratify=y)
    return x_training, x_validation

def GetDataset(data_dir: str = None, mode: str = "train", do_kfold: bool = True, k: int = None, do_augmentation: bool = False, do_normalization: bool = True, base_statistic_info_kwargs: dict = {}, for_pretraining: bool = False):
    '''
        the returned list is like
            (mode=='train', do_kfold=Flase): [[training dataset, validation dataset]]
            (mode=='train', do_kfold=True): [[training dataset 1, validation dataset 1], ..., [training dataset k, validation dataset k]]
            (mode=='test', do_kfold=Flase): [[testing dataset]]
    '''
    assert mode in ['train', 'test'], ValueError('mode error.')
    dataset_list = []
    if mode == 'train':
        df_x, df_y = GetDataframe(data_dir=data_dir, mode=mode)
        if do_kfold:
            assert k is not None, ValueError('k shuold be positive integer.')
            kfold_pid_list = GetKFoldPID(df_y, n_splits=k)

            for k_idx in range(len(kfold_pid_list)):
                training_df_x = df_x[df_x['pid'].isin(kfold_pid_list[k_idx][0])].copy(deep=True)
                training_df_y = df_y[df_y['pid'].isin(kfold_pid_list[k_idx][0])].copy(deep=True)
                validation_df_x = df_x[df_x['pid'].isin(kfold_pid_list[k_idx][1])].copy(deep=True)
                validation_df_y = df_y[df_y['pid'].isin(kfold_pid_list[k_idx][1])].copy(deep=True)

                base_statistic_info = BaseStatisticInfo(training_df_x, **base_statistic_info_kwargs)
                
                if for_pretraining:
                    training_dataset = PretrainingEHRDatasets((training_df_x, training_df_y), base_statistic_info, do_augmentation=do_augmentation, do_normalization=do_normalization)
                    validation_dataset = PretrainingEHRDatasets((validation_df_x, validation_df_y), base_statistic_info, do_augmentation=False, do_normalization=do_normalization)
                    
                else:
                    training_dataset = EHRDatasets((training_df_x, training_df_y), base_statistic_info, do_augmentation=do_augmentation, do_normalization=do_normalization)
                    validation_dataset = EHRDatasets((validation_df_x, validation_df_y), base_statistic_info, do_augmentation=False, do_normalization=do_normalization)
                
                dataset_list.append([training_dataset, validation_dataset])

        else:
            training_validation_pid = GetValidationPID(df_y, validation_ratio=.2)

            training_df_x = df_x[df_x['pid'].isin(training_validation_pid[0])].copy(deep=True)
            training_df_y = df_y[df_y['pid'].isin(training_validation_pid[0])].copy(deep=True)
            validation_df_x = df_x[df_x['pid'].isin(training_validation_pid[1])].copy(deep=True)
            validation_df_y = df_y[df_y['pid'].isin(training_validation_pid[1])].copy(deep=True)

            base_statistic_info = BaseStatisticInfo(training_df_x, **base_statistic_info_kwargs)
            if for_pretraining:
                training_dataset = PretrainingEHRDatasets((training_df_x, training_df_y), base_statistic_info, do_augmentation=do_augmentation, do_normalization=do_normalization)
                validation_dataset = PretrainingEHRDatasets((validation_df_x, validation_df_y), base_statistic_info, do_augmentation=False, do_normalization=do_normalization)
            else:
                training_dataset = EHRDatasets((training_df_x, training_df_y), base_statistic_info, do_augmentation=do_augmentation, do_normalization=do_normalization)
                validation_dataset = EHRDatasets((validation_df_x, validation_df_y), base_statistic_info, do_augmentation=False, do_normalization=do_normalization)
            
            dataset_list.append([training_dataset, validation_dataset])        

    if mode == 'test':
        testing_df_x, testing_df_y = GetDataframe(data_dir=data_dir, mode=mode)
        # to get the PID in validation set
        df_x_base, df_y_base = GetDataframe(data_dir=data_dir, mode='train') 

        training_validation_pid = GetValidationPID(df_y_base, validation_ratio=.2)

        df_x_base = df_x_base[df_x_base['pid'].isin(training_validation_pid[0])].copy(deep=True)
        base_statistic_info = BaseStatisticInfo(df_x_base, **base_statistic_info_kwargs)

        test_dataset = EHRDatasets((testing_df_x, testing_df_y), base_statistic_info, do_augmentation=do_augmentation, do_normalization=do_normalization)
        dataset_list.append([test_dataset])

    return dataset_list

################
#   Datasets   #
################
class EHRColInfo():
    # numerical value
    NUM_COLS = ["AFP", "ALB", "ALP", "ALT", "AST", "BUN", "CRE", "D_BIL", "GGT", "GlucoseAC", "HB",
                "HBVDNA", "HCVRNA", "HbA1c", "Lym", "Na", "PLT", "PT", "PT_INR", "Seg", "T_BIL", "TP", "WBC"
               ]
    NUM_COLS = ["final_" + col for col in NUM_COLS]
    NUM_COLS += ["HEIGHT", "WEIGHT", "fatty_liver", "parenchymal_liver_disease", "age", "hosp_days", "seg_entry_cnt"]
    NUM_LEN = len(NUM_COLS)
    # categorical value
    CAT_COLS = ["Anti_HBc", "Anti_HBe", "Anti_HBs", "Anti_HCV", "HBeAg", "HBsAg"]
    CAT_COLS = ["final_" + col for col in CAT_COLS]
    CAT_COLS += ["sex", "sono"]
    CAT_LEN = len(CAT_COLS)
    # seq info
    seq_len = 4

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
            assert m is not None, ValueError("outlier bound should be positive number.")
            na_drop_arr = arr[~np.isnan(arr)]
            d = np.abs(na_drop_arr - np.median(na_drop_arr))
            mdev = np.quantile(d, .75) - np.quantile(d, .25)
            s = d/mdev if mdev else np.zeros(len(d))
            return na_drop_arr[s<m]

        # compute statistic information from base data (training set) by entry.
        # initialize np array
        num_mean = np.zeros(EHRColInfo.NUM_LEN)
        num_median = np.zeros(EHRColInfo.NUM_LEN)
        num_std = np.zeros(EHRColInfo.NUM_LEN)
        cat_mode = np.zeros(EHRColInfo.CAT_LEN)
        # compute the statistic info.
        for idx in range(EHRColInfo.NUM_LEN):
            sub_data = base_data[EHRColInfo.NUM_COLS[idx]].values
            if drop_outlier:
                try:
                    sub_data = reject_outlier(sub_data, m=m)
                except:
                    sub_data = np.zeros_like(sub_data)
            num_mean[idx] = np.nanmean(sub_data)
            num_median[idx] = np.nanmedian(sub_data)
            num_std[idx] = np.nanstd(sub_data)

        for idx in range(EHRColInfo.CAT_LEN):
            cat_mode[idx] = stats.mode(base_data[EHRColInfo.CAT_COLS[idx]].values, nan_policy="omit")[0][0]

        return (num_mean, num_median, num_std, cat_mode)

    def plot(self, base_data: pd.DataFrame, output_dir: str = "./figure", drop_outlier: bool = False, m: float = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        def reject_outlier(arr: np.ndarray, m: float = 3.) -> np.ndarray:
            # REF: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
            # REF: https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
            assert m is not None, ValueError("outlier bound should be positive number.")
            na_drop_arr = arr[~np.isnan(arr)]
            d = np.abs(na_drop_arr - np.median(na_drop_arr))
            mdev = np.quantile(d, .75) - np.quantile(d, .25)
            s = d/mdev if mdev else np.zeros(len(d))
            return na_drop_arr[s<m]

        for col in EHRColInfo.NUM_COLS:
            sub_na_drop_arr = reject_outlier(base_data[col].values, m) if drop_outlier else base_data[col].values[~np.isnan(base_data[col].values)]
            fig = plt.figure()
            sns.set(style = "whitegrid")
            plt.title(col)
            sns.distplot(sub_na_drop_arr)
            fig.savefig(os.path.join(output_dir, col))

class EHRDatasets(Dataset):
    def __init__(self,
                data: Tuple[pd.DataFrame],
                base_statistic_info: BaseStatisticInfo,
                do_augmentation: bool = True,
                do_normalization: bool = True,
                ):
        # define variable and check it.
        x = data[0]
        y = data[1]
        self.normalize_num_mean, self.normalize_num_median, self.normalize_num_std, self.normalize_cat_mode = base_statistic_info.num_mean, base_statistic_info.num_median, base_statistic_info.num_std, base_statistic_info.cat_mode # for normalization
        self.do_augmentation = do_augmentation

        # seperate numerical and categorical data
        x_num = x[["pid", "date"] + EHRColInfo.NUM_COLS]
        x_cat = x[["pid", "date"] + EHRColInfo.CAT_COLS]

        # replace the categorical value from 0/1 (binary) to -1/+1 (pos/neg)
        x_cat = x_cat.replace(to_replace=0, value=-1)

        # concatenate numerical and categorical data
        x_num = x_num.sort_values(by=["pid", "date"])
        x_cat = x_cat.sort_values(by=["pid", "date"])
        self.x = pd.concat([x_num, x_cat.drop(columns=["pid", "date"])], axis=1)

        # define "len" of the dataset and "pid" list
        self.number_of_patient = self.x['pid'].nunique()
        self.PID_list = self.x['pid'].unique().tolist()

        # define y
        self.y = y

        # change "date" to pd.datetime
        self.x['date'] = pd.to_datetime(self.x['date'])
        self.y['date'] = pd.to_datetime(self.y['date'])
        self.y['index_date'] = pd.to_datetime(self.y['index_date'])

        # create mask column
        self.CreateMaskCol()

        # impute missing value
        self.impute(drop_outlier=True, m=3.)
        
        # normalization
        if do_normalization:
            self.normalization()

    def CreateMaskCol(self):
        # create mask
        for col in EHRColInfo.NUM_COLS:
            self.x['masked_'+col] = 1 * ~self.x[col].isna()
        for col in EHRColInfo.CAT_COLS:
            self.x['masked_'+col] = 1 * ~self.x[col].isna()

    def impute(self, drop_outlier: bool = False, m: float = 3.) -> None:
        '''
            This function must be applied after CreateMaskCol.
        '''
        # get mean, median, std, and mode for imputation from this dataset.
        impute_num_mean, impute_num_median, impute_num_std, impute_cat_mode = self.normalize_num_mean, self.normalize_num_median, self.normalize_num_std, self.normalize_cat_mode

        # impute the missing value with mean and mode for numerical and categorical data respectively.
        for col in EHRColInfo.NUM_COLS:
            self.x[col] = self.x[col].fillna(value=impute_num_mean[EHRColInfo.NUM_COLS.index(col)])
        for col in EHRColInfo.CAT_COLS:
            self.x[col] = self.x[col].fillna(value=impute_cat_mode[EHRColInfo.CAT_COLS.index(col)])

    def normalization(self):
        # normalize all numerical value "by_entry".
        for col in EHRColInfo.NUM_COLS:
            self.x[col] = (self.x[col] - self.normalize_num_mean[EHRColInfo.NUM_COLS.index(col)]) / (self.normalize_num_std[EHRColInfo.NUM_COLS.index(col)] + 1e-8)

    def augmentation(self, value_idx_1: np.ndarray, value_1: np.ndarray, value_mask_1: np.ndarray, value_idx_2: np.ndarray, value_2: np.ndarray, value_mask_2: np.ndarray):
        '''
            Apply random shuffle for EHR data.
        '''
        # REF: https://stackoverflow.com/questions/26194389/how-to-rearrange-array-based-upon-index-array
        shuffled_timestamp = np.arange(value_1.shape[0])
        np.random.shuffle(shuffled_timestamp)
        return value_idx_1, value_1[shuffled_timestamp], value_mask_1[shuffled_timestamp], value_idx_2, value_2[shuffled_timestamp], value_mask_2[shuffled_timestamp]

        """
        np.random.shuffle(value_idx)
        for row in range(value.shape[0]):
            value[row] = value[row][value_idx - 1]
        for row in range(value_mask.shape[0]):
            value_mask[row] = value_mask[row][value_idx - 1]
        return value_idx, value, value_mask
        """

    def __len__(self):
        return self.number_of_patient

    def __getitem__(self, idx):
        # get (x_num, x_num_mask, x_cat, x_cat_mask, y) pair whose 'pid' are identical.
        # change x to 3D array (this should be imputed data)
        pid = self.PID_list[idx]
        x_num = self.x[self.x['pid'] == pid][EHRColInfo.NUM_COLS].values
        x_num_mask = self.x[self.x['pid'] == pid][['masked_' + col for col in EHRColInfo.NUM_COLS]].values
        x_num_idx = np.arange(1, EHRColInfo.NUM_LEN+1)
        x_cat = self.x[self.x['pid'] == pid][EHRColInfo.CAT_COLS].values
        x_cat_mask = self.x[self.x['pid'] == pid][['masked_' + col for col in EHRColInfo.CAT_COLS]].values
        x_cat_idx = np.arange(1, EHRColInfo.CAT_LEN+1)

        y = self.y[self.y['pid'] == pid]['group'].values
        day_delta = (self.y[self.y['pid'] == pid]['index_date'] - self.y[self.y['pid'] == pid]['date']).values.astype('timedelta64[D]') / np.timedelta64(1, 'D')

        # augmentation (if needed)
        if self.do_augmentation:
            x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask = self.augmentation(x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask)
            #x_cat_idx, x_cat, x_cat_mask = self.augmentation(x_cat_idx, x_cat, x_cat_mask)
        
        # tile the idx_vector
        x_num_idx = np.tile(x_num_idx, (x_num.shape[0], 1))
        x_cat_idx = np.tile(x_cat_idx, (x_cat.shape[0], 1))

        # convert all numpy array to tensor
        x_num_idx = torch.from_numpy(x_num_idx).long()
        x_num = torch.from_numpy(x_num).float()
        x_num_mask = torch.from_numpy(x_num_mask).long()
        x_cat_idx = torch.from_numpy(x_cat_idx).long()
        x_cat = torch.from_numpy(x_cat).float()
        x_cat_mask = torch.from_numpy(x_cat_mask).long()
        
        # return preprocessed (x, y) pair
        return (x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask, y, day_delta)

class PretrainingEHRDatasets(Dataset):
    def __init__(self,
                data: Tuple[pd.DataFrame],
                base_statistic_info: BaseStatisticInfo,
                do_augmentation: bool = True,
                do_normalization: bool = True,
                ):
        # define variable and check it
        x = data[0]
        self.normalize_num_mean, self.normalize_num_median, self.normalize_num_std, self.normalize_cat_mode = base_statistic_info.num_mean, base_statistic_info.num_median, base_statistic_info.num_std, base_statistic_info.cat_mode # for normalization
        self.do_augmentation = do_augmentation

        # seperate numerical and categorical data
        x_num = x[["pid", "date"] + EHRColInfo.NUM_COLS]
        x_cat = x[["pid", "date"] + EHRColInfo.CAT_COLS]

        # replace the categorical value from 0/1 (binary) to -1/+1 (pos/neg)
        x_cat = x_cat.replace(to_replace=0, value=-1)

        # concatenate numerical and categorical data
        x_num = x_num.sort_values(by=["pid", "date"])
        x_cat = x_cat.sort_values(by=["pid", "date"])
        self.x = pd.concat([x_num, x_cat.drop(columns=["pid", "date"])], axis=1)

        # define "len" of the dataset and "pid" list
        self.number_of_patient = self.x['pid'].nunique()
        self.PID_list = self.x['pid'].unique().tolist()
        
        # change "date" to pd.datetime
        self.x['date'] = pd.to_datetime(self.x['date'])

        # create mask column
        self.CreateMaskCol()

        # impute missing value
        self.impute(drop_outlier=True, m=3.)
        
        # normalization
        if do_normalization:
            self.normalization()

    def CreateMaskCol(self):
        # create mask
        for col in EHRColInfo.NUM_COLS:
            self.x['masked_'+col] = 1 * ~self.x[col].isna()
        for col in EHRColInfo.CAT_COLS:
            self.x['masked_'+col] = 1 * ~self.x[col].isna()

    def impute(self, drop_outlier: bool = False, m: float = 3.) -> None:
        '''
            This function must be applied after CreateMaskCol.
        '''
        # get mean, median, std, and mode for imputation from this dataset.
        impute_num_mean, impute_num_median, impute_num_std, impute_cat_mode = BaseStatisticInfo.ComputeStatisticInfo(self.x, drop_outlier, m)

        # impute the missing value with mean and mode for numerical and categorical data respectively.
        for col in EHRColInfo.NUM_COLS:
            self.x[col] = self.x[col].fillna(value=impute_num_mean[EHRColInfo.NUM_COLS.index(col)])
        for col in EHRColInfo.CAT_COLS:
            self.x[col] = self.x[col].fillna(value=impute_cat_mode[EHRColInfo.CAT_COLS.index(col)])

    def normalization(self):
        # normalize all numerical value "by_entry".
        for col in EHRColInfo.NUM_COLS:
            self.x[col] = (self.x[col] - self.normalize_num_mean[EHRColInfo.NUM_COLS.index(col)]) / (self.normalize_num_std[EHRColInfo.NUM_COLS.index(col)] + 1e-8)

    def augmentation(self, value_idx: np.ndarray, value: np.ndarray, value_mask: np.ndarray):
        '''
            Apply random shuffle for EHR data.
        '''
        # REF: https://stackoverflow.com/questions/26194389/how-to-rearrange-array-based-upon-index-array
        np.random.shuffle(value_idx)
        for row in range(value.shape[0]):
            value[row] = value[row][value_idx - 1]
        for row in range(value_mask.shape[0]):
            value_mask[row] = value_mask[row][value_idx - 1]
        return value_idx, value, value_mask

    def __len__(self):
        return self.number_of_patient

    def __getitem__(self, idx):
        # get (x_num, x_num_mask, x_cat, x_cat_mask, y) pair whose 'pid' are identical.
        # change x to 3D array (this should be imputed data)
        pid = self.PID_list[idx]
        x_num = self.x[self.x['pid'] == pid][EHRColInfo.NUM_COLS].values
        x_num_mask = self.x[self.x['pid'] == pid][['masked_' + col for col in EHRColInfo.NUM_COLS]].values
        x_num_idx = np.arange(1, EHRColInfo.NUM_LEN+1)
        x_cat = self.x[self.x['pid'] == pid][EHRColInfo.CAT_COLS].values
        x_cat_mask = self.x[self.x['pid'] == pid][['masked_' + col for col in EHRColInfo.CAT_COLS]].values
        x_cat_idx = np.arange(1, EHRColInfo.CAT_LEN+1)

        # augmentation (if needed)
        if self.do_augmentation:
            x_num_idx, x_num, x_num_mask = self.augmentation(x_num_idx, x_num, x_num_mask)
            x_cat_idx, x_cat, x_cat_mask = self.augmentation(x_cat_idx, x_cat, x_cat_mask)
        
        # tile the idx_vector
        x_num_idx = np.tile(x_num_idx, (x_num.shape[0], 1))
        x_cat_idx = np.tile(x_cat_idx, (x_cat.shape[0], 1))

        # convert all numpy array to tensor
        x_num_idx = torch.from_numpy(x_num_idx).long()
        x_num = torch.from_numpy(x_num).float()
        x_num_mask = torch.from_numpy(x_num_mask).long()
        x_cat_idx = torch.from_numpy(x_cat_idx).long()
        x_cat = torch.from_numpy(x_cat).float()
        x_cat_mask = torch.from_numpy(x_cat_mask).long()
        
        # return preprocessed (x, y) pair
        return (x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask)







if __name__ == "__main__":
    path = '/home/user/r10942147/research/NTUH/peiying_code_備份/NTUH/PeiYing_codes/src/data/summarization_corrected_v2/0/20201119data_hw_us_hosp_emerg-summarize_cohort_first_diagnosis_5_year_v1.1.2/train/x_s.csv'
    x = pd.read_csv(path)
    path = '/home/user/r10942147/research/NTUH/peiying_code_備份/NTUH/PeiYing_codes/src/data/summarization_corrected_v2/0/20201119data_hw_us_hosp_emerg-summarize_cohort_first_diagnosis_5_year_v1.1.2/train/y.csv'
    y = pd.read_csv(path)
    #tmp = BaseStatisticInfo(data, statistic_plot=True, plot_output_dir='statistic_plot')
    #tmp = EHRDatasets((x, y), BaseStatisticInfo(x, True, 3.), 'train', True, True)
    #ttmp = DataLoader(tmp, batch_size=2, shuffle=False)
    path = '/home/user/r10942147/research/NTUH/peiying_code_備份/NTUH/PeiYing_codes/src/data/summarization_corrected_v2/0/20201119data_hw_us_hosp_emerg-summarize_cohort_first_diagnosis_5_year_v1.1.2/'
    tmp_train_kfold = GetDataset(data_dir=path, mode='train', do_kfold=True, k=5)
    tmp_train_validation = GetDataset(data_dir=path, mode='train', do_kfold=False)
    tmp_test = GetDataset(data_dir=path, mode='test')
    breakpoint()

    for step, (x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask, y) in enumerate(ttmp):
        breakpoint()
    print(tmp.num_mean)
    print(tmp.num_median)
    print(tmp.num_std)
    print(tmp.cat_mode)


