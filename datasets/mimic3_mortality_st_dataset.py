import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Optional
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils.util  import BaseStatisticInfo


def GetData(data_dir: str = None, mode: str = "train") -> pd.DataFrame:
    """
    Load MIMIC-III mortality dataset in CSV format for a specified mode.

    Parameters:
    - data_dir (str): path to the directory containing CSV files
    - mode (str): one of ['train', 'val', 'test'], determines which file to load

    Returns:
    - pd.DataFrame: loaded data as a pandas DataFrame
    """
    assert os.path.isdir(data_dir), ValueError(f"[Error] Data directory does not exist: {data_dir}")
    assert mode in ['train', 'val', 'test'], ValueError("mode should be 'train', 'val', or 'test'.")

    file_path = os.path.join(data_dir, f"mimic3_mortality_{mode}.csv")
    assert os.path.isfile(file_path), FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path)


def GetEHRDataset(
        data_dir: str=None, 
        mode: str="train", 
        # do_kfold: bool=True,
        # k: int=None,
        # for_pretraining: bool=False,
        base_statistic_info_kwargs: dict={},
        padding: str='None',
        do_normalization: bool=True, 
        do_augmentation: bool=False, 
        # len_sampling_bound: list=[0.3, 0.7],
        # dense_sampling_bound: list=[0.4, 0.6],
        # mask_ratio_per_seg: float=0.15,
        # segment_num: int=1,
        seq_len: int=0
        ) -> Tuple[list, int]:
    """
    Construct EHR datasets for train/validation/test based on provided configurations.

    Parameters:
    - data_dir (str): Directory where the data CSV files are stored.
    - mode (str): One of 'train' or 'test'.
    - base_statistic_info_kwargs (dict): Keyword args passed to BaseStatisticInfo constructor.
    - padding (str): Padding method.
    - do_normalization (bool): Apply normalization or not.
    - do_augmentation (bool): Apply data augmentation.
    - seq_len (int): Manually specified sequence length for test mode.

    Returns:
    - dataset_list (list): A list of datasets (train/val or test only).
    - seq_len (int): The determined or passed-in sequence length.
    """
    assert mode in ['train', 'test'], ValueError("mode should be train or test, but get {mode}.")

    dataset_list = []
    training_df = GetData(data_dir=data_dir, mode='train')
    if 'seg_entry_cnt' not in training_df.columns:
        MIMIC3MortalityAttri.NUM_COLS.remove('seg_entry_cnt')
        MIMIC3MortalityAttri.NUM_LEN = len(MIMIC3MortalityAttri.NUM_COLS)

    base_stat_info = BaseStatisticInfo(training_df, MIMIC3MortalityAttri, **base_statistic_info_kwargs)

    if mode == 'train':
        validation_df = GetData(data_dir=data_dir, mode='val')

        training_dataset = MIMIC3MortalityDataset(training_df, base_stat_info, padding, do_normalization=do_normalization)
        validation_dataset = MIMIC3MortalityDataset(validation_df, base_stat_info, padding, do_normalization=do_normalization)
            
        MIMIC3MortalityAttri.seq_len = training_dataset.seq_len
        dataset_list.append([training_dataset, validation_dataset])        

    if mode == 'test':
        testing_df = GetData(data_dir=data_dir, mode=mode)

        test_dataset = MIMIC3MortalityDataset(testing_df, base_stat_info, padding, do_normalization=do_normalization, do_augmentation=do_augmentation)
        dataset_list.append([test_dataset])
        MIMIC3MortalityAttri.seq_len = seq_len

    return dataset_list


class MIMIC3MortalityAttri():
    # numerical value
    NUM_COLS = [
        'HR', 'Age', 'RR', 'DBP', 'MBP', 'SBP', 'O2 Saturation', 'Temperature',
        'Weight', 'CRR', 'Base Excess',
        'Calcium Free', 'Lactate', 'PCO2', 'PO2', 'Potassium', 'Total CO2', 'pH Blood',
        'Glucose (Blood)', 'Urine', 'Solution', 'Normal Saline', 'FiO2', 'ALP', 'ALT',
        'AST', 'Anion Gap', 'BUN', 'Bicarbonate', 'Bilirubin (Total)', 'Calcium Total',
        'Chloride', 'Creatinine Blood', 'Glucose (Serum)', 'Hct', 'Hgb', 'INR', 'LDH',
        'MCH', 'MCHC', 'MCV', 'Magnesium', 'PT', 'PTT', 'Phosphate', 'Platelet Count',
        'RBC', 'RDW', 'Sodium', 'WBC', 'PO intake', 'Amiodarone', 'D5W', 'Heparin',
        'Famotidine', 'Height', 'Dextrose Other', 'KCl', 'SG Urine', 'pH Urine',
        'Fresh Frozen Plasma', 'Albumin 5%', 'Bilirubin (Direct)',
        'Bilirubin (Indirect)', 'Jackson-Pratt', 'Albumin', 'Neosynephrine',
        'Propofol', 'Unknown', 'EBL', 'OR/PACU Crystalloid', 'Intubated', 'Stool',
        'Gastric', 'Gastric Meds', 'Pre-admission Intake', 'Pre-admission Output',
        'Basophils', 'Eoisinophils', 'Lymphocytes', 'Monocytes', 'Neutrophils',
        'Nitroglycerine', 'Chest Tube', 'Packed RBC', 'Colloid', 'Insulin Regular',
        'Pantoprazole', 'Hydromorphone', 'Emesis', 'Insulin Humalog',
        'Insulin largine', 'Furosemide', 'Lactated Ringers', 'Morphine Sulfate',
        'Glucose (Whole Blood)', 'Calcium Gluconate', 'Metoprolol', 'Norepinephrine',
        'Vasopressin', 'Dopamine', 'Fentanyl', 'Midazolam', 'Creatinine Urine',
        'Piggyback', 'Magnesium Sulfate (Bolus)', 'Magnesium Sulphate',
        'KCl (Bolus)', 'Nitroprusside', 'Lorazepam', 'Piperacillin', 'Fiber',
        'Residual', 'Free Water', 'GT Flush', 'Vacomycin', 'Hydralazine',
        'Half Normal Saline', 'Cefazolin', 'Sterile Water', 'Ultrafiltrate', 'TPN',
        'Albumin 25%', 'Epinephrine', 'Milrinone', 'Insulin NPH',
        'Lymphocytes (Absolute)'
        , 'seg_entry_cnt'
    ]
    # NUM_COLS = ['seg_entry_cnt']
    NUM_LEN = len(NUM_COLS)
    # categorical value
    CAT_COLS = [
        'Gender', 
        'GCS_eye', 'GCS_motor', 'GCS_verbal'
    ]
    CAT_LEN = len(CAT_COLS)
    # seq info
    seq_len = 0

class MIMIC3MortalityDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-III EHR mortality prediction.

    Handles:
    - numerical + categorical column parsing
    - missing data imputation
    - normalization
    - masking (with optional corruption)
    - temporal padding (pre/post)
    - patient-wise slicing and batching

    Returns:
    - tuple: ((x_idx, x, x_mask, x_timestamp, x_records), y)
    """
    def __init__(
        self,
        data: pd.DataFrame,
        base_statistic_info: BaseStatisticInfo,
        padding: str = 'None',
        do_normalization: bool = True,
        do_augmentation: bool = False
    ):
        self.padding = padding
        self.do_augmentation = do_augmentation
        self.do_normalization = do_normalization
        self.num_mean = base_statistic_info.num_mean
        self.num_median = base_statistic_info.num_median
        self.num_std = base_statistic_info.num_std
        self.cat_mode = base_statistic_info.cat_mode

        x = data
        # seperate numerical and categorical data
        x_num = x[["pid", "timestamp"] + MIMIC3MortalityAttri.NUM_COLS].sort_values(by=["pid", "timestamp"])
        x_cat = x[["pid", "timestamp"] + MIMIC3MortalityAttri.CAT_COLS].sort_values(by=["pid", "timestamp"])

        # replace the categorical value from 0/1 (binary) to -1/+1 (pos/neg)
        x_cat = x_cat.replace(to_replace=0, value=-1)

        # concatenate numerical and categorical data
        self.x = pd.concat([x_num, x_cat.drop(columns=["pid", "timestamp"])], axis=1)

        # define "len" of the dataset and "pid" list
        self.number_of_patient = self.x['pid'].nunique()
        self.PID_list = self.x['pid'].unique().tolist()
        self.seq_len = self.x.groupby(by="pid").size().max()
        self.num_record = self.x.groupby(by="pid")['seg_entry_cnt'].sum() if 'seg_entry_cnt' in self.x.columns else self.x.groupby(by="pid").size()

        # define y
        self.y = x[["pid", "target"]]

        # create mask column
        self.CreateMaskCol()

        # impute missing value
        self.impute(drop_outlier=True, m=3.)
        
        # normalization
        if do_normalization:
            self.normalization()


    def CreateMaskCol(self, pattern = None, ratio: float=0.0) -> None:
        cols = MIMIC3MortalityAttri.NUM_COLS + MIMIC3MortalityAttri.CAT_COLS

        # Using a dictionary comprehension with direct assignment to a DataFrame
        mask_df = pd.DataFrame({f'masked_{col}': (1 * ~self.x[col].isna()) for col in cols}, index=self.x.index)

        # Concatenating directly with the selected columns
        self.x_mask = pd.concat([self.x[["pid", "timestamp"]], mask_df], axis=1)

        if pattern is not None:
            updated_df = self.x_mask.copy()
            if pattern == 1:
                print("Using Missing Pattern 1")
                print("[MCAR Missing] Before masking:", updated_df.values[:, 2:].sum())
                observed = np.argwhere(updated_df.values[:, 2:] == 1)
                num_to_add = int(len(observed) * ratio)
                chosen_indices = observed[
                    np.random.choice(len(observed), size=num_to_add, replace=False)
                ]
                for row, col in chosen_indices:
                    updated_df.iat[row, col+2] = 0
                print("[MCAR Missing] After masking:", updated_df.values[:, 2:].sum())
            elif pattern == 2:
                print("Using Missing Pattern 2")
                total_feats = updated_df.shape[1] - 2
                chosen_indices = np.random.choice(total_feats, size=int(total_feats * ratio), replace=False)
                print("[MAR Missing] Features masked:", len(chosen_indices))
                for col in chosen_indices:
                    updated_df.iloc[:, col+2] = 0

            self.x_mask = updated_df

    def impute(self, drop_outlier: bool = False, m: float = 3.) -> None:
        """
        This function must be applied after CreateMaskCol.
        """
        # impute the missing value with mean and mode for numerical and categorical data respectively.
        for col in MIMIC3MortalityAttri.NUM_COLS:
            self.x[col] = self.x[col].fillna(value=self.num_mean[MIMIC3MortalityAttri.NUM_COLS.index(col)])
        for col in MIMIC3MortalityAttri.CAT_COLS:
            self.x[col] = self.x[col].fillna(value=self.cat_mode[MIMIC3MortalityAttri.CAT_COLS.index(col)])

    def normalization(self):
        # normalize all numerical value "by_entry".
        for col in MIMIC3MortalityAttri.NUM_COLS:
            self.x[col] = (self.x[col] - self.num_mean[MIMIC3MortalityAttri.NUM_COLS.index(col)]) / (self.num_std[MIMIC3MortalityAttri.NUM_COLS.index(col)] + 1e-8)
            # address wrong norm of the outlier
            self.x[col] = np.clip(self.x[col], -5., 5.)

    def augmentation(self, value_idx: np.ndarray, value: np.ndarray, value_mask: np.ndarray, timestamp: np.ndarray):
        """
        Apply random shuffle for EHR data.
        """
        # REF: https://stackoverflow.com/questions/26194389/how-to-rearrange-array-based-upon-index-array
        shuffled_timestamp = np.arange(value.shape[0])
        np.random.shuffle(shuffled_timestamp)
        return value_idx[shuffled_timestamp], value[shuffled_timestamp], value_mask[shuffled_timestamp], timestamp[shuffled_timestamp]

    def __len__(self):
        return self.number_of_patient

    def __getitem__(self, idx):
        # get (x_num, x_num_mask, x_cat, x_cat_mask, y) pair whose 'pid' are identical.
        # change x to 3D array (this should be imputed data)
        pid = self.PID_list[idx]
        x = self.x[self.x['pid'] == pid].drop(columns=['pid', 'timestamp']).values
        x_mask = self.x_mask[self.x_mask['pid'] == pid].drop(columns=['pid', 'timestamp']).values
        x_idx = np.arange(1, MIMIC3MortalityAttri.NUM_LEN + MIMIC3MortalityAttri.CAT_LEN + 1)
        x_timestamp = self.x[self.x['pid'] == pid]['timestamp'].values.reshape(-1, 1)
        # x_timestamp = np.array((x_timestamp))
        x_records = self.num_record[self.num_record.index == pid].values

        seq_pad_len = max(MIMIC3MortalityAttri.seq_len - x.shape[0], 0)
        if (seq_pad_len > 0) & (self.padding.lower() in ['prepad', 'postpad']):
            pad_width = ((seq_pad_len, 0), (0, 0)) if self.padding.lower() == 'prepad' else ((0, seq_pad_len), (0, 0))
            x = np.pad(x, pad_width, constant_values=(0., 0.))
            x_mask = np.pad(x_mask, pad_width, constant_values=(0, 0))
            x_timestamp = np.pad(x_timestamp, pad_width, constant_values=(0, 0))
        x = x[-MIMIC3MortalityAttri.seq_len:]
        x_mask = x_mask[-MIMIC3MortalityAttri.seq_len:]
        x_timestamp = x_timestamp[-MIMIC3MortalityAttri.seq_len:]

        # tile the idx_vector
        x_idx = np.tile(x_idx, (x.shape[0], 1))

        if self.do_augmentation:
            x_idx, x, x_mask, x_timestamp = self.augmentation(x_idx, x, x_mask, x_timestamp)

        # y (target)
        y = self.y[self.y['pid'] == pid]['target'].values[0]
        y = np.array([y])

        return (
            (
                torch.from_numpy(x_idx).long(),
                torch.from_numpy(x).bfloat16(),
                torch.from_numpy(x_mask).long(),
                torch.from_numpy(x_timestamp).bfloat16(),
                torch.from_numpy(x_records).long(),
            ),
            y
        )


if __name__ == '__main__':
    path = './data/physionet_2012'
    tmp_train_validation = GetData(data_dir=path, mode='train', do_normalization=True)
    train_dataset = tmp_train_validation[0][0]
    val_dataset = tmp_train_validation[0][1]

    
    ttmp = DataLoader(train_dataset, batch_size=2, shuffle=False)
    for step, (x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask, y) in enumerate(ttmp):
        breakpoint()

