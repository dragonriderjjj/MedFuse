import pandas as pd
import numpy as np
import scipy.stats as scs

from enum import IntEnum
from tqdm import tqdm

# column information
pid = ['pid']
timestamp = ['timestamp']
numerical = [
        'Weight', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
        'Urine', 'WBC', 'pH', 'Age', 'Height'
        ]
categorical = [
        'Gender', 'ICUType'
         ]
target = ['target']

whole_column = pid + timestamp + numerical + categorical + target

vital = [
        'Weight', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
        'Urine', 'WBC', 'pH'
        ]

static = ['Age', 'Height', 'Gender', 'ICUType']

class SUMMARIZATION(IntEnum):
    NONE = 0
    SUM = 1
    MEAN = 2
    MODE = 3

def summarization(
    entry_df: pd.DataFrame,
    method_dict: dict,
    summarization_window_size: int = None,
    feature_window_size: int = 48,
    mode: str = "training",
    ) -> pd.DataFrame:

    def summarize(value, method=SUMMARIZATION.NONE):
        if np.isnan(value).all():
            return np.nan
        sval = np.nan
        if method.value == SUMMARIZATION.SUM.value:
            sval = np.sum(value)
        if method.value == SUMMARIZATION.MEAN.value:
            sval = np.nanmean(value)
        if method.value == SUMMARIZATION.MODE.value:
            sval = scs.mode(value, nan_policy='omit')
        return sval

    entry_df = entry_df.sort_values(['pid', 'timestamp'], axis=0, kind='quicksort').reset_index(drop=True)

    pids, cnt = np.unique(entry_df['pid'].values, return_counts=True)
    cum_cnt = np.insert(np.cumsum(cnt), 0, 0)

    res = []
    with tqdm(total=len(pids)) as pbar:
        pbar.set_description("[summarize {:s} data]".format(mode))
        for idx, (hid, tid) in enumerate(zip(cum_cnt[:-1], cum_cnt[1:])):
            pbar.update(1)
            patient_df = entry_df[hid: tid]

            first_timestamp=0
            for k in range(int(feature_window_size / summarization_window_size)):
                head_timestamp = first_timestamp + summarization_window_size * k
                if k == (int(feature_window_size / summarization_window_size) - 1):
                    tail_timestamp = first_timestamp + feature_window_size
                else:
                    tail_timestamp = first_timestamp + summarization_window_size * (k + 1)

                seg_df = patient_df[(patient_df['timestamp'] < tail_timestamp) & (patient_df['timestamp'] >= head_timestamp)]
                sub_summarization_row = pd.DataFrame(data=None, index = range(1), columns = whole_column)

                if not seg_df[vital].empty:
                    for col in method_dict:
                        method = method_dict[col]
                        val = summarize(seg_df[col].values, method = method)
                        sub_summarization_row[col] = val
                    sub_summarization_row['seg_entry_cnt'] = int(seg_df.shape[0])

                for col in ['Age', 'Height', 'Gender', 'ICUType']:
                    if k == 0:
                        try:
                            sub_summarization_row[col] = patient_df.reset_index(inplace=False)[col][0]
                        except:
                            continue
                sub_summarization_row['pid'] = patient_df.reset_index(inplace=False)['pid'][0]
                sub_summarization_row['target'] = patient_df.reset_index(inplace=False)['target'][0]
                sub_summarization_row['timestamp'] = tail_timestamp

                res.append(sub_summarization_row)
        res = pd.concat(res, ignore_index=True, axis=0).reset_index(drop=True)
    return res

# define method dict
method_dict = {}

for col in vital:
    method_dict[col] = SUMMARIZATION.MEAN

if __name__ == "__main__":
    PATH = './data/P12/physionet_2012_train.csv'
    df = pd.read_csv(PATH)
    # replace missing height to nan
    tmp_h = df['Height'].values
    tmp_h[tmp_h < 0] = np.nan
    df['Height'] = tmp_h
    # summarization
    df = summarization(df, method_dict, summarization_window_size=2, feature_window_size=48, mode="training")
    df.to_csv(PATH, index=False)

    PATH = './data/P12/physionet_2012_val.csv'
    df = pd.read_csv(PATH)
    # replace missing height to nan
    tmp_h = df['Height'].values
    tmp_h[tmp_h < 0] = np.nan
    df['Height'] = tmp_h
    # summarization
    df = summarization(df, method_dict, summarization_window_size=2, feature_window_size=48, mode="validation")
    df.to_csv(PATH, index=False)

    PATH = './data/P12/physionet_2012_test.csv'
    df = pd.read_csv(PATH)
    # replace missing height to nan
    tmp_h = df['Height'].values
    tmp_h[tmp_h < 0] = np.nan
    df['Height'] = tmp_h
    # summarization
    df = summarization(df, method_dict, summarization_window_size=2, feature_window_size=48, mode="testing")
    df.to_csv(PATH, index=False)
