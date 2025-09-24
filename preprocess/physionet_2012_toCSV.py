import tensorflow_datasets as tfds
import medical_ts_datasets
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


pid = ['pid']

timestamp = ['timestamp']

vital = [
    'Weight', 'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
    'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
    'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
    'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
    'Urine', 'WBC', 'pH'
         ]

static = [
    'Age', 'Gender', 'Height', 'ICUType'
         ]

target = ['target']

whole_column = pid + timestamp + vital + static + target

def ICUType(static_array: np.ndarray) -> int:
    if static_array[4] == 1: return 1
    elif static_array[5] == 1: return 2
    elif static_array[6] == 1: return 3
    else: return 4

def collect_data(mode: str = "train", folder: str = "", counter: int = 0):
    dataset = tfds.load(name='physionet2012', split=mode)
    if mode == "validation": mode = "val"
    df = pd.DataFrame(columns = whole_column)

    counter = counter
    pbar = tqdm(dataset)
    for data in pbar:
        pbar.set_description('collect {:s} data'.format(mode))
        timestamp_value = data['combined'][1].numpy()
        static_value = data['combined'][0].numpy()
        vital_value = data['combined'][2].numpy()
        target_value = data['target'].numpy()
        sub_df = pd.DataFrame(vital_value, columns = vital)
        sub_pid = "PS" + "0" * (8 - len(str(counter))) + str(counter)
        sub_df['pid'] = [sub_pid for i in range(vital_value.shape[0])]
        sub_df['timestamp'] = timestamp_value
        sub_df['target'] = [target_value for i in range(vital_value.shape[0])]
        sub_df['Age'] = [static_value[0] for i in range(vital_value.shape[0])]
        sub_df['Gender'] = [static_value[2] for i in range(vital_value.shape[0])]
        sub_df['Height'] = [static_value[3] for i in range(vital_value.shape[0])]
        sub_df['ICUType'] = [ICUType(static_value) for i in range(vital_value.shape[0])]
        df = pd.concat([df, sub_df[whole_column]], axis=0, ignore_index=True)
        counter += 1
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, "physionet_2012_{:s}.csv".format(mode)), index=False)

if __name__ == "__main__":
    collect_data("train", "../data/P12", counter=0)
    collect_data("validation", "../data/P12", counter=7671)
    collect_data("test", "../data/P12", counter=9588)
