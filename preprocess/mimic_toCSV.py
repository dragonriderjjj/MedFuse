import tensorflow_datasets as tfds
import medical_ts_datasets
import numpy as np
import pandas as pd
import os
from tqdm import tqdm



pid = ['pid']

timestamp = ['timestamp']

demographics = ['Height']

vital = [
    'Weight', 'Heart Rate', 'Mean blood pressure',
    'Diastolic blood pressure', 'Systolic blood pressure',
    'Oxygen saturation', 'Respiratory rate'
]

lab_measurement = [
    'Capillary refill rate', 'Glucose', 'pH', 'Temperature'
]

interventions = [
    'Fraction inspired oxygen', 'Glasgow coma scale eye opening',
    'Glasgow coma scale motor response', 'Glasgow coma scale total',
    'Glasgow coma scale verbal response'
]

target = ['target']

value = vital + lab_measurement + interventions

whole_col = pid + timestamp + demographics + value + target

def collect_data(mode: str = "train", folder: str = "", counter: int = 0):
    dataset = tfds.load(name='mimic3_mortality', split=mode)
    if mode == "validation": mode = "val"
    df = pd.DataFrame(columns = whole_col)

    counter = counter

    pbar = tqdm(dataset)
    for data in pbar:
        pbar.set_description('collect {:s} data'.format(mode))
        timestamp_value = data['combined'][1].numpy()
        static_value = data['combined'][0].numpy()
        obs_value = data['combined'][2].numpy()
        target_value = data['target'].numpy()
        sub_df = pd.DataFrame(obs_value, columns = value)
        sub_pid = "MI" + "0" * (8 - len(str(counter))) + str(counter)
        sub_df['pid'] = [sub_pid for i in range(obs_value.shape[0])]
        sub_df['timestamp'] = timestamp_value
        sub_df['target'] = [target_value for i in range(obs_value.shape[0])]
        sub_df['Height'] = [static_value[0]] + [np.nan  for i in range((obs_value.shape[0] - 1))]
        df = pd.concat([df, sub_df[whole_col]], axis=0, ignore_index=True)
        counter += 1

    df['Height'].replace(-1, np.NaN, inplace=True)
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, "mimic3_mortality_{:s}.csv".format(mode)), index=False)

if __name__ == "__main__":
    collect_data("train", "../data/MI3", counter=0)
    collect_data("validation", "../data/MI3", counter=32984)
    collect_data("test", "../data/MI3", counter=54873)

