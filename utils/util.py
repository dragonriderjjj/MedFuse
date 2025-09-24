###############
#   Package   #
###############
import os
import json
import time
import random
import importlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from typing import Tuple
from pathlib import Path
from datetime import date
from functools import reduce
from collections import OrderedDict
from matplotlib import pyplot as plt

import torch
import logging

#################
#   Functions   #
#################
def ConsumingTime(start_time, end_time):
    differnece_time = time.gmtime(end_time - start_time)
    formated_time = time.strftime("%H hours, %M minutes, %S seconds", differnece_time)
    return formated_time

def GetConfigDict(config_path: str = None) -> dict:
    assert os.path.isfile(config_path), "config file does not exist."
    with open(config_path) as config:
        config_dict = json.load(config)

    return config_dict

def SaveConfigDict(config_dict: dict, save_path) -> None:
    json_object = json.dumps(config_dict, indent=4, ensure_ascii=False)
    with open(save_path, "w", encoding='utf8') as outputfile:
        outputfile.write(json_object)

if __name__ == '__main__':
    path = '../config/template.json'
    tmp = GetConfigDict(path)
    print(tmp)


# ======= Statistical Summary Class =======
class BaseStatisticInfo():
    """
    Compute basic statistical information (mean, median, std, mode) for numeric and categorical attributes
    from a base DataFrame. Optionally generate histogram plots and drop outliers.

    Parameters:
    - base_dataframe (pd.DataFrame): input data
    - attributes: object with NUM_COLS, CAT_COLS, NUM_LEN, CAT_LEN attributes
    - drop_outlier (bool): whether to exclude outliers in numeric data
    - outlier_fence (float): threshold for outlier rejection
    - statistic_plot (bool): whether to plot distributions
    - plot_output_dir (str): output folder for plots
    """
    def __init__(self,
                base_dataframe: pd.DataFrame,
                attributes,
                drop_outlier: bool=False,
                outlier_fence: float=None,
                statistic_plot: bool=False,
                plot_output_dir: str='./statistic_plot'
                ):
        if statistic_plot:
            self.plot(base_dataframe, attributes, plot_output_dir)
        self.num_mean, self.num_median, self.num_std, self.cat_mode = self.ComputeStatisticInfo(base_dataframe, attributes, drop_outlier, outlier_fence)
    
    @staticmethod
    def ComputeStatisticInfo(base_data: pd.DataFrame, attributes, drop_outlier: bool = False, m: float = None) -> tuple:
        """
        Compute mean, median, std for numeric attributes and mode for categorical attributes.

        Parameters:
        - base_data (pd.DataFrame): dataset to compute statistics from
        - attributes: attribute info with NUM_COLS, CAT_COLS, NUM_LEN, CAT_LEN
        - drop_outlier (bool): whether to exclude outliers from computation
        - m (float): outlier threshold (used only if drop_outlier is True)

        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (mean, median, std, mode)
        """
        def reject_outlier(arr: np.ndarray, m: float = 3.) -> np.ndarray:
            # REF: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
            # REF: https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
            assert m is not None, ValueError("outlier bound should be positive number.")
            na_drop_arr = arr[~np.isnan(arr)]
            d = np.abs(na_drop_arr - np.median(na_drop_arr))
            mdev = np.quantile(d, .75) - np.quantile(d, .25)
            s = d/mdev if mdev else np.zeros(len(d))
            return na_drop_arr[s<m]

        num_mean = np.zeros(attributes.NUM_LEN)
        num_median = np.zeros(attributes.NUM_LEN)
        num_std = np.zeros(attributes.NUM_LEN)
        cat_mode = np.zeros(attributes.CAT_LEN)

        for idx, col in enumerate(attributes.NUM_COLS):
            sub_data = base_data[col].values
            if drop_outlier:
                try:
                    sub_data = reject_outlier(sub_data, m=m)
                except:
                    sub_data = np.zeros_like(sub_data)
            num_mean[idx] = np.nanmean(sub_data)
            num_median[idx] = np.nanmedian(sub_data)
            num_std[idx] = np.nanstd(sub_data)

        for idx, col in enumerate(attributes.CAT_COLS):
            cat_mode[idx] = stats.mode(base_data[col].values, nan_policy="omit")[0][0]

        return (num_mean, num_median, num_std, cat_mode)

    def plot(self, base_data: pd.DataFrame, attributes, output_dir: str = "./figure", drop_outlier: bool = False, m: float = None) -> None:
        """
        Generate and save histogram plots for numeric attributes.

        Parameters:
        - base_data (pd.DataFrame): input dataset
        - attributes: attributes object with NUM_COLS
        - output_dir (str): directory to save plots
        - drop_outlier (bool): whether to drop outliers before plotting
        - m (float): threshold for outlier removal
        """
        os.makedirs(output_dir, exist_ok=True)

        for col in attributes.NUM_COLS:
            values = base_data[col].values
            clean_values = values[~np.isnan(values)]
            if drop_outlier:
                clean_values = self.reject_outlier(clean_values, m)

            fig = plt.figure()
            sns.set_theme(style="whitegrid")
            sns.displot(clean_values)
            plt.title(f"{col} Distribution")
            plt.savefig(f"{output_dir}/{col}_histogram.png")
            plt.close(fig)


# ======= Visualization Tools =======        
def token_visualization(embeddings, feat_size: int, seq_lenth: int, png_name: str='token_visualization'):
    """
    Visualize token embeddings as a color grid.

    Parameters:
    - embeddings (Tensor): 4D tensor [B, T, F, D] from model output
    - feat_size (int): number of features per time step
    - seq_length (int): number of time steps (sequence length)
    - png_name (str): file name prefix for saved figure
    """
    tokens = [embeddings[0, i, j, :] for i in range(embeddings.size(1)) for j in range(embeddings.size(2))]
    unique_tokens = list(set(tuple(token.tolist()) for token in tokens))
    token_color_map = {token: plt.cm.tab10(i) for i, token in enumerate(unique_tokens)}
    color_matrix = np.zeros((seq_lenth, feat_size, 3))

    for i, token in enumerate(tokens):
        row, col = divmod(i, feat_size)
        color_matrix[row, col] = token_color_map[tuple(token.tolist())][:3]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(color_matrix, aspect='auto')
    ax.set_xticks(np.arange(feat_size))
    ax.set_yticks(np.arange(seq_lenth))
    ax.set_xticklabels([f'feat{i+1}' for i in range(feat_size)], rotation=45, ha='right')
    ax.set_yticklabels([f'T{i+1}' for i in range(seq_lenth)])
    plt.title("Token Visualization")
    plt.savefig(f'{png_name}.png')
    plt.close()


def safe_load_model_state_dict(model, checkpoint_path, logger=None, strict=False):
    """
    Safely load model state dict with option to ignore unexpected keys
    
    Args:
        model: PyTorch model instance
        checkpoint_path: Path to the checkpoint file
        logger: Logger instance for warnings (optional)
        strict: Whether to strictly match keys (default: False)
    
    Returns:
        tuple: (missing_keys, unexpected_keys)
    """
    checkpoint = torch.load(checkpoint_path)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if logger:
        if unexpected_keys:
            logger.warning(f"Ignored unexpected keys in checkpoint: {unexpected_keys}")
        if missing_keys:
            logger.warning(f"Missing keys in model: {missing_keys}")
    else:
        if unexpected_keys:
            print(f"Warning: Ignored unexpected keys in checkpoint: {unexpected_keys}")
        if missing_keys:
            print(f"Warning: Missing keys in model: {missing_keys}")
  
