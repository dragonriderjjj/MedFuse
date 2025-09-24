################
#   Packages   #
################
import os
import sys
import random
import logging
import argparse
import importlib
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datetime import date
from typing import Tuple

#######################
# package from myself #
#######################
from utils.util import *

#################
#   Functions   #
#################
def FixedSeed(seed: int = 1122) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def GetDevice(device_id: int = 0) -> torch.device:
    torch.cuda.set_device(device_id)
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    return device

if __name__ == '__main__':
    # variabel from commmand
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, help='the name of the experiment.', required=True)
    parser.add_argument('-a', '--annotation', type=str, help='the annotation of the experiment.', required=True)
    parser.add_argument('-m', '--mode', type=str, help='train or test model.', default='train', choices=['train', 'test'])
    parser.add_argument('-c', '--config', type=str, help='the path of config(.json) file.', required=True)
    parser.add_argument('-r', '--resume_path', type=str, help='checkpoint path of the model.', default=None)
    parser.add_argument('-i', '--impute_value', type=int, help='the number to impute missing AFP', default=1)
    command_variable = parser.parse_args()

    # for debug
    FixedSeed(302)
    device = GetDevice(0)

    # get the config dictionary
    assert os.path.isfile(command_variable.config), print('config file is not exist.')
    config = GetConfigDict(command_variable.config)
    config['exp_info']['exp_name'] = command_variable.exp_name
    config['exp_info']['annotation'] = command_variable.annotation

    # create logger
    allLogger = None

    
    if command_variable.mode == 'train':
        # get dataset
        try:
            dataset_module = importlib.import_module(name=config['dataset']['module'])
            dataset_list = getattr(dataset_module, config['dataset']['type'])(data_dir=config['dataset']['data_path'], base_statistic_info_kwargs=config['dataset']['BasicStatisticInfo'], **config['dataset']['train']['GetDataset'])
        except:
            raise ValueError("dataset error.")
        training_dataset, val_dataset = dataset_list[0][0], dataset_list[0][1]

        # creat dataloader
        training_dataloader = DataLoader(training_dataset, **config['dataloader']['train'])
        val_dataloader = DataLoader(val_dataset, **config['dataloader']['valid'])

        # create model
        try:
            model_module = importlib.import_module(name=config['model']['module']) # get the model module
            model = getattr(model_module, config['model']['type'])(**config['model']['kwargs'])
        except:
            model = None

        # loss function
        try:
            loss_module = importlib.import_module(name=config['loss']['module'])
            loss_fn = getattr(loss_module, config['loss']['type'])(**config['loss']['kwargs'])
        except:
            loss_fn = None

        # optimizer
        try:
            optimizer_module = importlib.import_module(name=config['optimizer']['module'])
            optimizer = getattr(optimizer_module, config['optimizer']['type'])(model.parameters(), **config['optimizer']['kwargs'])
        except:
            optimizer = None

        # create trainer

        #(TO BE CHECKED!) the block to create ckt_save dir.
        checkpoint_save_path = os.path.join(training_result_path, 'checkpoint')
        os.makedirs(checkpoint_save_path, exist_ok=True)
        
        trainer_module = importlib.import_module(name=config['trainer']['module'])
        trainer = getattr(trainer_module, config['trainer']['type'])(
                         model = model,
                         loss_fn = loss_fn,
                         optimizer = optimizer,
                         training_dataloader = training_dataloader,
                         validation_dataloader = val_dataloader,
                         logger = allLogger,
                         checkpoint_save_path = checkpoint_save_path,
                         plot_save_path = plot_save_path,
                         device = device,
                         **config['trainer']['kwargs']
                         )
        trainer.train(**config['trainer']['train_kwargs'])
    
    if command_variable.mode == 'test':
        # get dataset
        try:
            dataset_module = importlib.import_module(name=config['dataset']['module'])
            dataset_list = getattr(dataset_module, config['dataset']['type'])(data_dir=config['dataset']['data_path'], base_statistic_info_kwargs=config['dataset']['BasicStatisticInfo'], impute_value=command_variable.impute_value, **config['dataset']['test']['GetDataset'])
        except:
            raise ValueError('dataset error.')
        testing_dataset = dataset_list[0][0]

        # create dataloader
        testing_dataloader = DataLoader(testing_dataset, **config['dataloader']['test'])

        # create model
        model_module = importlib.import_module(name=config['model']['module'])
        model = getattr(model_module, config['model']['type'])(**config['model']['kwargs'])

        # loss function
        loss_module = importlib.import_module(name=config['loss']['module'])
        loss_fn = getattr(loss_module, config['loss']['type'])(**config['loss']['kwargs'])

        # create tester
        tester_module = importlib.import_module(name=config['tester']['module'])
        tester = getattr(tester_module, config['tester']['type'])(
                         model = model,
                         loss_fn = loss_fn,
                         testing_dataloader = testing_dataloader,
                         logger = allLogger,
                         resume_model_path = command_variable.resume_path,
                         device = device
                         )
        tester.test(do_bootstrape=False)
