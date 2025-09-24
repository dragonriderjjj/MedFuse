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
import optuna
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.empty_cache()

from torch.utils.data import DataLoader
from datetime import date, datetime

from typing import Tuple

#######################
# package from myself #
#######################
from utils.util import *
from hp_tunning import objective

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
def ConstructFolder(exp_name: str = 'debug', annotation: str = "", main_folder: str = 'exp_output') -> Tuple[str]:
    '''
        exp_output dictionary structure:
            exp_output -> [checkpoint_dir] (composed of ckp_dict_prefix) -> [exp_date + postfix] -> training_result
                                                                                                    -> checkpoint -> [model_ckt]
                                                                                                    -> config.json
                                                                                                    -> log_file
                                                                                                    -> fig -> [training_loss_plot]
                                                                                                 -> testing_result
                                                                                                    -> log_file
    '''
    # Get current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M")  # Format: YYYYMMDD_HHMM
    folder_name = f"{current_time}_{annotation}"

    training_result_path = os.path.join(main_folder, exp_name, folder_name, 'training_result')
    testing_result_path = os.path.join(main_folder, exp_name, folder_name, 'testing_result')
    os.makedirs(training_result_path, exist_ok=True)
    os.makedirs(testing_result_path, exist_ok=True)
    return (training_result_path, testing_result_path)

if __name__ == '__main__':
    # variabel from commmand
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, help='the name of the experiment.', required=True)
    parser.add_argument('-a', '--annotation', type=str, help='the annotation of the experiment.', required=True)
    parser.add_argument('-m', '--mode', type=str, help='train or test model.', default='train', choices=['train', 'test'])
    parser.add_argument('-c', '--config', type=str, help='the path of config(.json) file.', required=True)
    parser.add_argument('-r', '--resume_path', type=str, help='checkpoint path of the model.', default=None)
    parser.add_argument('-b', '--bootstrapping', type=bool, help='whether to do bootstrap.', default=False)
    parser.add_argument('-o', '--optuna', action="store_true", help='whether to use optuna to tune hyperparameters.', default=False)
    parser.add_argument('-d', '--device_id', type=int, help='the id of the device to use.', default=0)
    command_variable = parser.parse_args()

    # for debug
    FixedSeed(302)
    device = GetDevice(command_variable.device_id)

    # set exp_output folder struture
    training_result_path, testing_result_path = ConstructFolder(exp_name=command_variable.exp_name, annotation=command_variable.annotation)

    # get the config dictionary
    assert os.path.isfile(command_variable.config), print('config file is not exist.')
    config = GetConfigDict(command_variable.config)
    config['exp_info']['exp_name'] = command_variable.exp_name
    config['exp_info']['annotation'] = command_variable.annotation
    SaveConfigDict(config, os.path.join(training_result_path, 'config.json')) if command_variable.mode == 'train' else SaveConfigDict(config, os.path.join(testing_result_path, 'config.json')) # save the config file to training_result folder

    #(TO BE CHECKED!) the block to create fig dir.
    plot_save_path = os.path.join(training_result_path, 'fig') if command_variable.mode == 'train' else os.path.join(testing_result_path, 'fig')
    os.makedirs(plot_save_path, exist_ok=True)

    # create logger
    logName = os.path.join(training_result_path, 'debug.log') if command_variable.mode == 'train' else os.path.join(testing_result_path, 'debug.log')
    allLogger = logging.getLogger('allLogger')
    allLogger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(logName, mode='a')
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)

    AllFormat = logging.Formatter("%(asctime)s - [%(filename)s, line: %(lineno)d]: %(message)s")
    file_handler.setFormatter(AllFormat)
    stream_handler.setFormatter(AllFormat)

    allLogger.addHandler(file_handler)
    allLogger.addHandler(stream_handler)
    
    if command_variable.mode == 'train':
        if command_variable.optuna:
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, command_variable, config, allLogger, device, training_result_path, plot_save_path), n_trials=30, gc_after_trial=True)
            joblib.dump(study, "study.pkl")
            print('Number of finished trials: ', len(study.trials))
            print('Best trial:', study.best_trial.params)
            print('Best value:', study.best_value)

            sys.exit(0)
        # get dataset
        try:
            print(config['dataset']['data_path'])
            dataset_module = importlib.import_module(name=config['dataset']['module'])
            dataset_list = getattr(dataset_module, config['dataset']['type'])(data_dir=config['dataset']['data_path'], base_statistic_info_kwargs=config['dataset']['BasicStatisticInfo'], **config['dataset']['train']['GetDataset'])
        except:
            raise ValueError("dataset error.")
        # print(dataset_list[0].shape)
        training_dataset, val_dataset = dataset_list[0][0], dataset_list[0][1]

        # creat dataloader
        training_dataloader = DataLoader(training_dataset, **config['dataloader']['train'])
        val_dataloader = DataLoader(val_dataset, **config['dataloader']['valid'])

        # create model

        params = {**config['model']['kwargs']}
        model_module = importlib.import_module(name=config['model']['module']) # get the model module
        model = getattr(model_module, config['model']['type'])(**params)

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
        _ = trainer.train(**config['trainer']['train_kwargs'])
    
    if command_variable.mode == 'test':
        # copy the checkpoint to the testing folder
        shutil.copy2(command_variable.resume_path, os.path.join(testing_result_path, "ckt.pth"))

        # get dataset
        try:
            dataset_module = importlib.import_module(name=config['dataset']['module'])
            dataset_list = getattr(dataset_module, config['dataset']['type'])(data_dir=config['dataset']['data_path'], base_statistic_info_kwargs=config['dataset']['BasicStatisticInfo'], **config['dataset']['test']['GetDataset'])
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
        print(f"Testing model: {config['tester']['module']}")
        tester = getattr(tester_module, config['tester']['type'])(
                         model = model,
                         loss_fn = loss_fn,
                         testing_dataloader = testing_dataloader,
                         logger = allLogger,
                         resume_model_path = command_variable.resume_path,
                         plot_probability_distribution = True,
                         plot_save_path = plot_save_path,
                         device = device
                         )
        tester.test(do_bootstrape=command_variable.bootstrapping)
