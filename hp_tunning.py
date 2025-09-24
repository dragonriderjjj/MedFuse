import os 
import sys
from typing import Tuple
import importlib

from torch.utils.data import DataLoader

def objective(trial, args, config, allLogger, device, training_result_path, plot_save_path):

    d_model = trial.suggest_int('d_model', 80, 180, step=8)
    num_heads = trial.suggest_int('num_heads', 1, 2)
    num_layers = trial.suggest_int('num_layers', 1, 2)
    ff_dim = trial.suggest_int('ff_dim', 128, 256, step=16)
    ff_dropout = trial.suggest_float('ff_dropout', 0.1, 0.4)
    attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.2)
    decoder_dropout = trial.suggest_float('decoder_dropout', 0.1, 0.4)
    decoder_down_factor = trial.suggest_int('decoder_down_factor', 2, 4, step=2)
    lr = trial.suggest_float('lr', 1e-5, 5e-4)
    value_proj_dim= trial.suggest_int('value_proj_dim', 2, 8, step=2)
    allLogger.info(
        f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, ff_dim={ff_dim}, "
        f"ff_dropout={ff_dropout}, attn_dropout={attn_dropout}, decoder_dropout={decoder_dropout}, "
        f"decoder_down_factor={decoder_down_factor}, lr={lr}, value_proj_dim={value_proj_dim}")



    try:
        dataset_module = importlib.import_module(name=config['dataset']['module'])
        dataset_list = getattr(dataset_module, config['dataset']['type'])(data_dir=config['dataset']['data_path'], base_statistic_info_kwargs=config['dataset']['BasicStatisticInfo'], **config['dataset']['train']['GetDataset'])
    except:
        raise ValueError("dataset error.")
    training_dataset, val_dataset = dataset_list[0][0], dataset_list[0][1]

    # creat dataloader
    params = {**config['dataloader']['train']}
    # params['batch_size'] = batch_size
    training_dataloader = DataLoader(training_dataset, **params, )
    params = {**config['dataloader']['valid']}
    # params['batch_size'] = batch_size
    val_dataloader = DataLoader(val_dataset, **params, )

    # create model
    try:
        model_module = importlib.import_module(name=config['model']['module']) # get the model module
        params = {**config['model']['kwargs']}
        params['d_model'] = d_model
        params['num_heads'] = num_heads
        params['num_layers'] = num_layers
        params['ff_dim'] = ff_dim 
        params['ff_dropout'] = ff_dropout
        params['attn_dropout'] = attn_dropout
        params['decoder_dropout'] = decoder_dropout
        params['decoder_down_factor'] = decoder_down_factor
        params['embedding_module_kwargs']['value_proj_dim'] = value_proj_dim
        model = getattr(model_module, config['model']['type'])(**params)
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
        params = {**config['optimizer']['kwargs']}
        params['lr'] = lr
        optimizer = getattr(optimizer_module, config['optimizer']['type'])(model.parameters(), **params)
    except:
        optimizer = None

    # create trainer

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
    record = trainer.train(**config['trainer']['train_kwargs'])

    return record['val_AUPRC'][-1]