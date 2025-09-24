###############
#   Package   #
###############
import os
import time
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from typing import List, Dict, Tuple, Optional
from torch import Tensor

#######################
# package from myself #
#######################
from utils.util import ConsumingTime

#############
#   Class   #
#############
class BaseTrainer():
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim, # should be checked.
                 training_dataloader: torch.utils.data.DataLoader,
                 validation_dataloader: torch.utils.data.DataLoader,
                 logger: logging.Logger,
                 do_early_stop: bool = False,
                 lr_scheduler: torch.optim = None,
                 checkpoint_save_path: str = '',
                 checkpoint_period: int = 5,
                 plot_process: bool = True,
                 plot_probability_distribution: bool = False,
                 plot_save_path: str = '',
                 resume: bool = False,
                 resume_checkpoint_path: str = '',
                 device: torch.device = torch.device("cpu"),
                ):
            # check the correction of variables
            assert os.path.isdir(checkpoint_save_path), ValueError('checkpoint saving dictionary does not exist.')
            assert not (plot_process or plot_probability_distribution) or os.path.isdir(plot_save_path), ValueError('plot saving dictionary does not exist.')
            assert not resume or os.path.isfile(resume_checkpoint_path), ValueError('resumed checkpoint file does not exist.')

            # define the variables of the training step
            self.model = model
            if resume:
                self.model.load_state_dict(torch.load(resume_checkpoint_path))
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.training_dataloader = training_dataloader
            self.validation_dataloader = validation_dataloader
            self.do_early_stop = do_early_stop
            self.lr_scheduler = lr_scheduler

            self.logger = logger
            self.ckp_save_path = checkpoint_save_path
            self.ckp_period = checkpoint_period

            self.plot_probability_distribution = plot_probability_distribution
            self.plot_process = plot_process
            self.plot_save_path = plot_save_path

            self.device = device

    def _train_epoch(self) -> tuple:
        self.model.train()
        self.model.to(self.device)
        outputs = torch.FloatTensor().to(self.device)
        targets = torch.FloatTensor().to(self.device)
        start_time = time.time()

        for batch_idx, (x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask, target, day_delta) in enumerate(self.training_dataloader):
            # put all variables to appropriate device.
            x_num_idx = x_num_idx.to(self.device)
            x_num = x_num.to(self.device)
            x_num_mask = x_num_mask.to(self.device)
            x_cat_idx = x_cat_idx.to(self.device)
            x_cat = x_cat.to(self.device)
            x_cat_mask = x_cat_mask.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            # feed data to the model and get the output
            output = self.model(x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask)
            
            # collect the target and output.
            outputs = torch.cat((outputs, output))
            targets = torch.cat((targets, target))

            # compute loss
            loss = self.loss_fn(output, target)

            # optimization step
            loss.backward()
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        end_time = time.time()

        return (outputs, targets, ConsumingTime(start_time, end_time))

    def _valid_epoch(self) -> Tuple[Tensor]:
        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)
            outputs = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)

            for batch_idx, (x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask, target, day_delta) in enumerate(self.validation_dataloader):
                # put all variables to appropriate device.
                x_num_idx = x_num_idx.to(self.device)
                x_num = x_num.to(self.device)
                x_num_mask = x_num_mask.to(self.device)
                x_cat_idx = x_cat_idx.to(self.device)
                x_cat = x_cat.to(self.device)
                x_cat_mask = x_cat_mask.to(self.device)
                target = target.to(self.device)

                # feed data to the model and get the output
                output = self.model(x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask)

                # collect the target and output
                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, target))

        return (outputs, targets)

    def _plot(self, record: dict, metric_list: list = ['loss', 'accuracy', 'AUROC', 'AUPRC']) -> None:
        for metric in metric_list:
            assert len(record[metric]) == len(record['val_'+metric]), 'record length error.'
            x_ = np.arange(1, len(record[metric])+1)
            fig = plt.figure()
            plt.plot(x_, record[metric], color='r', marker='.', label='train')
            plt.plot(x_, record['val_'+metric], color='c', marker='.', label='valid')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            if metric == 'loss':
                plt.legend(loc="upper right")
            else:
                plt.legend(loc="lower right")
            plt.title(metric)
            fig.savefig(os.path.join(self.plot_save_path, f'{metric}.png'))
            plt.close()

    def _probability_distribution_plot(self, prediction: np.ndarray, targets: np.ndarray, fig_name: str = "valid") -> None:
        prediction_0 = prediction[(targets == 0)]
        prediction_1 = prediction[(targets == 1)]
        fig = plt.figure()
        sns.histplot(prediction_0, stat='density', bins=[0.01*x for x in range(101)], edgecolor="none", kde=True, color='green', label=f'label 0 ({len(prediction_0)} samples)')
        sns.histplot(prediction_1, stat='density', bins=[0.01*x for x in range(101)], edgecolor="none", kde=True, color='red', label=f'label 1 ({len(prediction_1)} samples)')
        plt.ylabel('Density')
        plt.xlabel('Probability')
        plt.legend()
        plt.title(fig_name)
        fig.savefig(os.path.join(self.plot_save_path, fig_name+'.png'))
        plt.close()

    def train(self, epoch: int = 100):
        # we only track loss, accuracy, AUROC, AUPRC, and c_index of the training set and the validation set.
        ### c_index not implemented.
        key_list = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'AUROC', 'val_AUROC', 'AUPRC', 'val_AUPRC']
        record = dict([(key, []) for key in key_list])
        
        # define metric measurer
        accuracy_measurer = torchmetrics.classification.BinaryAccuracy().to(self.device)
        AUROC_measurer = torchmetrics.classification.BinaryAUROC().to(self.device)
        AUPRC_measurer = torchmetrics.classification.BinaryAveragePrecision().to(self.device)


        for ep_idx in range(1, epoch+1):
            training_outputs, training_targets, training_time = self._train_epoch()
            val_outputs, val_targets = self._valid_epoch()

            # loss
            training_loss = self.loss_fn(training_outputs, training_targets)
            val_loss = self.loss_fn(val_outputs, val_targets)
            record['loss'].append(training_loss.cpu().item())
            record['val_loss'].append(val_loss.cpu().item())

            # accuracy
            training_acc = accuracy_measurer(training_outputs, training_targets)
            val_acc = accuracy_measurer(val_outputs, val_targets)
            record['accuracy'].append(training_acc.cpu().item())
            record['val_accuracy'].append(val_acc.cpu().item())

            # AUROC
            training_AUROC = AUROC_measurer(training_outputs, training_targets)
            val_AUROC = AUROC_measurer(val_outputs, val_targets)
            record['AUROC'].append(training_AUROC.cpu().item())
            record['val_AUROC'].append(val_AUROC.cpu().item())

            # AUPRC
            training_AUPRC = AUPRC_measurer(training_outputs, training_targets.long())
            val_AUPRC = AUPRC_measurer(val_outputs, val_targets.long())
            record['AUPRC'].append(training_AUPRC.cpu().item())
            record['val_AUPRC'].append(val_AUPRC.cpu().item())

            # message builder
            msg_line_1 = f'Epoch [{ep_idx}/{epoch}] | '
            msg_line_2 = " "*(len(msg_line_1)-2) + "| "
            msg_line_1 += '(train) loss = {:.6f}, accuracy = {:.6f}, AUROC = {:.6f}, AUPRC = {:.6f}\n'.format(record['loss'][-1], record['accuracy'][-1], record['AUROC'][-1], record['AUPRC'][-1])
            msg_line_2 += '(valid) loss = {:.6f}, accuracy = {:.6f}, AUROC = {:.6f}, AUPRC = {:.6f}'.format(record['val_loss'][-1], record['val_accuracy'][-1], record['val_AUROC'][-1], record['val_AUPRC'][-1])
            msg = '\n' + msg_line_1 + msg_line_2

            self.logger.warning(msg)
            
            if self.plot_probability_distribution:
                self._probability_distribution_plot(training_outputs.cpu().detach().numpy(), training_targets.cpu().detach().numpy(), f'Ep_{ep_idx}_train')
                self._probability_distribution_plot(val_outputs.cpu().detach().numpy(), val_targets.cpu().detach().numpy(), f'Ep_{ep_idx}_valid')

            if ep_idx % self.ckp_period == 0:
                torch.save(self.model.state_dict(), os.path.join(self.ckp_save_path, f'ckt_ep_{ep_idx}.pth'))

        if self.plot_process:
            self._plot(record)

if __name__ == '__main__':
    pass
