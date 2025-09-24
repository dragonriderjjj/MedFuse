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
from sklearn.utils import resample

#######################
# package from myself #
#######################
from metric.concordance_index import ConcordanceIndex
from utils.util import safe_load_model_state_dict

#############
#   Class   #
#############
class BaseTester():
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 testing_dataloader: torch.utils.data.DataLoader,
                 logger: logging.Logger,
                 resume_model_path: str = None,
                 plot_probability_distribution: bool = False,
                 plot_save_path: str = '',
                 device: torch.device = torch.device("cpu"),
                ):
        # check if the resumed model (.pth) file exists.
        assert os.path.isfile(resume_model_path), 'resumed model file error.'
        assert not plot_probability_distribution or os.path.isdir(plot_save_path), 'plot saving dictionary does not exist.'

        # define the variables of the tester
        self.model = model
        
        # Use safe loading function to handle unexpected keys
        safe_load_model_state_dict(
            self.model, resume_model_path, logger, strict=False
        )
        
            
        self.loss_fn = loss_fn
        self.testing_dataloader = testing_dataloader
        
        self.logger = logger
        
        self.plot_probability_distribution = plot_probability_distribution
        self.plot_save_path = plot_save_path

        self.device = device

    def _probability_distribution_plot(self, prediction: np.ndarray, targets: np.ndarray, fig_name: str = 'testing_result') -> None:
        prediction_0 = prediction[(targets == 0)]
        prediction_1 = prediction[(targets == 1)]
        fig = plt.figure()
        sns.histplot(prediction_0, stat='density', bins=[0.01*x for x in range(101)], edgecolor='none', kde=True, color='green', label=f'label 0 ({len(prediction_0)} samples)')
        sns.histplot(prediction_1, stat='density', bins=[0.01*x for x in range(101)], edgecolor='none', kde=True, color='red', label=f'label 1 ({len(prediction_1)} samples)')
        plt.ylabel('Density')
        plt.xlabel('Probability')
        plt.legend()
        plt.title(fig_name)
        fig.savefig(os.path.join(self.plot_save_path, fig_name + '.png'))
        plt.close()

    def _test_epoch(self) -> Tuple[Tensor]:
        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)
            outputs = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)

            for batch_idx, (x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask, target) in enumerate(self.testing_dataloader):
                # put all variables to the appropriate device
                x_num_idx = x_num_idx.to(self.device)
                x_num = x_num.to(self.device)
                x_num_mask = x_num_mask.to(self.device)
                x_cat_idx = x_cat_idx.to(self.device)
                x_cat = x_cat.to(self.device)
                x_cat_mask = x_cat_mask.to(self.device)
                target = target.to(self.device)

                # feed the data to the model and get the output
                output = self.model(x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask)

                # collect the target and output
                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, target))

        return (outputs, targets)

    def test(self, do_bootstrape: bool = False) -> None:
        # define metric measurer
        accuracy_measurer = torchmetrics.classification.BinaryAccuracy().to(self.device)
        AUROC_measurer = torchmetrics.classification.BinaryAUROC().to(self.device)
        AUPRC_measurer = torchmetrics.classification.BinaryAveragePrecision().to(self.device)
        concordance_index_measurer = ConcordanceIndex()
        
        # get the output and target
        testing_outputs, testing_targets = self._test_epoch() # this should be changed.

        # loss
        testing_loss = self.loss_fn(testing_outputs, testing_targets)

        # accuracy
        testing_acc = accuracy_measurer(testing_outputs, testing_targets)
        if do_bootstrape: acc_half_CI_length = self._bootsrapping(testing_outputs, testing_targets, accuracy_measurer)

        # AUROC
        testing_AUROC = AUROC_measurer(testing_outputs, testing_targets)
        if do_bootstrape: AUROC_half_CI_length = self._bootsrapping(testing_outputs, testing_targets, AUROC_measurer)

        # AUPRC
        testing_AUPRC = AUPRC_measurer(testing_outputs, testing_targets.long())
        if do_bootstrape: AUPRC_half_CI_length = self._bootsrapping(testing_outputs, testing_targets.long(), AUPRC_measurer)


        # message builder
        msg_line_1 = f'Testing Result | '
        msg_line_2 = " "*(len(msg_line_1)-2) + "| "
        msg_line_3 = " "*(len(msg_line_1)-2) + "| "
        msg_line_4 = " "*(len(msg_line_1)-2) + "| "
        msg_line_1 += 'loss = {:.6f}\n'.format(testing_loss.cpu().item())
        msg_line_2 += 'accuracy = {:.6f}\n'.format(testing_acc.cpu().item()) if not do_bootstrape else 'accuracy = {:.6f} ({:.6f})\n'.format(testing_acc.cpu().item(), acc_half_CI_length)
        msg_line_3 += 'AUROC = {:.6f}\n'.format(testing_AUROC.cpu().item()) if not do_bootstrape else 'AUROC = {:.6f} ({:.6f})\n'.format(testing_AUROC.cpu().item(), AUROC_half_CI_length)
        msg_line_4 += 'AUPRC = {:.6f}\n'.format(testing_AUPRC.cpu().item()) if not do_bootstrape else 'AUPRC = {:.6f} ({:.6f})\n'.format(testing_AUPRC.cpu().item(), AUPRC_half_CI_length)
        msg = '\n' + msg_line_1 + msg_line_2 + msg_line_3 + msg_line_4

        self.logger.warning(msg)

        if self.plot_probability_distribution:
            self._probability_distribution_plot(testing_outputs.cpu().detach().numpy(), testing_targets.cpu().detach().numpy(), f'testing_result')

    def _sampler(self, number_of_samples: int, sample_times: int = 1000):
        self.sample = []
        for i in range(sample_times):
            self.sample.append(resample(np.arange(number_of_samples)))

    def _bootsrapping(self, testing_outputs: Tensor, testing_targets: Tensor, measurer = None, sample_times: int = 1000):
        assert measurer is not None, print("Measurer Wrong.")
        metric_value_record = []
        try:
            isinstance(self.sample, list)
        except:
            self._sampler(len(testing_outputs), sample_times)
        
        with torch.no_grad():
            for sample_idx in self.sample:
                metric_value = measurer(testing_outputs[sample_idx, :], testing_targets[sample_idx, :])
                metric_value_record.append(metric_value.cpu().item())

            metric_value_record = np.array(metric_value_record)
            half_confidence_interval_length = (np.quantile(metric_value_record, q=0.975) - np.quantile(metric_value_record, q=0.025)) / 2

        return half_confidence_interval_length


if __name__ == '__main__':
    pass
