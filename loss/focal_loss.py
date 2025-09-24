###############
#   Package   #
###############
import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

#####################
#   Loss Function   #
#####################
# REF: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = .5, gamma: float = 2.0, reduction: str = 'none', eps: float = 0.0) -> None:
        assert reduction in ['none', 'mean', 'sum'], print('reduction method error.')
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def _LossKernel(self, output: Tensor, target: Tensor, alpha: float, gamma: float, reduction: str, eps: Optional[float]) -> Tensor:
        loss = -1 * alpha * torch.pow(1 - output, gamma) * target * torch.log(output + eps) \
               - (1 - alpha) * torch.pow(output, gamma) * (1 - target) * torch.log(1 - output + eps)

        if reduction == 'mean':
            loss = torch.mean(loss)
        if reduction == 'sum':
            loss = torch.sum(loss)

        return loss

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return self._LossKernel(output, target, self.alpha, self.gamma, self.reduction, self.eps)
