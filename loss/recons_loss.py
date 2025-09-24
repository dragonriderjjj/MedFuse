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
from typing import Optional, Tuple

#####################
#   Loss Function   #
#####################
class ReconstructLoss(nn.Module):
    def __init__(self, weight: Tuple[int] = (1, 1, 1, 1)) -> None:
        assert (len(weight) == 4), ValueError("the number of weight is not true.")
        super(ReconstructLoss, self).__init__()
        self.weight = [i / sum(weight) for i in weight]

    def _MSELossKernel(self, output: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
        return (((output - target) * mask).pow(2)).mean() if mask is not None else ((output - target).pow(2)).mean()

    def forward(self,
                num_idx: Tensor,
                num_idx_reconst: Tensor,
                num_value: Tensor,
                num_value_reconst: Tensor,
                cat_idx: Tensor,
                cat_idx_reconst: Tensor,
                cat_value: Tensor,
                cat_value_reconst: Tensor,
                num_mask: Tensor = None,
                cat_mask: Tensor = None,
                ) -> Tuple[Tensor]:
        # adjust the shape of each tensor
        # for "numerical data", i use the batch size as anchor; for "idx" and "categorical" data, i use the last dimension as anchor.
        num_idx_reconst = num_idx_reconst.view(-1, num_idx_reconst.shape[-1])
        num_idx = num_idx.view(-1) * num_mask.view(-1) if num_mask is not None else num_idx.view(-1)
        cat_idx_reconst = cat_idx_reconst.view(-1, cat_idx_reconst.shape[-1])
        cat_idx = cat_idx.view(-1) * cat_mask.view(-1) if cat_mask is not None else cat_idx.view(-1)
        cat_value_reconst = cat_value_reconst.view(-1, cat_value_reconst.shape[-1])
        cat_value = cat_value.view(-1) * cat_mask.view(-1) if cat_mask is not None else cat_value.view(-1)
        cat_value = cat_value + 1
        num_value = num_value.view(num_value.shape[0], -1)
        num_value_reconst = num_value_reconst.view(num_value_reconst.shape[0], -1)
        num_mask = num_mask.view(num_mask.shape[0], -1)
        # compute loss
        num_idx_reconstruct_loss = F.cross_entropy(num_idx_reconst, num_idx, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)
        num_value_reconstruct_loss = self._MSELossKernel(num_value_reconst, num_value, num_mask)
        cat_idx_reconstruct_loss = F.cross_entropy(cat_idx_reconst, cat_idx, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)
        cat_value_reconstruct_loss = F.cross_entropy(cat_value_reconst, cat_value.long(), weight=None, ignore_index=1, reduction='mean', label_smoothing=0.0)
        return ((self.weight[0] * num_idx_reconstruct_loss + self.weight[1] * num_value_reconstruct_loss + self.weight[2] * cat_idx_reconstruct_loss + self.weight[3] * cat_value_reconstruct_loss), num_idx_reconstruct_loss, num_value_reconstruct_loss, cat_idx_reconstruct_loss, cat_value_reconstruct_loss)

if __name__ == '__main__':
    pass
