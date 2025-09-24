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

from typing import List, Dict, Tuple, Optional
from torch import Tensor
from lifelines.utils import concordance_index

#############
#   Class   #
#############
class ConcordanceIndex():
    def __init__(self):
        pass

    def __call__(self, event_times: Tensor, targets: Tensor, outputs: Tensor):
        with torch.no_grad():
            return concordance_index(event_times.cpu().numpy(), -outputs.cpu().numpy(), targets.cpu().numpy())

