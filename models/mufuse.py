###############
#   Package   #
###############
import os

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from copy import deepcopy
from typing import Tuple, List

#######################
#   Embedding Layer   #
#######################

class ValueProjector(nn.Module):
    def __init__(self, value_LEN, value_proj_dim, k_gates):
        super().__init__()
        self.fc1 = nn.Linear(1, value_proj_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(value_proj_dim, k_gates) 
        self.feature_scale = nn.Parameter(torch.ones(value_LEN, k_gates))
        self.feature_bias = nn.Parameter(torch.zeros(value_LEN, k_gates))

    def forward(self, x):
        # x: (B, T, F, 1)
        
        out = self.fc1(x)  # (B, T, F, value_proj_dim)
        out = self.act(out)
        out = self.fc2(out)  # (B, T, F, k_gates)
        
        out = out * self.feature_scale.unsqueeze(0).unsqueeze(0) + \
                self.feature_bias.unsqueeze(0).unsqueeze(0)  # (B, T, F, k_gates)
        # out = self.gate_act(out)

        return out 

class MUFUSE(nn.Module):


    def __init__(self,
                type: str = 'numerical',  # 'numerical' or 'categorical'
                d_model: int = 128,
                value_LEN: int = None,
                independent_padding: bool = True,
                max_norm: float = None,
                value_proj_dim: int = 4,
                k_gates: int = 4, 
                time_LEN: int = 48,
                use_pretrained: bool = False,
                freeze_epochs: int = 0,
                pretrained_weight_path: str = None
                ):
        super(MUFUSE, self).__init__()

        assert (value_LEN is not None) and (value_LEN > 0), ValueError("value_LEN should be positive integer.")
        assert d_model % k_gates == 0, ValueError(f"d_model ({d_model}) must be divisible by k_gates ({k_gates})")
        
        self.type = type
        self.value_LEN = value_LEN
        self.independent_padding = independent_padding
        self.time_LEN = time_LEN
        self.d_model = d_model
        self.k_gates = k_gates
        self.subspace_dim = d_model // k_gates

        self.value_projector = ValueProjector(self.value_LEN, value_proj_dim, k_gates)

        # Add categorical value embedding for categorical features
        if type == 'categorical':
            # For categorical features, we need separate embeddings for categorical values
            max_cat_values = 100  # Maximum number of categorical values per feature
            self.cat_value_embedding = nn.Embedding(
                num_embeddings=max_cat_values,
                embedding_dim=d_model,
                padding_idx=0  # 0 for missing/unknown values
            )
            # Feature-specific transformation for categorical features
            self.cat_transform = nn.Linear(d_model * 2, d_model)
            self.cat_dropout = nn.Dropout(0.1)

        if independent_padding:
            self.value_embedding = nn.Embedding(num_embeddings=value_LEN+1,
                                                embedding_dim=d_model,
                                                padding_idx=0,
                                                max_norm=max_norm,
                                                )
        else:
            self.value_embedding = nn.Embedding(num_embeddings=value_LEN,
                                                embedding_dim=d_model,
                                                max_norm=max_norm,
                                                )

    def forward(self, x_idx, x, x_mask):
        """
            param: 
                x_idx: the vector to indicate each value's index.
                            dim = (batch size, summarization times, value_LEN)
                x: the value of data.
                            dim = (batch size, summarization times, value_LEN)
                x_mask: the mask to indicate which value is missing.
                            0 = missing value, 1 = non-missing value.
                            dim = (batch size, summarization times, value_LEN)
            output:
                x_emb: (batch size, summarization times, value_LEN, d_model)
        """

        if self.type == 'categorical':
            
            # Get feature-specific embeddings based on feature indices  
            embedding_idx = x_idx * x_mask if self.independent_padding else x_idx
            feature_emb = self.value_embedding(embedding_idx)  # (B, T, F, d_model)
            
            # Convert categorical values to integers and handle missing values
            cat_values = x.long()  # (B, T, F)
            cat_values_masked = cat_values * x_mask.long()  # Set missing values to 0
            
            # Clip values to valid range to avoid index errors
            cat_values_clipped = torch.clamp(cat_values_masked, 0, 99)  # 0-99 range
            
            # Get categorical value embeddings
            cat_emb = self.cat_value_embedding(cat_values_clipped)  # (B, T, F, d_model)
            
            # Combine feature embedding and categorical value embedding
            combined = torch.cat([feature_emb, cat_emb], dim=-1)  # (B, T, F, 2*d_model)
            x_emb = self.cat_transform(combined)  # (B, T, F, d_model)
            x_emb = self.cat_dropout(x_emb)
            
            # Apply activation
            x_emb = torch.tanh(x_emb)
            
            # Apply mask to zero out missing values
            x_emb = x_emb * x_mask.unsqueeze(-1).expand_as(x_emb)

        else:    
            embedding_idx = x_idx * x_mask if self.independent_padding else x_idx
            embedding_value = x * x_mask if self.independent_padding else x  # (B, T, F)
            x_feature_emb = self.value_embedding(embedding_idx)  # (B, T, F, d_model)

            embedding_value = embedding_value.unsqueeze(-1).to(self.value_projector.fc1.weight.dtype)  # (B, T, F, 1)
            gates = self.value_projector(embedding_value)  # (B, T, F, k_gates)


            # x_feature_emb: (B, T, F, d_model) -> (B, T, F, k_gates, subspace_dim)
            B, T, F, _ = x_feature_emb.shape
            x_feature_subspaces = x_feature_emb.view(B, T, F, self.k_gates, self.subspace_dim)
            

            gates_expanded = gates.unsqueeze(-1)  # (B, T, F, k_gates, 1)
            
            gated_subspaces = x_feature_subspaces * gates_expanded  # (B, T, F, k_gates, subspace_dim)
            
            x_emb = gated_subspaces.view(B, T, F, self.d_model)  # (B, T, F, d_model)

        # old version of SCANE
        # embedding_idx = x_idx * x_mask if self.independent_padding else x_idx
        # embedding_value = x * x_mask if self.independent_padding else x  # (B, T, F)
        # x_feature_emb = self.value_embedding(embedding_idx)  # (B, T, F, d_model)

        # # 共用 projector + scaling
        # x_value_emb = self.value_projector(embedding_value.unsqueeze(-1))  # (B, T, F)

        # x_emb = x_value_emb * x_feature_emb  # (B, T, F, d_model)
        return x_emb
        
if __name__ == '__main__':
    pass