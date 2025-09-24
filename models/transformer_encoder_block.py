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

############################
#   Positional Embedding   #
############################
# REF: https://blog.csdn.net/qq_41897800/article/details/114777064'''

class TimeEncoder(nn.Module):
    """
    Simple time encoder that transforms timesteps into d_model shape for element-wise 
    multiplication with feature embeddings using a linear → activation → linear architecture.
    """
    def __init__(self,
                 d_model: int,
                 hidden_dim: int = 16,
                 dropout: float = 0.1,
                 activation: str = 'tanh'):
        super(TimeEncoder, self).__init__()
        self.d_model = d_model

        # Simple linear -> activation -> linear architecture
        self.time_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(activation.lower(), nn.GELU())
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.time_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x_timestamp):
        """
        Args:
            x_timestamp: (batch_size, seq_len, num_features) - timestamps for each feature
        
        Returns:
            time_encoding: (batch_size, seq_len, num_features, d_model) - time encodings for multiplication
        """
        # Handle input shape and normalize
        batch_size, seq_len, num_features = x_timestamp.shape
        
        # Flatten for processing: (batch_size * seq_len * num_features, 1)
        time_flat = x_timestamp.view(-1, 1).float()
        
        # Normalize timestamps to a reasonable range (0-1 based on min-max in batch)
        time_min = time_flat.min()
        time_max = time_flat.max()
        # if time_max > time_min:
        #     time_normalized = (time_flat - time_min) / (time_max - time_min)
        # else:
        #     time_normalized = time_flat  # All timestamps are the same
        
        # Apply time encoding network
        time_encoding = self.time_net(time_flat )  # (batch_size * seq_len * num_features, d_model)
        
        # Reshape back to original dimensions
        time_encoding = time_encoding.view(batch_size, seq_len, num_features, self.d_model)
        
        return time_encoding

class SinusoidalEmbedding3d(nn.Module):
    def __init__(self,
        d_model: int,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        mode: str=["local", "global", "cve"],
        seq_len: int=48,
        learn_emb: bool=False,
        d_keys: int=None,
        num_heads: int=1,
    ):
        super(SinusoidalEmbedding3d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.seq_len = seq_len
        self.learn_emb = learn_emb

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        ### APE
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        ### tAPE
        # pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_seq_len))
        # pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_seq_len))
        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, x, timestamp = 0):
        ''' to be check!'''
        if self.mode == "local":
            x = x + self.pe[:, :x.size(1)]
        elif self.mode == "global":
            scaled_time = torch.round((timestamp / self.seq_len) * (self.max_seq_len-1)).long() # for P12
            x[:] = x[:] + self.pe[:, scaled_time.squeeze(2)]
            
        return self.dropout(x)

###########################
#   Transformer Encoder   #
###########################
class EncoderLayer(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                ff_dim: int,
                ff_dropout: float = 0.1,
                attn_dropout: float = 0.1,
                norm_type: str = "LayerNorm",
                seq_len: int = None,
                num_value: int = None,
                ):
        super(EncoderLayer, self).__init__()
        assert seq_len is not None, ValueError("seq_len assigned uncorrectly.")
        assert num_value is not None, ValueError("num_value assigned uncorrectly.")
        self.seq_len = seq_len
        self.num_value = num_value
        self.num_heads = num_heads

        self.enc_self_attn = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            dropout = attn_dropout,
            batch_first = True,
            )

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_dim, d_model),
            )

        assert norm_type in ["LayerNorm", "BatchNorm1d"], "Wrong Normalization Type."
        if norm_type == "LayerNorm":
            self.attn_norm = nn.LayerNorm(d_model)
            self.ff_norm = nn.LayerNorm(d_model)
        else:
            self.attn_norm = nn.BatchNorm1d(self.seq_len*(self.num_value))
            self.ff_norm = nn.BatchNorm1d(self.seq_len*(self.num_value))
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ff_dropout = nn.Dropout(ff_dropout)

    def forward(self,
                src: Tensor,
                src_mask: Tensor,
                src_key_padding_mask: Tensor,
                ):
        output, attn_weight = self.enc_self_attn(
                                    query = src,
                                    key = src,
                                    value = src,
                                    key_padding_mask = src_key_padding_mask,
                                    attn_mask = src_mask
                                    )
        output = src + self.attn_dropout(output) # residual connection
        output = self.attn_norm(output) #normalization

        output_ = self.ff(output) # feed forward
        output = output + self.ff_dropout(output_) # residual connection
        output = self.ff_norm(output) # normalization

        return output, attn_weight

class Encoder(nn.Module):
    def __init__(self,
                encoder_layer: EncoderLayer,
                num_layers: int,
                only_mask_first: bool = False,
                time_steps: Tensor = None
                ):
        super(Encoder, self).__init__()
        self.seq_len = encoder_layer.seq_len
        self.num_value = encoder_layer.num_value
        self.num_heads = encoder_layer.num_heads
        self.only_mask_first = only_mask_first

        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
            )

    def get_key_padding_mask(self, mask: Tensor) -> Tensor:
        """
        param:
            mask: (Batch Size, seq_len, num_value)
                The mask indicates the positions of missing values.

        output:
            key_padding_mask: (Batch Size, seq_len * num_value)
        """
        flatten_mask = mask.view(mask.size(0), -1)
        return ~flatten_mask.bool()

    def forward(self,
                src: Tensor,
                mask: Tensor
                ) -> Tuple[Tensor, Tensor]:
        src_key_padding_mask = self.get_key_padding_mask(mask) # create key_padding_mask
        attns_all = []

        output = src
        counter = 0 if self.only_mask_first else -1

        for layer in self.layers:
            if counter == -1:
                output, attn = layer(src=output,
                                    src_mask=None,
                                    src_key_padding_mask=src_key_padding_mask,
                                    )
                attns_all.append(attn)

            elif counter == 0:
                output, attn = layer(src=output,
                                    src_mask=None,
                                    src_key_padding_mask=src_key_padding_mask,
                                    )
                attns_all.append(attn)
                counter += 1

            else:
                output, attn = layer(src=output,
                                    src_mask=None,
                                    src_key_padding_mask=None,
                                    )
                attns_all.append(attn)

        attns_all = torch.stack(attns_all)
        return output, attns_all

if __name__ == "__main__":
    pass
