###############
#   Package   #
###############
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple

#######################
# package from myself #
#######################
from models.transformer_encoder_block import SinusoidalEmbedding3d, Encoder, EncoderLayer
from models.mufuse import MUFUSE

##############
#   Models   #
##############
class medfuse(nn.Module):
    def __init__(self,
                d_model: int = 128,
                num_heads: int = 2,
                ff_dim: int = 1024,
                ff_dropout: float = 0.05,
                attn_dropout: float = 0.05,
                norm_type: str = "LayerNorm",
                only_mask_first: bool = False,
                seq_len: int = None,
                NUM_LEN: int = None,
                CAT_LEN: int = None,
                num_layers: int = 5,
                decoder_down_factor: int = 4,
                decoder_dropout: float = 0.5,
                output_size: int = 1,
                num_embedding_module_kwargs: dict = {},
                cat_embedding_module_kwargs: dict = {},
                ):
        super(medfuse, self).__init__()
        # define variables and check it
        assert (seq_len is not None) and (seq_len > 0), ValueError("seq_len should be positive integer.")
        assert (NUM_LEN is not None) and (NUM_LEN > 0), ValueError("NUM_LEN should be positive integer.")
        assert (CAT_LEN is not None) and (CAT_LEN > 0), ValueError("CAT_LEN should be positive integer.")
        self.seq_len = seq_len

        self.d_model = d_model

        # value embedding layer
        self.num_embedding = MUFUSE(type = 'numerical' ,d_model=d_model, value_LEN=NUM_LEN, **num_embedding_module_kwargs)
        self.cat_embedding = MUFUSE(type = 'categorical', d_model=d_model, value_LEN=CAT_LEN, **cat_embedding_module_kwargs)

        # sinusoidal embedding layer
        ''' to be checked! '''
        self.pos_emb = SinusoidalEmbedding3d(d_model, dropout=0.1, max_seq_len=5000, mode='local',seq_len=seq_len) # unlearnable



        # transformer encoder
        encoder_layer = EncoderLayer(d_model = d_model,
                                    num_heads = num_heads,
                                    ff_dim = ff_dim,
                                    ff_dropout = ff_dropout,
                                    attn_dropout = attn_dropout,
                                    norm_type = norm_type,
                                    seq_len = seq_len,
                                    num_value = (NUM_LEN + CAT_LEN),
                                    )
        self.encoder = Encoder(encoder_layer = encoder_layer,
                               num_layers = num_layers,
                               only_mask_first = only_mask_first,
                               )
        
        self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model // decoder_down_factor),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(d_model // decoder_down_factor, output_size),
                                    )
    
    def forward(self, x_num_idx, x_num, x_num_mask, x_cat_idx, x_cat, x_cat_mask):
        '''
            param:
                x_num_idx: the vector to indicate the idx of each numerical data.
                        dim = (batch size, summarization times, NUM_LEN)
                x_num: the value of each numerical data.
                        dim = (batch size, summarization times, NUM_LEN)
                x_num_mask: the mask to indicate which value is missing.
                        0 = missing value, 1 = non-missing value.
                        dim = (batch size, summarization times, NUM_LEN)
                x_cat_idx: the vector to indicate the idx of each categorical data.
                        dim = (batch size, summarization times, CAT_LEN)
                x_cat: the value of each categorical data.
                        dim = (batch size, summarization times, CAT_LEN)
                x_cat_mask: the mask to indicate which value is missing.
                        0 = missing value, 1 = non-missing value.
                        dim = (batch size, summarization times, CAT_LEN)
            output:
                prob: the probability of the sample tested positive.
                        dim = (batch size, output_size=1)
        '''
        # value as word embedding
        x_num_embedded = self.num_embedding(x_num_idx, x_num, x_num_mask)
        x_cat_embedded = self.cat_embedding(x_cat_idx, x_cat, x_cat_mask)

        # concatenate numerical and categorical data
        x = torch.cat([x_num_embedded, x_cat_embedded], dim=2)
        x_mask = torch.cat([x_num_mask, x_cat_mask], dim=2)

        # positional encoding
        x = self.pos_emb(x)

        # flatten the vectors. it dimension will be (batch size, summarization times * number of values, d_model)
        x = x.view(x.shape[0], -1, x.shape[-1])

        # put x to transformer encoder
        x_embedded, _ = self.encoder(x, x_mask)

        # deal with all vectors.
        last_vec = torch.mean(x_embedded, dim=1)

        # decoder layer
        output = self.decoder(last_vec)
        prob = torch.sigmoid(output)
        return prob

if __name__ == '__main__':
    pass
