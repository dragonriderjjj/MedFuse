import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_encoder_block import SinusoidalEmbedding3d, Encoder, EncoderLayer, TimeEncoder
from models.mufuse import MUFUSE


class medfuse(nn.Module):
    def __init__(
        self,
        d_model: int=128,
        num_heads: int=2,
        pe_mode: str="local",
        ff_dim: int=1024,
        ff_dropout: float=0.05,
        attn_dropout: float=0.05,
        norm_type: str='LayerNorm',
        only_mask_first: bool=False,
        seq_len: int=None,
        NUM_LEN: int=None,
        CAT_LEN: int=None,
        num_layers: int=5,
        decoder_down_factor: int=4,
        decoder_dropout: float=0.5,
        output_size: int=1,
        embedding_module_kwargs: dict={},
        bert_pooling: bool=False,
        device: torch.device=torch.device("cpu"),
    ):
        super(medfuse, self).__init__()
        assert seq_len and (seq_len > 0), ValueError("seq_len should be positive integer.")
        assert NUM_LEN and (NUM_LEN > 0), ValueError("NUM_LEN should be positive integer.")
        assert CAT_LEN and (CAT_LEN > 0), ValueError("CAT_LEN should be positive integer.")
        self.seq_len = seq_len
        self.d_model = d_model
        self.bert_pooling = bert_pooling
        
        self.concat_embedding = MUFUSE(type='numerical', d_model=d_model, value_LEN=NUM_LEN + CAT_LEN, **embedding_module_kwargs)

        # Time encoder for timestep-based feature modulation
        # self.time_encoder = TimeEncoder(d_model=d_model, dropout=0.1, hidden_dim=d_model // 4)  # Default hidden dimension
        
        # Comment out positional encoding for future use
        # self.pos_emb = SinusoidalEmbedding3d(d_model, dropout=0.1, max_seq_len=5000, mode=pe_mode,seq_len=seq_len)
        self.cls_emb = nn.Embedding(1, d_model)

        # self.total_embedding = nn.Linear(d_model, d_model)

        encoder_layer = EncoderLayer(
        # encoder_layer = Attn2DLayer(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            norm_type=norm_type,
            seq_len=seq_len,
            num_value=(NUM_LEN + CAT_LEN),
        )
        self.encoder = Encoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            only_mask_first=only_mask_first,
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // decoder_down_factor),
            nn.GELU(),
            nn.Dropout(decoder_dropout),
            nn.Linear(d_model // decoder_down_factor, output_size)
        )
    

    def forward(self, x_idx, x, x_mask, x_timestamp, *_):
        '''
            output:
                prob: the probability of the sample tested positive.
                        dim = (batch size, output_size=1)
        '''
        # value as word embedding
        x = self.concat_embedding(x_idx, x, x_mask)

        # Apply time-based feature modulation using element-wise multiplication
        # if x_timestamp is not None:
        #     time_encoding = self.time_encoder(x_timestamp)  # (batch_size, seq_len, num_features, d_model)
        #     x = x * time_encoding  # Element-wise multiplication for time-aware features
        
        # Comment out positional encoding (replaced by time encoder)
        # x = self.pos_emb(x, x_timestamp)

        # flatten the vectors. it dimension will be (batch size, summarization times * number of values, d_model)
        x = x.view(x.shape[0], -1, x.shape[-1])

        if self.bert_pooling:
            cls_tokens = torch.zeros(x.shape[0]).to(x.device).long()
            cls_tokens = self.cls_emb(cls_tokens)
            x = torch.cat([cls_tokens.unsqueeze(1), x], dim=1)

        # Compute time steps for rotary positional encoding
        if x_timestamp is not None:
            # x_timestamp shape: (batch_size, seq_len, num_value)
            # We need time steps per sequence position, so take the first value feature's timestamp
            time_steps = x_timestamp[:, :, 0].view(-1)  # Flatten to (batch_size * seq_len,)
            # Create a sequence of positions for the flattened sequence
            seq_positions = torch.arange(x.shape[1], device=x.device).float()
            time_steps = seq_positions
        else:
            time_steps = None

        # put x to transformer encoder
        x_embedded, _ = self.encoder(x, x_mask)

        # deal with all vectors.
        ''' this part should be surveyed.'''
        last_vec = x_embedded[:, 0] if self.bert_pooling else torch.mean(x_embedded, dim=1)
        # print("last_vec: ", last_vec.shape)

        # decoder layer
        output = self.decoder(last_vec)
        prob = torch.sigmoid(output)
        return prob
