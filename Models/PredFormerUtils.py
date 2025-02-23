import torch
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    def make_square(self, B, L, device="cpu"): 
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.ones(mask_shape, dtype=torch.bool).to(device)

    @property
    def mask(self):
        return self._mask


class PositionalEmbedding2D(torch.nn.Module):

    def __init__(self, d_model, height, width):
        super(PositionalEmbedding2D, self).__init__()
        self.d_model = d_model

        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))

        pe_2d = torch.zeros(height, width, d_model)
        pe_2d.require_grad = False
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) *
            -(math.log(10000.0) / d_model))  # [d_model/2]

        pos_w = torch.arange(0., width).unsqueeze(1)  # [W, 1]
        pos_h = torch.arange(0., height).unsqueeze(1)  # [H, 1]

        pe_2d[:, :,
              0:d_model:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(
                  height, 1, 1)
        pe_2d[:, :,
              1:d_model:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(
                  height, 1, 1)
        pe_2d[:, :,
              d_model::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(
                  1, width, 1)
        pe_2d[:, :, d_model + 1::2] = torch.cos(
            pos_h * div_term).unsqueeze(1).repeat(1, width, 1)

        pe_2d = pe_2d.unsqueeze(0)
        self.register_buffer('pe_2d', pe_2d)

    def forward(self, x):
        return self.pe_2d[:, :x.size(1), :x.size(2), :]


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        queries = rearrange(queries, 'B L (H D) -> B L H D', H=H)
        keys = rearrange(keys, 'B S (H D) -> B S H D', H=H)
        values = rearrange(values, 'B S (H D) -> B S H D', H=H)
        # print(queries.shape, keys.shape, values.shape)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        # print(out.shape)
        out = rearrange(out, 'B L H D -> B L (H D)')

        return self.out_projection(out), attn
