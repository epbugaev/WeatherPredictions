import torch 

import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from Models.PredFormerUtils import PositionalEmbedding2D, FullAttention, AttentionLayer, TriangularCausalMask


class GatedTransformerBlock(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation=nn.SiLU):
        super(GatedTransformerBlock, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x, attn_mask=None, tau=None, delta=None, WindowAttention=False):
        # Attention
        res = x
        x = self.norm1(x)

        if WindowAttention: 
            x = self.attention(x)
        else:
            x, attn = self.attention(
                x, x, x,
                attn_mask,
            )
        
        x = res + self.dropout(x)
        if WindowAttention: 
            bt, h, w, d = x.shape
            x = rearrange(x, 'b h w d -> b (h w) d')

        # GLU
        res = x
        x = self.norm2(x)
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))) * self.conv2(x.transpose(-1, 1)))
        x = self.dropout(self.conv3(x).transpose(-1, 1))
        
        x = res + x

        if WindowAttention: 
            x = rearrange(x, 'b (h w) d -> b h w d', h=h, w=w)

        return x, None


class FullAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(FullAttentionLayer, self).__init__()
        self.gtb = GatedTransformerBlock(
            attention=attention,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
    def forward(self, x, attn_mask=None):
        attn_mask = TriangularCausalMask(x.shape[1])
        x, attn = self.gtb(x, attn_mask=attn_mask)
        return x, attn


class BinaryTSLayer(nn.Module):
    def __init__(self, attention_t, attention_s, d_model, d_ff=None, dropout=0.1):
        super(BinaryTSLayer, self).__init__()
        self.gtb_t = GatedTransformerBlock(
            attention=attention_t,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.gtb_s = GatedTransformerBlock(
            attention=attention_s,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None):
        B, T, H, W, D = x.shape

        # [b, t, h, w, d] -> 
        x = rearrange(x, 'b t h w d -> (b h w) t d')
        attn_mask = TriangularCausalMask(x.shape[0], x.shape[1], device=x.device)
        x, attn = self.gtb_t(x, attn_mask=attn_mask)
        x = rearrange(x, '(b h w) t d -> (b t) h w d', h=H, w=W)
        attn_mask.make_square(x.shape[0], x.shape[1], device=x.device)
        x, attn = self.gtb_s(x, attn_mask=attn_mask, WindowAttention=True) # No attn_mask for Space, applying WindowAttention
        x = rearrange(x, '(b t) h w d -> b t h w d', b=B, t=T)
        return x[:, -1, ...]
