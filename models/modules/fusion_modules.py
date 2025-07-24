# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/07/14 17:24


import torch
from torch import nn
from einops import repeat


class FusionTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=2, num_heads=8, dim_feedforward=3072, dropout=0.1):
        super(FusionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Embedding(512, self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        try:
            enc_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, attn_drop=nn.Dropout(p=dropout))
        except:

            enc_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.layer_norm = nn.LayerNorm(
            embed_dim, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: [batch size, seq num, d]
        batch_size, seq_len, fdim = x.shape
        # layer position encoding
        position_indices = torch.arange(seq_len + 1, dtype=torch.long, device=x.device)
        position_indices = position_indices.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.pos_embed(position_indices)  # (batch_size, seq_len + 1, dim)
        # print('layer_position_embeddings shape: ', layer_position_embeddings.shape)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (bs, seq_len + 1, d)
        x = x + position_embeddings
        x = self.layer_norm(x)
        x = self.dropout(x)
        # (batch_size, seq_len + 1, dim) -> (seq_len + 1, batch_size, dim) -> (batch_size, seq_len + 1, dim)
        out = self.encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
        return out[:, 0, :], out[:, 1:, :]

