# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/08/08 20:14

import torch
import torch.nn as nn

class BiomarkerAttention(nn.Module):
    def __init__(self, bio_num, hidden_dim=256, nlayers=2, nheads=8, feedforward_dim=1024, dropout=0.1):
        super(BiomarkerAttention, self).__init__()
        self.bio_embedding = torch.nn.Parameter(torch.randn((bio_num, hidden_dim)))  # (N, d)
        self.cls_embedding = torch.nn.Parameter(torch.randn((1, hidden_dim)))  # (N, d)
        _layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nheads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(_layer, nlayers)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-8, elementwise_affine=True)
        self.apply(self._init_weights)
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.output_hidden_size = hidden_dim

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, x, missing_mask=None):
        """
        x: input biomarker vector, (bs, N)
        mask(binary matrix): indicate which is missing value, (bs, N):
            0 means ground truth, 1 means missing
        """
        assert missing_mask is not None, "the mask of missing lab test value is not specified."
        batch_size, bio_num = x.shape
        # expand each test value into a test feature: (bs, N) * (N, d) -> (bs, N, d)
        bio_embeddings = torch.einsum('bi,ij->bij', x, self.bio_embedding)

        # add global embedding to input and mask, and take layer norm: (bs, N+1, d)
        cls_embedding = torch.stack([self.cls_embedding]*batch_size, dim=0)
        input_embeddings = torch.cat([cls_embedding, bio_embeddings], dim=1)
        input_embeddings = self.layer_norm(input_embeddings)
        missing_mask = torch.cat([torch.zeros(batch_size, 1, device=missing_mask.device), missing_mask], dim=-1)  # (bs, N+1)

        # transpose input: (N+1, bs, d)
        input_embeddings = input_embeddings.transpose(0, 1)
        missing_mask = (missing_mask != 0)

        # output: (N+1, bs, d) -> (bs, N+1, d)
        output_embeddings = self.transformer_encoder(input_embeddings, src_key_padding_mask=missing_mask)
        output_embeddings = output_embeddings.transpose(0, 1)

        return output_embeddings[:, 0, ...], output_embeddings[:, 1:, ...]






