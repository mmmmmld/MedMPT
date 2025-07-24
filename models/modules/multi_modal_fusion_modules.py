# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/08/15 14:59

import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, input_dim, nlayers=2, hidden_dim=128, nheads=4, feedforward_dim=512, dropout=0.1):
        super(AttentionFusion, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = list(input_dim)
        self.input_num = len(input_dim)
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim[i], input_dim[i]),
                    nn.ReLU(),
                    nn.Linear(input_dim[i], hidden_dim)
                )
                for i in range(len(input_dim))]
        )

        LayerClass = (
            nn.TransformerEncoderLayer
        )
        try:
            _layer = LayerClass(
                hidden_dim,
                nheads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                attn_drop=nn.Dropout(p=dropout),
            )
        except:
            _layer = LayerClass(
                hidden_dim,
                nheads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
            )
        self.transformer_encoder = nn.TransformerEncoder(_layer, nlayers)
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

    def forward(self, feature_tuple):
        split_hidden_state = [self.mlp[i](feature_tuple[i]) for i in range(self.input_num)]  # [(bs, d)*3]
        concat_input = torch.stack(split_hidden_state, dim=0)  # (3, bs, d)
        # print('input shape: ', concat_input.shape)
        output = self.transformer_encoder(concat_input)  # (l, batch_size, hidden_size)
        # Undo the transpose and bring batch to dim 0.
        output = torch.squeeze(self.avgpool1d(output.permute(1, 2, 0)), dim=-1)  # (batch_size, max_caption_length, vocab_size)
        return output, 0
