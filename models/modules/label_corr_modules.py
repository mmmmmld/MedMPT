# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/08/08 20:14

import torch
import torch.nn as nn
from .graph_modules import DynamicGraphAttentionLayer
import numpy as np
import torch.nn.functional as F

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

class GAT(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, adj_file, adj_t=0.1, nlayers=2, nheads=[8, 1],
                 mha_split=False, alpha=0.2, dropout=0.2, pool='avg', **kwargs):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        assert len(nheads) == nlayers, "num of heads in MHA is not indicated clearly."
        self.mha_split = mha_split
        if nheads[-1] != 1:
            raise UserWarning("The output layer of GAT is multi-head attentions, which might be not encouraged.")
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.nheads = nheads
        self.pool = pool
        self.attentions = []
        for n in range(nlayers):
            if self.mha_split:
                assert hidden_dim % nheads[n] == 0, \
                    f'The num of attn_head in GAT layer {n}: {nheads[n]} is not divisible!'
            if n == 0:
                layer_in = input_dim
            else:
                if self.mha_split:
                    layer_in = hidden_dim
                else:
                    layer_in = hidden_dim * nheads[n - 1]
            # layer_in = hidden_dim if (n == 0 or self.mha_split) else hidden_dim * nheads[n - 1]
            layer_out = hidden_dim // nheads[n] if self.mha_split else hidden_dim
            attentions = [
                DynamicGraphAttentionLayer(layer_in, layer_out, dropout=dropout, alpha=alpha)
                for _ in range(nheads[n])]
            for i, attention in enumerate(attentions):
                self.add_module('attention_{}_{}'.format(n, i), attention)
            self.attentions.append(attentions)

        _, _adj_hat = self.gen_A(num_classes, adj_t, adj_file, debug=kwargs.get('adj_debug', False))
        self.A_hat = nn.Parameter(torch.from_numpy(_adj_hat).float(), requires_grad=False)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.apply(_init_weights)

    def forward(self, x):
        # x: (bs, N, in_dim)
        # x = self.in_proj(x)
        # x = self.dropout(x)
        for attns in self.attentions[:-1]:
            x = torch.cat([att(x, self.A_hat) for att in attns], dim=-1)
            x = self.dropout(x)
        x = [att(x, self.A_hat) for att in self.attentions[-1]]
        if self.nheads[-1] == 1:
            # x: [(bs, N, hidden_dim)]
            x = x[0]
        elif self.mha_split:
            # x: [(bs, N, hidden_dim // head_num) * head_num]
            x = torch.cat(x, dim=-1)
        else:
            # x: [(bs, N, hidden_dim) * head_num]
            x = self.avgpool(torch.stack(x, dim=-1))
        # x: (batch size, N, dim)
        # x = self.dropout(x)
        # x = self.out_proj(x)

        # (batch size, N, dim) -> (BATCH SIZE, dim)
        if self.pool == 'avg':
            x = self.avgpool(x.permute(0, 2, 1))
        elif self.pool == 'sum':
            x = torch.sum(x.permute(0, 2, 1), dim=-1, keepdim=True)
        elif self.pool == 'max':
            x = torch.max(x.permute(0, 2, 1), dim=-1, keepdim=True)
        elif self.pool == 'none':
            x = x.reshape(x.shape[0], -1, 1)
        else:
            raise NotImplementedError

        return x.reshape(x.shape[0], -1)

    def gen_A(self, num_classes, t, adj_file, debug=False):
        result = np.load(adj_file)
        _adj = result['adj']
        _nums = result['nums']
        _nums = _nums[:, np.newaxis]
        _adj = _adj / _nums

        if debug == 'random':
            _adj = np.random.random(_adj.shape)
            row, col = np.diag_indices_from(_adj)
            _adj[row, col] = 0
        elif debug == 'zeros':
            _adj = np.zeros(_adj.shape)
        elif debug == 'ones':
            _adj = np.ones(_adj.shape)
            row, col = np.diag_indices_from(_adj)
            _adj[row, col] = 0
        else:
            pass

        _adj_hat = np.where(_adj < t, 0, 1)

        _adj = _adj + np.identity(num_classes, np.int)
        _adj_hat = _adj_hat + np.identity(num_classes, np.int)
        return _adj.T, _adj_hat.T

class GPGAT_alt(nn.Module):  # ABAB
    def __init__(self, num_classes, input_dim, hidden_dim, adj_file='', adj_t=0.1, nlayers=2, nheads=[8, 1], alpha=0.2,
                 mha_split=False, dropout=0.2, pool='avg', **kwargs):
        """Dense version of GAT."""
        super(GPGAT_alt, self).__init__()
        assert len(nheads) == nlayers, "num of heads in MHA is not indicated clearly."
        self.mha_split = mha_split
        if nheads[-1] != 1:
            raise UserWarning("The output layer of GAT is multi-head attentions, which might be not encouraged.")
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.nheads = nheads
        self.pool = pool
        self.attentions = []
        for n in range(nlayers):
            if self.mha_split:
                assert hidden_dim % nheads[n] == 0, \
                    f'The num of attn_head in GAT layer {n}: {nheads[n]} is not divisible!'
            if n == 0:
                global_in = input_dim
            else:
                if self.mha_split:
                    global_in = hidden_dim
                else:
                    global_in = hidden_dim * nheads[n - 1]
            # global_in = hidden_dim if (n == 0 or self.mha_split) else hidden_dim * nheads[n - 1]
            global_out = hidden_dim // nheads[n] if self.mha_split else hidden_dim
            prior_in = hidden_dim if self.mha_split else hidden_dim * nheads[n]
            prior_out = hidden_dim // nheads[n] if self.mha_split else hidden_dim
            layer_global = [
                DynamicGraphAttentionLayer(global_in, global_out, dropout=dropout, alpha=alpha)
                for _ in range(nheads[n])]
            layer_prior = [
                DynamicGraphAttentionLayer(prior_in, prior_out, dropout=dropout, alpha=alpha)
                for _ in range(nheads[n])]
            for i, attention in enumerate(layer_global):
                self.add_module('attention_{}_global_{}'.format(n, i), attention)
            for i, attention in enumerate(layer_prior):
                self.add_module('attention_{}_prior_{}'.format(n, i), attention)
            attentions = [layer_global, layer_prior]
            self.attentions.append(attentions)

        _, _adj_hat_prior = self.gen_A(num_classes, adj_t, adj_file, debug=kwargs.get('adj_debug', False))
        self.A_hat_prior = nn.Parameter(torch.from_numpy(_adj_hat_prior).float(), requires_grad=False)
        _, _adj_hat_global = self.gen_A(num_classes, -1, adj_file, debug=kwargs.get('adj_debug', False))
        self.A_hat_global = nn.Parameter(torch.from_numpy(_adj_hat_global).float(), requires_grad=False)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.apply(_init_weights)

    def forward(self, x):
        # x: (bs, N, in_dim) -> (bs, N, hidden_dim)
        # x = self.in_proj(x)
        # x = self.dropout(x)
        for group_attns in self.attentions[:-1]:  # nlayers of [global, prior] groups
            x = torch.cat([att(x, self.A_hat_global) for att in group_attns[0]], dim=-1)
            # x = self.dropout(x)
            x = torch.cat([att(x, self.A_hat_prior) for att in group_attns[1]], dim=-1)
            x = self.dropout(x)

        x = torch.cat([att(x, self.A_hat_global) for att in self.attentions[-1][0]], dim=-1)
        x = [att(x, self.A_hat_prior) for att in self.attentions[-1][1]]
        if self.nheads[-1] == 1:
            # x: [(bs, N, hidden_dim)]
            x = x[0]
        elif self.mha_split:
            # x: [(bs, N, hidden_dim // head_num) * head_num]
            x = torch.cat(x, dim=-1)
        else:
            # x: [(bs, N, hidden_dim) * head_num]
            x = self.avgpool(torch.stack(x, dim=-1))
        # x: (batch size, N, dim)
        # x = self.dropout(x)
        # x = self.out_proj(x)

        # (batch size, N, dim) -> (BATCH SIZE, dim)
        if self.pool == 'avg':
            x = self.avgpool(x.permute(0, 2, 1))
        elif self.pool == 'sum':
            x = torch.sum(x.permute(0, 2, 1), dim=-1, keepdim=True)
        elif self.pool == 'max':
            x = torch.max(x.permute(0, 2, 1), dim=-1, keepdim=True)
        elif self.pool == 'none':
            x = x.reshape(x.shape[0], -1, 1)
        else:
            raise NotImplementedError

        return x.reshape(x.shape[0], -1)

    def gen_A(self, num_classes, t, adj_file, debug=False):
        result = np.load(adj_file)
        _adj = result['adj']
        _nums = result['nums']
        _nums = _nums[:, np.newaxis]
        _adj = _adj / _nums

        if debug == 'random':
            _adj = np.random.random(_adj.shape)
            row, col = np.diag_indices_from(_adj)
            _adj[row, col] = 0
        elif debug == 'zeros':
            _adj = np.zeros(_adj.shape)
        elif debug == 'ones':
            _adj = np.ones(_adj.shape)
            row, col = np.diag_indices_from(_adj)
            _adj[row, col] = 0
        else:
            pass

        _adj_hat = np.where(_adj < t, 0, 1)

        _adj = _adj + np.identity(num_classes, np.int)
        _adj_hat = _adj_hat + np.identity(num_classes, np.int)
        return _adj.T, _adj_hat.T

class GPGAT_seq(nn.Module):  # AABB
    def __init__(self, num_classes, input_dim, hidden_dim, adj_file='', adj_t=0.1, nlayers=2, nheads=[8, 1], alpha=0.2,
                 mha_split=False, dropout=0.2, pool='avg', **kwargs):
        """Dense version of GAT."""
        super(GPGAT_seq, self).__init__()
        assert len(nheads) == nlayers, "num of heads in MHA is not indicated clearly."
        self.mha_split = mha_split
        if nheads[-1] != 1:
            raise UserWarning("The output layer of GAT is multi-head attentions, which might be not encouraged.")
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.nheads = nheads
        self.pool = pool
        self.attentions = [[], []]  # global attentions, prior attentions
        for n in range(nlayers):
            if self.mha_split:
                assert hidden_dim % nheads[n] == 0, \
                    f'The num of attn_head in GAT layer {n}: {nheads[n]} is not divisible!'
            if n == 0:
                global_in = input_dim
            else:
                if self.mha_split:
                    global_in = hidden_dim
                else:
                    global_in = hidden_dim * nheads[n - 1]
            # global_in = hidden_dim if (n == 0 or self.mha_split) else hidden_dim * nheads[n - 1]
            global_out = hidden_dim // nheads[n] if self.mha_split else hidden_dim
            prior_in = hidden_dim if (n == 0 or self.mha_split) else hidden_dim * nheads[n - 1]
            prior_out = hidden_dim // nheads[n] if self.mha_split else hidden_dim
            layer_global = [
                DynamicGraphAttentionLayer(global_in, global_out, dropout=dropout, alpha=alpha)
                for _ in range(nheads[n])]
            layer_prior = [
                DynamicGraphAttentionLayer(prior_in, prior_out, dropout=dropout, alpha=alpha)
                for _ in range(nheads[n])]
            for i, attention in enumerate(layer_global):
                self.add_module('global_attention_{}'.format(n, i), attention)
            for i, attention in enumerate(layer_prior):
                self.add_module('prior_attention_{}'.format(n, i), attention)
            self.attentions[0].append(layer_global)
            self.attentions[1].append(layer_prior)

        _, _adj_hat_prior = self.gen_A(num_classes, adj_t, adj_file, debug=kwargs.get('adj_debug', False))
        self.A_hat_prior = nn.Parameter(torch.from_numpy(_adj_hat_prior).float(), requires_grad=False)
        _, _adj_hat_global = self.gen_A(num_classes, -1, adj_file, debug=kwargs.get('adj_debug', False))
        self.A_hat_global = nn.Parameter(torch.from_numpy(_adj_hat_global).float(), requires_grad=False)

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.apply(_init_weights)

    def forward(self, x):
        # x: (bs, N, in_dim) -> (bs, N, hidden_dim)
        # x = self.in_proj(x)
        # x = self.dropout(x)

        # global attentions
        for attns in self.attentions[0][:-1]:  # nlayers of global attentions
            x = torch.cat([att(x, self.A_hat_global) for att in attns], dim=-1)
            x = self.dropout(x)
        x = [att(x, self.A_hat_global) for att in self.attentions[0][-1]]
        if self.nheads[-1] == 1:
            # x: [(bs, N, hidden_dim)]
            x = x[0]
        elif self.mha_split:
            # x: [(bs, N, hidden_dim // head_num) * head_num]
            x = torch.cat(x, dim=-1)
        else:
            # x: [(bs, N, hidden_dim) * head_num]
            x = self.avgpool(torch.stack(x, dim=-1))
        x = self.dropout(x)

        # prior attentions
        for attns in self.attentions[1][:-1]:  # nlayers of prior attentions
            x = torch.cat([att(x, self.A_hat_prior) for att in attns], dim=-1)
            x = self.dropout(x)
        x = [att(x, self.A_hat_prior) for att in self.attentions[1][-1]]
        if self.nheads[-1] == 1:
            # x: [(bs, N, hidden_dim)]
            x = x[0]
        elif self.mha_split:
            # x: [(bs, N, hidden_dim // head_num) * head_num]
            x = torch.cat(x, dim=-1)
        else:
            # x: [(bs, N, hidden_dim) * head_num]
            x = self.avgpool(torch.stack(x, dim=-1))
        # x: (batch size, N, dim)
        # x = self.dropout(x)
        # x = self.out_proj(x)

        # (batch size, N, dim) -> (BATCH SIZE, dim)
        if self.pool == 'avg':
            x = self.avgpool(x.permute(0, 2, 1))
        elif self.pool == 'sum':
            x = torch.sum(x.permute(0, 2, 1), dim=-1, keepdim=True)
        elif self.pool == 'max':
            x = torch.max(x.permute(0, 2, 1), dim=-1, keepdim=True)
        elif self.pool == 'none':
            x = x.reshape(x.shape[0], -1, 1)
        else:
            raise NotImplementedError

        return x.reshape(x.shape[0], -1)

    def gen_A(self, num_classes, t, adj_file, debug=False):
        result = np.load(adj_file)
        _adj = result['adj']
        _nums = result['nums']
        _nums = _nums[:, np.newaxis]
        _adj = _adj / _nums

        if debug == 'random':
            _adj = np.random.random(_adj.shape)
            row, col = np.diag_indices_from(_adj)
            _adj[row, col] = 0
        elif debug == 'zeros':
            _adj = np.zeros(_adj.shape)
        elif debug == 'ones':
            _adj = np.ones(_adj.shape)
            row, col = np.diag_indices_from(_adj)
            _adj[row, col] = 0
        else:
            pass

        _adj_hat = np.where(_adj < t, 0, 1)

        _adj = _adj + np.identity(num_classes, np.int)
        _adj_hat = _adj_hat + np.identity(num_classes, np.int)
        return _adj.T, _adj_hat.T
