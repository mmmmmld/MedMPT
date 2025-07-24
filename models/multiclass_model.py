# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/04/23 20:45

import os
from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from timm.models.layers import trunc_normal_
transformers.logging.set_verbosity_error()
from .modules.vit import VisionTransformer
from .modules.pos_embed import interpolate_pos_embed
from .modules.fusion_modules import FusionTransformer

class MultiClassModel(nn.Module):
    def __init__(self, args, stage='train', **kwargs):
        super(MultiClassModel, self).__init__()
        # define parameters
        self.args = args
        self.stage = stage
        self.class_num = self.args.class_num
        self.dropout = nn.Dropout(p=self.args.vision_dropout)
        self.vision_encoder = VisionTransformer(
            patch_size=self.args.vision_patch_size, embed_dim=self.args.vision_embed_dim,
            depth=self.args.vision_depth, num_heads=self.args.vision_num_heads, num_classes=-1,
            mlp_ratio=self.args.vision_mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pretrained=self.args.vision_pretrained, checkpoint=self.args.vision_pretrained_weight,
            freeze_pretrained_layers=self.args.vision_freeze_pretrained_layers,
        )
        self.vision_width = self.args.vision_embed_dim
        self.vision_fusion = FusionTransformer(embed_dim=self.vision_width)
        if self.stage == 'train' and self.args.vision_load_dir != '':
            self.load_local_pretrained_weight()
        self.head = nn.Sequential(
            nn.Linear(self.vision_width, self.vision_width),
            nn.ReLU(),
            nn.Linear(self.vision_width, self.class_num),
        )

        self.avgpool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)

        if self.args.loss == 'bce':
            assert self.class_num > 2 
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.args.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        if self.args.linear_probe:
            self.head = nn.Linear(self.vision_width, self.class_num)
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.vision_fusion.parameters():
                param.requires_grad = False

    def load_pretrained_weight(self, weights):
        if weights == 'imagenet':
            load_path = '/root/maliangdi/models/vit/vit_base_patch16_224_in21k.pth'
        else:
            raise NotImplementedError

        checkpoint = torch.load(load_path, map_location='cpu')
        print("Load vision encoder from pre-trained checkpoint: %s" % load_path)

        vision_state_dict = self.vision_encoder.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint and k in vision_state_dict and checkpoint[k].shape != vision_state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        # interpolate position embedding
        interpolate_pos_embed(self.vision_encoder, checkpoint)

        # load pre-trained model
        msg = self.vision_encoder.load_state_dict(checkpoint, strict=False)
        print(msg)
        if 'head.weight' in vision_state_dict or 'head.bias' in vision_state_dict:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
            # manually initialize fc layer
            trunc_normal_(self.vision_encoder.head.weight, std=2e-5)

    def load_local_pretrained_weight(self):
        load_path = os.path.join(self.args.vision_load_dir,
                                 f'checkpoints/checkpoint_{self.args.vision_load_epoch}.pth')
        checkpoint = torch.load(load_path, map_location='cpu')
        print("Load vision encoder from pre-trained checkpoint: %s" % load_path)
        checkpoint_model = checkpoint['model']

        ### load vision model
        vision_pretrained_weights = OrderedDict()
        vision_fusion_pretrained_weights = OrderedDict()
        for k, v in checkpoint_model.items():
            if k.startswith('module.vision_encoder_q.'):
                vision_pretrained_weights[k[len('module.vision_encoder_q.'):]] = v
            if k.startswith('module.vision_fusion_q.'):
                vision_fusion_pretrained_weights[k[len('module.vision_fusion_q.'):]] = v

        vision_state_dict = self.vision_encoder.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in vision_pretrained_weights and vision_pretrained_weights[k].shape != vision_state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del vision_pretrained_weights[k]

        # interpolate position embedding
        interpolate_pos_embed(self.vision_encoder, vision_pretrained_weights)

        # load pre-trained model
        msg = self.vision_encoder.load_state_dict(vision_pretrained_weights, strict=False)
        print(msg)
        if 'head.weight' in vision_state_dict or 'head.bias' in vision_state_dict:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
            # manually initialize fc layer
            trunc_normal_(self.vision_encoder.head.weight, std=2e-5)

        self.vision_fusion.load_state_dict(vision_fusion_pretrained_weights, strict=True)
        print("Load vision fusion encoder from pre-trained checkpoint: %s" % load_path)


    def forward(self, image, target=None, mode=''):
        if len(image.size()) == 5:
            batch_size, seq_len, c, h, w = image.shape
        else:
            batch_size, c, h, w = image.shape
            seq_len = -1
        image = image.reshape(-1, c, h, w)
        image_feat = self.vision_encoder(image)  # (bs * seq_len, d)
        if seq_len > 0:
            image_feat = image_feat.reshape(batch_size, seq_len, image_feat.shape[-1])  # (bs, seq, d)
            image_feat, _ = self.vision_fusion(image_feat)

        # image_feat = self.dropout(image_feat)
        out = self.head(image_feat)

        if mode == 'vis':
            return out

        if self.args.loss == 'bce':
            target = target.unsqueeze(-1)
        loss = self.criterion(out, target)

        if self.args.loss == 'bce':
            prediction = F.sigmoid(out)  # (bs, 1)
            prediction = torch.cat([1 - prediction, prediction], dim=-1)  # (bs, 2)
        else:
            prediction = F.softmax(out, dim=-1)  # (bs, c)

        return {'loss': loss, 'prediction': prediction}


