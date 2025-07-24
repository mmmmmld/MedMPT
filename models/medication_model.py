# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/25 20:28

import os
import torch
import numpy as np
from torch import nn
from functools import partial
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from .modules.fusion_modules import FusionTransformer
from .modules.structured_encoder import BiomarkerAttention
from .modules.multi_modal_fusion_modules import AttentionFusion
from .modules.label_corr_modules import GPGAT_alt
from .modules.bert import TransformerEncoder
from .modules import pooler
from .modules.vit import VisionTransformer
from .modules.pos_embed import interpolate_pos_embed


class ParallelMedModel(nn.Module):
    def __init__(self, args, stage='train'):  # config = config
        super(ParallelMedModel, self).__init__()
        self.args = args
        self.stage = stage
        self.class_num = self.args.num_classes
        self.dropout = nn.Dropout(p=0.1)
        self.input = []
        self.single_modal_output_feature_sizes = []

        if 'ct' in self.args.input:
            self.input.append('ct')
            self.visual_dropout = nn.Dropout(p=self.args.vision_dropout)
            self.visual_encoder = self.build_visual_encoder()
            self.visual_feature_size = self.args.vision_embed_dim
            self.single_modal_output_feature_sizes.append(self.visual_feature_size)
            if self.args.model not in ["v1", "v8"]:
                self.fusion_module = FusionTransformer(embed_dim=self.visual_feature_size)

            if self.stage == 'train' and self.args.vision_load_dir != '':
                load_path = os.path.join(self.args.vision_load_dir, f'checkpoints/checkpoint_{self.args.vision_load_epoch}.pth')
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
                vision_state_dict = self.visual_encoder.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in vision_pretrained_weights and vision_pretrained_weights[k].shape != vision_state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del vision_pretrained_weights[k]

                # interpolate position embedding
                interpolate_pos_embed(self.visual_encoder, vision_pretrained_weights)

                # load pre-trained model
                msg = self.visual_encoder.load_state_dict(vision_pretrained_weights, strict=False)
                print(msg)
                if 'head.weight' in vision_state_dict or 'head.bias' in vision_state_dict:
                    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
                    # manually initialize fc layer
                    trunc_normal_(self.visual_encoder.head.weight, std=2e-5)

                if self.args.model not in ["v1", "v8"]:
                    self.fusion_module.load_state_dict(vision_fusion_pretrained_weights, strict=True)
                    print("Load vision fusion encoder from pre-trained checkpoint: %s" % load_path)


        if 'report' in self.args.input:
            self.input.append('report')
            self.textual_dropout = nn.Dropout(p=self.args.text_dropout)
            self.textual_encoder = self.build_textual_encoder()
            self.textual_feature_size = self.textual_encoder.config.hidden_size
            self.single_modal_output_feature_sizes.append(self.textual_feature_size)

            if self.stage == 'train' and self.args.text_load_dir != '':
                load_path = os.path.join(self.args.text_load_dir, f'checkpoints/checkpoint_{self.args.text_load_epoch}.pth')
                checkpoint = torch.load(load_path, map_location='cpu')
                print("Load text encoder from pre-trained checkpoint: %s" % load_path)
                checkpoint_model = checkpoint['model']
                text_pretrained_weights = OrderedDict()
                for k, v in checkpoint_model.items():
                    if k.startswith('module.text_encoder_q.'):
                        text_pretrained_weights[k[len('module.text_encoder_q.'):]] = v

                # load pre-trained model
                msg = self.textual_encoder.load_state_dict(text_pretrained_weights, strict=False)
                print(msg)

        if 'biomarker' in self.args.input:
            self.input.append('biomarker')
            self.biomarker_dropout = nn.Dropout(p=self.args.bio_dropout)
            self.biomarker_encoder = self.build_structured_encoder()
            self.biomarker_feature_size = self.biomarker_encoder.output_hidden_size
            self.single_modal_output_feature_sizes.append(self.biomarker_feature_size)

        print(f"=> input mode: {self.input}")

        if len(self.input) > 1:
            self.cross_modal_fusion_module = self.bulid_multimodal_fusion_module()
            self.fusion_modal_output_feature_size = self.cross_modal_fusion_module.output_hidden_size
            self.fusion_dropout = nn.Dropout(p=self.args.fusion_dropout)
            self.output_feature_size = self.fusion_modal_output_feature_size
        else:
            self.output_feature_size = self.single_modal_output_feature_sizes[0]

        self.cls_head = nn.Linear(self.output_feature_size, self.class_num, bias=False)

        if self.args.corr_backbone != 'none':  # no label correlation module
            if self.args.corr_input_embed == 'one-hot':
                self.med_embedding = nn.Parameter(torch.eye(self.class_num, self.class_num),
                                                  requires_grad=self.args.corr_input_embed_learnable)
            elif self.args.corr_input_embed == 'random':
                self.med_embedding = nn.Parameter(torch.randn(self.class_num, self.args.corr_input_dim),
                                                  requires_grad=self.args.corr_input_embed_learnable)
            elif self.args.corr_input_embed == 'bert':
                p = os.path.join(self.args.root_dir, 'datasets', 'medicine', 'medicines', str(self.class_num),
                                 'word_embedding.npy')  # (c, d)
                self.med_embedding = nn.Parameter(torch.Tensor(np.load(p)),
                                                  requires_grad=self.args.corr_input_embed_learnable)  #(C, d)
                self.args.corr_input_dim = self.med_embedding.shape[-1]
            else:
                raise NotImplementedError
            self.corr_encoder = self.build_correlation_encoder()
            self.patient_projection = nn.Linear(self.output_feature_size, self.output_feature_size)
            self.medication_projection = nn.Linear(self.args.corr_hidden_dim, self.output_feature_size)
            self.post_cls_head = nn.Linear(self.output_feature_size + self.args.corr_hidden_dim, self.class_num, bias=False)
            self.relu = nn.ReLU()

        self.att_mean_pooler = pooler.MeanPooling()
        self.avgpool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.maxpool1d = torch.nn.AdaptiveMaxPool1d(output_size=1)
        self.already_freeze = False
        self.logit_scale = nn.Parameter(torch.Tensor([self.args.corr_temperature]), requires_grad=True)

        if self.args.linear_probe:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            for param in self.fusion_module.parameters():
                param.requires_grad = False
            for param in self.textual_encoder.parameters():
                param.requires_grad = False

        if args.loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()  # why i set reduction=none?? label should be one-hot format
        else:
            raise NotImplementedError
        self.contrastive_loss = nn.BCEWithLogitsLoss()

    def forward(self, data, label=None, stage='train', mode=''):
        single_modal_output_features = []

        if 'ct' in self.input:
            images = data["image"]
            batch_size, seq_len, channels, height, width = images.shape
            visual_input = images.reshape(-1, channels, height, width)
            slice_embeds = self.visual_encoder(visual_input)  # (bs * seq_len, d)
            scan_embeds = slice_embeds.reshape(batch_size, seq_len, slice_embeds.shape[-1])  # (bs, seq, d)
            visual_features, _ = self.fusion_module(scan_embeds)

            drop_visual_features = self.visual_dropout(visual_features)
            single_modal_output_features.append(drop_visual_features)

        if 'report' in self.input:
            texts = data["text"]
            batch_size = texts['input_ids'].shape[0]
            text_output = self.textual_encoder(texts, return_dict=True)
            text_embeds = text_output.last_hidden_state
            textual_enc_features = self.att_mean_pooler(text_embeds, texts['attention_mask'])
            drop_textual_features = self.textual_dropout(textual_enc_features)
            single_modal_output_features.append(drop_textual_features)

        if 'biomarker' in self.input:
            biomarkers = data["biomarker"]
            batch_size = biomarkers.shape[0]
            missing_mask = data["biomarker_missing_mask"]
            bio_enc_features, bio_enc_token_features = self.biomarker_encoder(biomarkers, missing_mask=missing_mask)
            drop_bio_features = self.biomarker_dropout(bio_enc_features)
            single_modal_output_features.append(drop_bio_features)

        assert len(single_modal_output_features) > 0, "no input data is given."

        if len(single_modal_output_features) > 1:
            fusion_features, _ = self.cross_modal_fusion_module(single_modal_output_features)
            output_feature = fusion_features
        else:
            output_feature = single_modal_output_features[0]

        patient_output = self.cls_head(self.dropout(output_feature))  # (bs, c)

        if label is not None:
            patient_loss = self.criterion(patient_output, label)

        corr_input_feature = self.med_embedding.unsqueeze(0).repeat(output_feature.shape[0], 1, 1)  # (bs, c, d_in)
        corr_output_feature = self.corr_encoder(corr_input_feature)  # (bs, c*d_out)
        corr_output_feature = corr_output_feature.unsqueeze(-1).reshape(corr_output_feature.shape[0], self.class_num,
                                                                        -1)  # (bs, c, d_out)
        medication_output = self.post_cls_head(
            self.dropout(torch.cat(
                [output_feature, self.avgpool1d(corr_output_feature.permute(0, 2, 1)).squeeze(-1)], dim=-1)))

        medication_loss = self.criterion(medication_output, label)

        total_loss = self.args.corr_loss_medication * medication_loss + \
            self.args.corr_loss_patient * patient_loss

        return {"patient_loss": patient_loss, "medication_loss": medication_loss, "loss": total_loss,
                "patient_output": patient_output, "medication_output": medication_output}

    def build_correlation_encoder(self):
        adj_path = os.path.join(self.args.root_dir, 'datasets', 'medicine', 'medicines', str(self.class_num),
                                'adj_file.npz')
        if self.args.corr_pooling != 'none':
            raise UserWarning("The corr_pooling is not none, which is not allow for ParallelMedModel.")
        net = GPGAT_alt(num_classes=self.class_num,
                        input_dim=self.args.corr_input_dim,
                        hidden_dim=self.args.corr_hidden_dim,
                        adj_file=adj_path,
                        adj_t=self.args.corr_thred,
                        nlayers=self.args.corr_nlayers,
                        nheads=self.args.corr_nheads,
                        mha_split=self.args.corr_mha_split,
                        dropout=self.args.corr_dropout,
                        pool="none")
        return net

    def build_visual_encoder(self):
        net = VisionTransformer(
            patch_size=self.args.vision_patch_size, embed_dim=self.args.vision_embed_dim,
            depth=self.args.vision_depth, num_heads=self.args.vision_num_heads, num_classes=-1,
            mlp_ratio=self.args.vision_mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pretrained=self.args.vision_pretrained, checkpoint=self.args.vision_pretrained_weight,
            freeze_pretrained_layers=self.args.vision_freeze_pretrained_layers,
        )

        return net

    def build_textual_encoder(self):
        net = TransformerEncoder(
            backbone=self.args.text_backbone,
            num_hidden_layers=self.args.text_num_hidden_layers,
            output_hidden_states=self.args.text_output_hidden_states,
            output_attentions=self.args.text_output_attentions,
            pretrained=self.args.text_pretrained,
            freeze_pretrained=self.args.text_freeze_pretrained,
            freeze_pretrained_layers=self.args.text_freeze_pretrained_layers)
        return net

    def build_structured_encoder(self):
        net = BiomarkerAttention(
            bio_num=self.args.bio_num,
            hidden_dim=self.args.bio_hidden_dim,
            nlayers=self.args.bio_nlayers,
            nheads=self.args.bio_nheads,
            feedforward_dim=self.args.bio_feedforward_dim,
            dropout=self.args.bio_dropout)
        return net

    def bulid_multimodal_fusion_module(self):
        net = AttentionFusion(
            input_dim=self.single_modal_output_feature_sizes,
            hidden_dim=self.args.fusion_hidden_dim,
            nlayers=self.args.fusion_nlayers,
            nheads=self.args.fusion_nheads,
            feedforward_dim=self.args.fusion_feedforward_dim,
            dropout=self.args.fusion_dropout,
        )
        return net

    def get_config_optim(self, lr, lrp, filter_bias_and_bn=True):
        p_params, i_params = [], []
        p_params_without_wd, i_params_without_wd = [], []

        if hasattr(self, 'visual_encoder'):
            for name, param in self.visual_encoder.named_parameters():
                if param.requires_grad:
                    if self.args.vision_load_dir != '':
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            p_params_without_wd.append(param)
                        else:
                            p_params.append(param)
                    else:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)
        if hasattr(self, 'fusion_module'):
            for name, param in self.fusion_module.named_parameters():
                if param.requires_grad:
                    if self.args.vision_load_dir != '':
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            p_params_without_wd.append(param)
                        else:
                            p_params.append(param)
                    else:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)

        if hasattr(self, 'textual_encoder'):
            for name, param in self.textual_encoder.named_parameters():
                if param.requires_grad:
                    if self.args.text_load_dir != '':
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            p_params_without_wd.append(param)
                        else:
                            p_params.append(param)
                    else:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)

        if hasattr(self, 'biomarker_encoder'):
            for name, param in self.biomarker_encoder.named_parameters():
                if param.requires_grad:
                    if self.args.bio_load_dir != '':
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            p_params_without_wd.append(param)
                        else:
                            p_params.append(param)
                    else:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)

        if hasattr(self, 'cross_modal_fusion_module'):
            for name, param in self.cross_modal_fusion_module.named_parameters():
                if param.requires_grad:
                    if self.args.fusion_load_dir != '':
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            p_params_without_wd.append(param)
                        else:
                            p_params.append(param)
                    else:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)

        if hasattr(self, 'corr_encoder'):
            for name, param in self.corr_encoder.named_parameters():
                if param.requires_grad:
                    if self.args.corr_load_dir != '':
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            p_params_without_wd.append(param)
                        else:
                            p_params.append(param)
                    else:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)

        if hasattr(self, 'cls_head'):
            for name, param in self.cls_head.named_parameters():
                if param.requires_grad:
                    if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                        i_params_without_wd.append(param)
                    else:
                        i_params.append(param)
        if hasattr(self, 'post_cls_head'):
            for name, param in self.post_cls_head.named_parameters():
                if param.requires_grad:
                    if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                        i_params_without_wd.append(param)
                    else:
                        i_params.append(param)
        if hasattr(self, 'med_embedding'):
            if self.med_embedding.requires_grad:
                i_params.append(self.med_embedding)

        if hasattr(self, 'patient_projection'):
            for name, param in self.patient_projection.named_parameters():
                if param.requires_grad:
                    if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                        i_params_without_wd.append(param)
                    else:
                        i_params.append(param)
        if hasattr(self, 'medication_projection'):
            for name, param in self.medication_projection.named_parameters():
                if param.requires_grad:
                    if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                        i_params_without_wd.append(param)
                    else:
                        i_params.append(param)
        if hasattr(self, 'logit_scale'):
            if self.logit_scale.requires_grad:
                i_params.append(self.logit_scale)

        return [
            {'params': p_params, 'lr': lrp},
            {'params': i_params, 'lr': lr},
            {'params': p_params_without_wd, 'lr': lrp, 'weight_decay': 0.},
            {'params': i_params_without_wd, 'lr': lr, 'weight_decay': 0.},
        ]

    def get_two_stage_optim(self, lr, lrp, stage, filter_bias_and_bn=True):
        p_params, i_params = [], []
        p_params_without_wd, i_params_without_wd = [], []
        if stage == 'feature':
            if hasattr(self, 'visual_encoder'):
                for name, param in self.visual_encoder.named_parameters():
                    if param.requires_grad:
                        if self.args.vision_load_dir != '':
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                p_params_without_wd.append(param)
                            else:
                                p_params.append(param)
                        else:
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                i_params_without_wd.append(param)
                            else:
                                i_params.append(param)
            if hasattr(self, 'fusion_module'):
                for name, param in self.fusion_module.named_parameters():
                    if param.requires_grad:
                        if self.args.vision_load_dir != '':
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                p_params_without_wd.append(param)
                            else:
                                p_params.append(param)
                        else:
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                i_params_without_wd.append(param)
                            else:
                                i_params.append(param)

            if hasattr(self, 'textual_encoder'):
                for name, param in self.textual_encoder.named_parameters():
                    if param.requires_grad:
                        if self.args.text_load_dir != '':
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                p_params_without_wd.append(param)
                            else:
                                p_params.append(param)
                        else:
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                i_params_without_wd.append(param)
                            else:
                                i_params.append(param)

            if hasattr(self, 'biomarker_encoder'):
                for name, param in self.biomarker_encoder.named_parameters():
                    if param.requires_grad:
                        if self.args.bio_load_dir != '':
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                p_params_without_wd.append(param)
                            else:
                                p_params.append(param)
                        else:
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                i_params_without_wd.append(param)
                            else:
                                i_params.append(param)

            if hasattr(self, 'cross_modal_fusion_module'):
                for name, param in self.cross_modal_fusion_module.named_parameters():
                    if param.requires_grad:
                        if self.args.fusion_load_dir != '':
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                p_params_without_wd.append(param)
                            else:
                                p_params.append(param)
                        else:
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                i_params_without_wd.append(param)
                            else:
                                i_params.append(param)
            if hasattr(self, 'cls_head'):
                for name, param in self.cls_head.named_parameters():
                    if param.requires_grad:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)
        elif stage == 'correlation':
            if hasattr(self, 'corr_encoder'):
                for name, param in self.corr_encoder.named_parameters():
                    if param.requires_grad:
                        if self.args.corr_load_dir != '':
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                p_params_without_wd.append(param)
                            else:
                                p_params.append(param)
                        else:
                            if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                                i_params_without_wd.append(param)
                            else:
                                i_params.append(param)
            if hasattr(self, 'post_cls_head'):
                for name, param in self.post_cls_head.named_parameters():
                    if param.requires_grad:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)
            if hasattr(self, 'med_embedding'):
                if self.med_embedding.requires_grad:
                    i_params.append(self.med_embedding)
            if hasattr(self, 'patient_projection'):
                for name, param in self.patient_projection.named_parameters():
                    if param.requires_grad:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)
            if hasattr(self, 'medication_projection'):
                for name, param in self.medication_projection.named_parameters():
                    if param.requires_grad:
                        if filter_bias_and_bn and (len(param.shape) == 1 or name.endswith(".bias")):
                            i_params_without_wd.append(param)
                        else:
                            i_params.append(param)
            if hasattr(self, 'logit_scale'):
                if self.logit_scale.requires_grad:
                    i_params.append(self.logit_scale)
        else:
            raise NotImplementedError("only support stage in [feature, correlation].")

        return [
            {'params': p_params, 'lr': lrp},
            {'params': i_params, 'lr': lr},
            {'params': p_params_without_wd, 'lr': lrp, 'weight_decay': 0.},
            {'params': i_params_without_wd, 'lr': lr, 'weight_decay': 0.},
        ]

    def freeze_feature_extraction_net(self):
        if hasattr(self, 'visual_encoder'):
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        if hasattr(self, 'fusion_module'):
            for param in self.fusion_module.parameters():
                param.requires_grad = False
        if hasattr(self, 'textual_encoder'):
            for param in self.textual_encoder.parameters():
                param.requires_grad = False
        if hasattr(self, 'biomarker_encoder'):
            for param in self.biomarker_encoder.parameters():
                param.requires_grad = False
        if hasattr(self, 'cross_modal_fusion_module'):
            for param in self.cross_modal_fusion_module.parameters():
                param.requires_grad = False
        if hasattr(self, 'cls_head'):
            for param in self.cls_head.parameters():
                param.requires_grad = False
        self.already_freeze = True
        print("Freeze the feature extraction net.")
