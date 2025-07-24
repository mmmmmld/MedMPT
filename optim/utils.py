# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/04/25 16:35


def add_lr_weight_decay(model, args, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': args.weight_decay}]


def add_lr_weight_decay_in_modality(model, args, skip_list=()):
    vision_encoder_decay, vision_encoder_no_decay = [], []
    vision_decoder_decay, vision_decoder_no_decay = [], []
    vision_fusion_decay, vision_fusion_no_decay = [], []
    text_encoder_decay, text_encoder_no_decay = [], []
    text_decoder_decay, text_decoder_no_decay = [], []
    other_decay, other_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'vision_encoder' in name:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                vision_encoder_no_decay.append(param)
            else:
                vision_encoder_decay.append(param)
            continue
        if 'vision_decoder' in name:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                vision_decoder_no_decay.append(param)
            else:
                vision_decoder_decay.append(param)
            continue
        if 'vision_fusion' in name:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                vision_fusion_no_decay.append(param)
            else:
                vision_fusion_decay.append(param)
            continue
        if 'text_encoder' in name:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                text_encoder_no_decay.append(param)
            else:
                text_encoder_decay.append(param)
            continue
        if 'text_decoder' in name:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                text_decoder_no_decay.append(param)
            else:
                text_decoder_decay.append(param)
            continue

        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            other_no_decay.append(param)
        else:
            other_decay.append(param)


    return [
        {'params': vision_encoder_no_decay, 'weight_decay': 0, 'lr': args.vision_encoder_lr},
        {'params': vision_encoder_decay, 'weight_decay': args.weight_decay, 'lr': args.vision_encoder_lr},
        {'params': vision_fusion_no_decay, 'weight_decay': 0, 'lr': args.vision_fusion_lr},
        {'params': vision_fusion_decay, 'weight_decay': args.weight_decay, 'lr': args.vision_fusion_lr},
        {'params': vision_decoder_no_decay, 'weight_decay': 0, 'lr': args.vision_decoder_lr},
        {'params': vision_decoder_decay, 'weight_decay': args.weight_decay, 'lr': args.vision_decoder_lr},
        {'params': text_encoder_no_decay, 'weight_decay': 0, 'lr': args.text_encoder_lr},
        {'params': text_encoder_decay, 'weight_decay': args.weight_decay, 'lr': args.text_encoder_lr},
        {'params': text_decoder_no_decay, 'weight_decay': 0, 'lr': args.text_decoder_lr},
        {'params': text_decoder_decay, 'weight_decay': args.weight_decay, 'lr': args.text_decoder_lr},
        {'params': other_no_decay, 'weight_decay': 0., 'lr': args.lr},
        {'params': other_decay, 'weight_decay': args.weight_decay, 'lr': args.lr}
    ]