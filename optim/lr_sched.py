# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/04/23 21:30

import math

def adjust_learning_rate(optimizer, step, total_step, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    """step is the global step during training"""
    lr_groups = []
    for param_group in optimizer.param_groups:
        base_lr = param_group["base_lr"]
        if step < args.warmup_steps:
            lr = base_lr * step / args.warmup_steps
        else:
            lr = args.min_lr + (base_lr - args.min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (step - args.warmup_steps) / (total_step - args.warmup_steps)))

        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

        lr_groups.append(lr)

    return lr_groups