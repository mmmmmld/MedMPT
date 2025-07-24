# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/05/14 21:01

import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, confusion_matrix, \
    precision_score, recall_score

def compute_multi_class_metrics(y_true, y_score):
    # y_true: torch.Tensor, (N,)
    # y_score: torch.Tensor, (N, c)
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    if isinstance(y_true, type([])):
        y_true = np.array(y_true).reshape((-1))

    if isinstance(y_score, type([])):
        y_score = np.array(y_score).reshape((-1))

    if len(y_true.shape) == 2 and y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)

    y_pred = np.argmax(y_score, axis=1)
    if y_score.shape[-1] == 2:
        y_score = y_score[:, -1]
    v_metrics = {}
    v_metrics['acc'] = accuracy_score(y_true, y_pred)
    v_metrics['auc'] = roc_auc_score(y_true, y_score, multi_class="ovr")

    return v_metrics
