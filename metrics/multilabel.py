# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/06/23 21:04


import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def predict_topk(scores, k=3):
    # array in (N, C)
    n_sample, n_class = scores.shape
    preds_scalar = np.argsort(scores, axis=-1)[:, -k:]
    preds = np.zeros_like(scores)  # [n_sample, n_class]
    for sample in range(n_sample):
        for c in preds_scalar[sample]:
            preds[sample, c] = 1
    return preds


def compute_multi_label_metrics(y_true, y_scores, thred=0.5, topk=-1, return_test=False):
    # y_true: (N, c), y_scores: (N, c), multi-hot matrix
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, type([])):
        y_true = np.array(y_true)

    if isinstance(y_scores, type([])):
        y_scores = np.array(y_scores)

    assert len(y_true.shape) == 2
    n_sample, n_class = y_true.shape
    if topk > 0:
        y_preds = predict_topk(y_scores, topk)  # [n_sample, n_class]
    else:
        y_preds = np.where(y_scores >= thred, 1, 0)  # array, (n, c)

    v_metrics = {}
    
    f1_scores_sample = f1_score(y_true, y_preds, average='samples')
    if isinstance(f1_scores_sample, np.float64):
        v_metrics['f1'] = [f1_scores_sample]
    else:
        v_metrics['f1'] = list(f1_scores_sample)
    
    rec_scores_sample = recall_score(y_true, y_preds, average='samples')
    if isinstance(rec_scores_sample, np.float64):
        v_metrics['recall'] = [rec_scores_sample]
    else:
        v_metrics['recall'] = list(rec_scores_sample)

    if return_test:
        return v_metrics

    topk_list = [k for k in [6, 8, 10, 12, 15, 20] if k <= n_class]
    for k in topk_list:
        tmp_preds = predict_topk(y_scores, k)  # [n_sample, n_class]
        rec_topk = recall_score(y_true, tmp_preds, average=None)
        rec_topk_sample = recall_score(y_true, tmp_preds, average='samples')
        if isinstance(rec_topk, np.float64):
            v_metrics[f'recall@{k}'] = [rec_topk]
        else:
            v_metrics[f'recall@{k}'] = list(rec_topk)
        if isinstance(rec_topk_sample, np.float64):
            v_metrics[f'recall@{k}'] = [rec_topk_sample]
        else:
            v_metrics[f'recall@{k}'] = list(rec_topk_sample)

    return v_metrics
