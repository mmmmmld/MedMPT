# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/28 21:18

import numpy as np
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge


def compute_metrics(gts, res, **kwargs):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    eval_gts = None
    eval_res = None
    caption_record = ''
    if isinstance(gts, dict):
        eval_gts = gts
    if isinstance(res, dict):
        eval_res = res

    if isinstance(gts, list) and isinstance(gts[0], str):
        eval_gts = {i: [g] for i, g in enumerate(gts)}
    if isinstance(res, list) and isinstance(res[0], str):
        eval_res = {i: [r] for i, r in enumerate(res)}

    if isinstance(gts, np.ndarray):
        gts = torch.Tensor(gts)
    if isinstance(res, np.ndarray):
        res = torch.Tensor(res)

    if isinstance(gts, torch.Tensor) or isinstance(res, torch.Tensor):
        gts_captions = []
        res_captions = []
        try:
            tokenizer = kwargs['tokenizer']
        except KeyError:
            raise ValueError

        if 'img_path' in kwargs.keys():
            path_list = kwargs['img_path']
            has_path = True
        else:
            has_path = False

        for bs in range(gts.shape[0]):
            real_token_ids = gts[bs]
            real_tokens = tokenizer.convert_ids_to_tokens(real_token_ids)[1:-1]
            real_captions = tokenizer.convert_tokens_to_string(real_tokens)  #.split('[SEP]')[0]

            gen_token_ids = res[bs]
            gen_tokens = tokenizer.convert_ids_to_tokens(gen_token_ids)[:-1]
            gen_captions = tokenizer.convert_tokens_to_string(gen_tokens).split('[SEP]')[0]  # + '[SEP]'

            gts_captions.append(real_captions)
            res_captions.append(gen_captions)

            if has_path:
                caption_record += '【Image Path】 ' + path_list[bs] + '\n'
            caption_record += '【Ground Truth】 ' + real_captions.replace(' ', '') + '\n'
            caption_record += '【Generated Caption】 ' + gen_captions.replace(' ', '') + '\n'
            caption_record += '--------------------------------------------\n'

        eval_gts = {i: [g] for i, g in enumerate(gts_captions)}
        eval_res = {i: [r] for i, r in enumerate(res_captions)}

    if eval_gts is None or eval_res is None:
        raise ValueError('The given ground truth caption and generated caption have undesired format.')

    # print(type(eval_gts))
    # print(type(eval_res))
    # exit()
    # Set up scorers
    if kwargs.get('target_scorer', False):
        scorers = []
        if 'bleu' in kwargs['target_scorer']:
            scorers.append((Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]))
        if 'meteor' in kwargs['target_scorer']:
            scorers.append((Meteor(), "METEOR"))
        if 'rouge_l' in kwargs['target_scorer']:
            scorers.append((Rouge(), "ROUGE_L"))
    else:
        scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]
    eval_metrics = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(eval_gts, eval_res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(eval_gts, eval_res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_metrics[m] = sc
        else:
            eval_metrics[method] = score

    return eval_metrics, caption_record

