# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/30 17:03

import copy
import os
import json
import time
import random
import argparse
import numpy as np
import tqdm
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import link
from utils.basic import *
from metrics.multilabel import compute_multi_label_metrics
from tokenizer import tokenizer_dict
from datasets.medicine_dataset import MedDataset
from models.medication_model import ParallelMedModel
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_models = {
    "ParallelMed": ParallelMedModel,
}

_checkpoints = {
    0: "",
    }

def get_parse():
    parser = argparse.ArgumentParser()
    # default
    default_group = parser.add_argument_group(title='Default experiment options')
    default_group.add_argument("--root_dir", type=str, default="/root/maliangdi")
    default_group.add_argument("--experiment", type=str, default="medication")
    default_group.add_argument("--exp_name", type=str, default="debug")
    default_group.add_argument("--output_dir", type=str, default="./output")
    default_group.add_argument("--save_dir", type=str, default="")
    default_group.add_argument("--ckpt_path", type=str, default="")
    default_group.add_argument("--event_path", type=str, default="")
    default_group.add_argument("--result_path", type=str, default="")

    # model
    model_group = parser.add_argument_group(title='Model options')
    model_group.add_argument("--input", type=str, default="ct,report,biomarker",
                        help='input data modal: ct|report|biomarker')
    model_group.add_argument("--model", type=str, default="v5")
    model_group.add_argument("--resume", type=str, default="")
    model_group.add_argument("--load_dir", type=str, default="")
    model_group.add_argument("--load_dir_ind", type=int, default=-1)
    model_group.add_argument("--load_prefix", type=str, default="optimal")
    model_group.add_argument("--load_postfix", type=str, default="medication")
    ## vision
    model_group.add_argument("--vision_backbone", type=str, default="vit-b/16")
    model_group.add_argument("--vision_pretrained", type=str2bool, default=False)
    model_group.add_argument("--vision_pretrained_weight", type=str, default='imagenet')
    model_group.add_argument("--vision_freeze_pretrained", type=str2bool, default=False)
    model_group.add_argument("--vision_freeze_pretrained_layers", type=str2list, default=[],
                             help="list of layers stated as: 1,2,3")
    model_group.add_argument("--vision_depth", type=int, default=12)
    model_group.add_argument("--vision_embed_dim", type=int, default=768)
    model_group.add_argument("--vision_patch_size", type=int, default=16)
    model_group.add_argument("--vision_num_heads", type=int, default=12)
    model_group.add_argument("--vision_mlp_ratio", type=float, default=4)
    model_group.add_argument("--vision_dropout", type=float, default=0.1)
    ## text
    model_group.add_argument("--text_backbone", type=str, default="chinesebert")
    model_group.add_argument("--text_pretrained", type=str2bool, default=False)
    model_group.add_argument("--text_num_hidden_layers", type=int, default=6)
    model_group.add_argument("--text_output_hidden_states", type=str2bool, default=True)
    model_group.add_argument("--text_output_attentions", type=str2bool, default=True)
    model_group.add_argument("--text_freeze_pretrained", type=str2bool, default=False)
    model_group.add_argument("--text_freeze_pretrained_layers", type=str2list, default=[])
    model_group.add_argument("--text_pooling", type=str, default="mean")
    model_group.add_argument("--text_dropout", type=float, default=0.1)
    ## biomarker
    model_group.add_argument("--bio_backbone", type=str, default="transformer")
    model_group.add_argument("--bio_freeze", type=str2bool, default=False)
    model_group.add_argument("--bio_freeze_layers", type=str2list, default="", help="list of layers stated as: 1,2,3")
    model_group.add_argument("--bio_hidden_dim", type=int, default=768)
    model_group.add_argument("--bio_nlayers", type=int, default=4)
    model_group.add_argument("--bio_nheads", type=int, default=8)
    model_group.add_argument("--bio_feedforward_dim", type=int, default=2048)
    model_group.add_argument("--bio_dropout", type=float, default=0.1)
    ## multimodal fusion
    model_group.add_argument("--fusion_backbone", type=str, default="transformer")
    model_group.add_argument("--fusion_hidden_dim", type=int, default=768)
    model_group.add_argument("--fusion_nlayers", type=int, default=2)
    model_group.add_argument("--fusion_nheads", type=int, default=4)
    model_group.add_argument("--fusion_feedforward_dim", type=int, default=1024)
    model_group.add_argument("--fusion_dropout", type=float, default=0.1)
    ## label correlation
    model_group.add_argument("--corr_load_embedding", type=str2bool, default=True)
    model_group.add_argument("--corr_backbone", type=str, default="gpgat_alt",
                             choices=["gat", "gpgat_alt", "gpgat_seq", "none"])
    model_group.add_argument("--corr_mha_split", type=str2bool, default=False,
                             help="split the hidden size by num of heads")
    model_group.add_argument("--corr_thred", type=float, default=0.5, help="threshold of the adjacent matrix")
    model_group.add_argument("--corr_nlayers", type=int, default=2)
    model_group.add_argument("--corr_nheads", type=str2list, default="8,1")
    model_group.add_argument("--corr_input_embed", type=str, default="one-hot", choices=["one-hot", "bert", "random"])
    model_group.add_argument("--corr_input_embed_learnable", type=str2bool, default=True)
    model_group.add_argument("--corr_input_dim", type=int, default=256, help="invalid if corr_embed == one-hot")
    model_group.add_argument("--corr_hidden_dim", type=int, default=256)
    model_group.add_argument("--corr_pooling", type=str, default="none", choices=["avg", "sum", "max", "none"])
    model_group.add_argument("--corr_multimlp", type=str2bool, default=False)
    model_group.add_argument("--corr_use_nonlinear", type=str2bool, default=False)
    model_group.add_argument("--corr_dropout", type=float, default=0.1)
    model_group.add_argument("--corr_loss_patient", type=float, default=1)
    model_group.add_argument("--corr_loss_medication", type=float, default=1)
    model_group.add_argument("--corr_loss_contrastive", type=float, default=0.1)
    model_group.add_argument("--corr_temperature", type=float, default=0.07,
                             help="temperature coefficient, smaller for hard sample mining (dangerous)")
    model_group.add_argument("--corr_target", type=str, default="medication")

    # dataset
    dataset_group = parser.add_argument_group(title='Dataset options')
    dataset_group.add_argument("--dataset_root", type=str, default="/root/maliangdi/datasets/medicine")
    dataset_group.add_argument("--buffer_root", type=str, default="/buffer/maliangdi/medicine")
    dataset_group.add_argument("--num_classes", type=int, default=55, help="num of medication types")
    dataset_group.add_argument("--val_pct", type=float, default=1.0, help="percentage of valid dataset")
    ## vision
    dataset_group.add_argument("--slice_num", type=int, default=32)
    dataset_group.add_argument("--image_size", type=int, default=224)

    ## biomarker
    dataset_group.add_argument("--bio_num", type=int, default=100)
    dataset_group.add_argument("--bio_v_null", type=float, default=0, help="missing value")
    dataset_group.add_argument("--bio_discrete", type=str2bool, default=False)
    dataset_group.add_argument("--bio_normalize", type=str2bool, default=False)
    dataset_group.add_argument("--bio_embed", type=str2bool, default=False)

    # tokenizer
    tokenizer_group = parser.add_argument_group(title='Tokenizer options')
    tokenizer_group.add_argument("--tokenizer", type=str, default='chinesebert')
    tokenizer_group.add_argument("--text_max_length", type=int, default=100)

    # loss
    loss_group = parser.add_argument_group(title='Loss options')
    loss_group.add_argument("--loss", type=str, default="bce", help='bce')

    # training, valid, evaluation
    default_group.add_argument("--batch_size", type=int, default=2, help="batch size per gpu")
    default_group.add_argument("--num_workers", type=int, default=5)
    default_group.add_argument("--pin_memory", type=str2bool, default=True)
    default_group.add_argument("--non_blocking", type=str2bool, default=True)

    # other
    default_group.add_argument("--pred_thred", type=float, default=0.5)
    default_group.add_argument("--pred_topk", type=int, default=-1)
    default_group.add_argument("--valid_index", type=str, default="auc")
    default_group.add_argument("--valid_print_freq", type=int, default=20)
    default_group.add_argument("--distributed", type=str2bool, default=True)
    default_group.add_argument("--local_rank", type=int, default=0, help="node rank for distributed training")
    default_group.add_argument("--dist_url", type=str, default="env://")
    default_group.add_argument("--seed", type=int, default=66, help="random seed")
    args = parser.parse_args()
    return args

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def initialize(args):
    args.exp_name = getTime() + '_TEST_' + args.exp_name

    assert args.load_dir == ""
    load_dir = _checkpoints[args.load_dir_ind]
    train_opt = json.load(open(os.path.join(load_dir, 'train_options.json')))
    args.load_dir = os.path.join(load_dir, 'checkpoints', args.load_prefix + '_model_' + args.load_postfix + '.pth')

    target_keys = ['vision_', 'text_', 'bio_', 'fusion_', 'corr_']
    skip_keys = ['_pretrained']
    recover_keys = []
    args.input = train_opt['input']
    args.model = train_opt['model']
    args.linear_probe = train_opt['linear_probe']
    recover_keys.extend(['input', 'model'])
    for key in train_opt.keys():
        is_skip = False
        for skip_k in skip_keys:
            if skip_k in key:
                is_skip = True
                break
        if is_skip:
            continue
        for target_k in target_keys:
            if target_k in key and hasattr(args, key):
                setattr(args, key, train_opt[key])
                recover_keys.append(key)
                break

    print("=> load options from checkpoint: ", recover_keys)

    args.output_dir = os.path.join(args.output_dir, args.experiment, args.input)
    if args.linear_probe:
        args.output_dir = os.path.join(args.output_dir, 'freeze_pretrained', 'test')
    if args.save_dir != "":
        args.output_dir = os.path.join(args.output_dir, args.save_dir)
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    ckpt_path = os.path.join(args.output_dir, 'checkpoints')
    event_path = os.path.join(args.output_dir, 'events')
    result_path = os.path.join(args.output_dir, 'results')
    args.ckpt_path = ckpt_path
    args.event_path = event_path
    args.result_path = result_path

    argsDict = args.__dict__
    if link.is_main_process():
        makedir(args.output_dir)
        makedir(args.ckpt_path)
        makedir(args.event_path)
        makedir(args.result_path)
        with open(os.path.join(args.output_dir, 'test_options.json'), 'w', encoding='utf-8') as f:
            json.dump(argsDict, f)
        show_options = '------------------ test options ------------------' + '\n'
        for eachArg, value in argsDict.items():
            show_options += eachArg + ' : ' + str(value) + '\n'
        show_options += '------------------- end -------------------'
        with open(os.path.join(args.output_dir, 'test_options.txt'), 'w', encoding='utf-8') as f:
            f.write(show_options)
        print(show_options)
        save_code('.', os.path.join(args.output_dir, 'code.zip'))


def main(args):
    assert link.get_world_size() == 1
    ### env initialize ###
    link.init_distributed_mode(args)
    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else 'cpu'

    ### experiment initialize ###
    init_seeds(args.seed + link.get_rank())
    initialize(args)

    ### build tokenizer ###
    print(f"=> Build {args.tokenizer} tokenizer")
    tokenizer_path = tokenizer_dict[args.tokenizer]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    ### build model and optimizer ###
    print(f"=> Build ParallelMed Model")
    model = ParallelMedModel(args, stage="test")
    load_ckpt = torch.load(args.load_dir, map_location='cpu')
    for k in list(load_ckpt.keys()):
        if k.startswith("module."):
            load_ckpt[k[len("module."):]] = load_ckpt[k]
            del load_ckpt[k]
    model.load_state_dict(load_ckpt, strict=True)
    print(f"load checkpoint from {args.load_dir}")

    model = model.to(device)

    ### build dataset ###
    print(f'=> Build test dataset')
    test_dataset = MedDataset(input=args.input, class_num=args.num_classes, image_size=args.image_size,
                              image_slice_num=args.slice_num, bio_num=args.bio_num, bio_v_null=args.bio_v_null,
                              bio_discrete=args.bio_discrete, bio_normalize=args.bio_normalize, pct=args.val_pct,
                              stage="test", dataset_root=args.dataset_root,
                              buffer_root=args.buffer_root)
    print(f'+ Test data num = {len(test_dataset)}')
    test_sampler = DistributedSampler(test_dataset) if link.get_world_size() > 1 else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)

    valid_n_iter = 0

    print('===============START TEST================')

    model.eval()
    valid_n_iter += 1
    buffer_valid_scores, buffer_valid_targets = [], []
    valid_ids = []
    model.eval()
    for step, data in tqdm.tqdm(enumerate(test_loader)):
        input = {}
        name_id = data["name"]
        valid_ids.extend(name_id)  # should be extend
        if 'ct' in args.input:
            input['image'] = data["image"].to(device, non_blocking=args.non_blocking)
        if 'biomarker' in args.input:
            input['biomarker'] = data["biomarker"].to(device, non_blocking=args.non_blocking)
            input['biomarker_missing_mask'] = data["biomarker_missing_mask"].to(device, non_blocking=args.non_blocking)
        if 'report' in args.input:
            enc_tokens = tokenizer(data["caption"], padding="max_length",
                                   max_length=args.text_max_length,
                                   truncation=True, return_tensors='pt')
            for k, v in enc_tokens.items():
                enc_tokens[k] = v.to(device, non_blocking=args.non_blocking)
            input['text'] = enc_tokens
        label = data['one_hot_label'].to(device, non_blocking=args.non_blocking)

        with torch.no_grad():
            output = model(input, label)  # (bs, c)

        scores = output["medication_output"]
        buffer_valid_scores.append(torch.sigmoid(scores.detach()))  # (bs, c)
        buffer_valid_targets.append(label.detach())

    valid_scores = torch.cat(buffer_valid_scores, dim=0)  # (N, C)
    valid_targets = torch.cat(buffer_valid_targets, dim=0)  # (N, C)  one-hot

    valid_metric_medication = compute_multi_label_metrics(
        valid_targets, valid_scores, thred=args.pred_thred, topk=args.pred_topk
    )

    print_info = f'Test data num: {len(valid_ids)}\n'
    print_info += f'Test Results: '
    for k, v in valid_metric_medication.items():
        print_info += f'{k.upper()}={100 * np.mean(v).item():.2f}, '
    print_info = print_info[:-2]
    print(print_info)
    print(f"====================End====================")
    if link.is_main_process():
        with open(os.path.join(args.output_dir, 'test_results.txt'), 'w', encoding='utf-8') as f:
            f.write(print_info)
        try:
            postfix = f",f1={100 * np.mean(valid_metric_medication['f1']):.2f}"
            os.rename(args.output_dir, args.output_dir + postfix)
        except:
            pass

    # torch.cuda.synchronize()


if __name__ == '__main__':
    args = get_parse()
    main(args)
