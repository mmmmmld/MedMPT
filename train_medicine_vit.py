# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/30 17:03


import json
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
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

_checkpoints = {}

def get_parse():
    parser = argparse.ArgumentParser()
    # default
    default_group = parser.add_argument_group(title='Default experiment options')
    default_group.add_argument("--root_dir", type=str, default="/root/maliangdi")
    default_group.add_argument("--experiment", type=str, default="medication")
    default_group.add_argument("--linear_probe", action="store_true", default=False)
    default_group.add_argument("--evaluate", action="store_true", default=False)
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
    ## vision
    model_group.add_argument("--vision_load_dir", type=str, default="")
    model_group.add_argument("--vision_load_dir_ind", type=int, default=-1)
    model_group.add_argument("--vision_load_epoch", type=str, default="0")
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
    model_group.add_argument("--text_load_dir", type=str, default="")
    model_group.add_argument("--text_load_dir_ind", type=int, default=-1)
    model_group.add_argument("--text_load_epoch", type=str, default="0")
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
    model_group.add_argument("--bio_load_dir", type=str, default="")
    model_group.add_argument("--bio_load_dir_ind", type=int, default=-1)
    model_group.add_argument("--bio_load_epoch", type=str, default="optimal")
    model_group.add_argument("--bio_backbone", type=str, default="transformer")
    model_group.add_argument("--bio_freeze", type=str2bool, default=False)
    model_group.add_argument("--bio_freeze_layers", type=str2list, default="", help="list of layers stated as: 1,2,3")
    model_group.add_argument("--bio_hidden_dim", type=int, default=768)
    model_group.add_argument("--bio_nlayers", type=int, default=4)
    model_group.add_argument("--bio_nheads", type=int, default=8)
    model_group.add_argument("--bio_feedforward_dim", type=int, default=2048)
    model_group.add_argument("--bio_dropout", type=float, default=0.1)
    ## multimodal fusion
    model_group.add_argument("--fusion_load_dir", type=str, default="")
    model_group.add_argument("--fusion_load_dir_ind", type=int, default=-1)
    model_group.add_argument("--fusion_load_epoch", type=str, default="optimal")
    model_group.add_argument("--fusion_backbone", type=str, default="transformer")
    model_group.add_argument("--fusion_hidden_dim", type=int, default=768)
    model_group.add_argument("--fusion_nlayers", type=int, default=2)
    model_group.add_argument("--fusion_nheads", type=int, default=4)
    model_group.add_argument("--fusion_feedforward_dim", type=int, default=1024)
    model_group.add_argument("--fusion_dropout", type=float, default=0.1)
    ## label correlation
    model_group.add_argument("--corr_load_dir", type=str, default="")
    model_group.add_argument("--corr_load_dir_ind", type=int, default=-1)
    model_group.add_argument("--corr_load_epoch", type=str, default="optimal")
    model_group.add_argument("--corr_load_embedding", type=str2bool, default=True)
    model_group.add_argument("--corr_backbone", type=str, default="gpgat_alt",
                             choices=["gat", "gpgat_alt", "gpgat_seq", "none"])
    model_group.add_argument("--corr_mha_split", type=str2bool, default=False,
                             help="split the hidden size by num of heads")
    model_group.add_argument("--corr_thred", type=float, default=0.5, help="threshold of the adjacent matrix")
    model_group.add_argument("--corr_nlayers", type=int, default=2)
    model_group.add_argument("--corr_nheads", type=str2list, default="8,1")
    model_group.add_argument("--corr_input_embed", type=str, default="bert", choices=["one-hot", "bert", "random"])
    model_group.add_argument("--corr_input_embed_learnable", type=str2bool, default=True)
    model_group.add_argument("--corr_input_dim", type=int, default=256, help="invalid if corr_embed == one-hot")
    model_group.add_argument("--corr_hidden_dim", type=int, default=256)
    model_group.add_argument("--corr_pooling", type=str, default="none", choices=["avg", "sum", "max", "none"])
    model_group.add_argument("--corr_multimlp", type=str2bool, default=False)
    model_group.add_argument("--corr_use_nonlinear", type=str2bool, default=False)
    model_group.add_argument("--corr_dropout", type=float, default=0.1)
    model_group.add_argument("--corr_loss_patient", type=float, default=2.0)
    model_group.add_argument("--corr_loss_medication", type=float, default=1)
    model_group.add_argument("--corr_loss_contrastive", type=float, default=0.0)
    model_group.add_argument("--corr_temperature", type=float, default=0.07,
                             help="temperature coefficient, smaller for hard sample mining (dangerous)")
    model_group.add_argument("--corr_target", type=str, default="medication")

    # dataset
    dataset_group = parser.add_argument_group(title='Dataset options')
    dataset_group.add_argument("--dataset_root", type=str, default="/root/maliangdi/datasets/medicine")
    dataset_group.add_argument("--buffer_root", type=str, default="/buffer/maliangdi/medicine")
    dataset_group.add_argument("--num_classes", type=int, default=55, help="num of medication types")
    dataset_group.add_argument("--train_pct", type=float, default=1.0, help="precentage of training dataset")
    dataset_group.add_argument("--val_pct", type=float, default=1.0, help="percentage of valid dataset")
    ## vision
    dataset_group.add_argument("--slice_num", type=int, default=32)
    dataset_group.add_argument("--image_size", type=int, default=224)
    ## text
    dataset_group.add_argument("--caption_load_ind", type=int, default=-1)
    ## biomarker
    dataset_group.add_argument("--bio_num", type=int, default=100)
    dataset_group.add_argument("--bio_v_null", type=float, default=0, help="missing value")
    dataset_group.add_argument("--bio_discrete", type=str2bool, default=False)
    dataset_group.add_argument("--bio_normalize", type=str2bool, default=True)
    dataset_group.add_argument("--bio_embed", type=str2bool, default=False)

    # tokenizer
    tokenizer_group = parser.add_argument_group(title='Tokenizer options')
    tokenizer_group.add_argument("--tokenizer", type=str, default='chinesebert')
    tokenizer_group.add_argument("--text_max_length", type=int, default=100)

    # loss
    loss_group = parser.add_argument_group(title='Loss options')
    loss_group.add_argument("--loss", type=str, default="bce")

    # optimizer
    optimizer_group = parser.add_argument_group(title='Optimizer options')
    optimizer_group.add_argument("--optimizer", type=str, default='adamw')
    optimizer_group.add_argument("--lr", type=float, default=0.00001)
    optimizer_group.add_argument("--pretrained_lr", type=float, default=0.000001)
    optimizer_group.add_argument("--weight_decay", type=float, default=0.0001)
    optimizer_group.add_argument("--eps", type=float, default=1e-8)
    optimizer_group.add_argument("--beta0", type=float, default=0.9)
    optimizer_group.add_argument("--beta1", type=float, default=0.999)
    optimizer_group.add_argument("--momentum", type=float, default=0.9)
    optimizer_group.add_argument("--filter_bias_and_bn", type=str2bool, default=True)

    optimizer_group.add_argument("--netf_optimizer", type=str, default='adamw')
    optimizer_group.add_argument("--netf_lr", type=float, default=0.00001)
    optimizer_group.add_argument("--netf_pretrained_lr", type=float, default=0.000001)
    optimizer_group.add_argument("--netf_weight_decay", type=float, default=0.0001)
    optimizer_group.add_argument("--netf_eps", type=float, default=1e-8)
    optimizer_group.add_argument("--netf_beta0", type=float, default=0.9)
    optimizer_group.add_argument("--netf_beta1", type=float, default=0.999)
    optimizer_group.add_argument("--netf_momentum", type=float, default=0.9)

    optimizer_group.add_argument("--netc_optimizer", type=str, default='adamw')
    optimizer_group.add_argument("--netc_lr", type=float, default=0.00001)
    optimizer_group.add_argument("--netc_pretrained_lr", type=float, default=0.000001)
    optimizer_group.add_argument("--netc_weight_decay", type=float, default=0.0001)
    optimizer_group.add_argument("--netc_eps", type=float, default=1e-8)
    optimizer_group.add_argument("--netc_beta0", type=float, default=0.9)
    optimizer_group.add_argument("--netc_beta1", type=float, default=0.999)
    optimizer_group.add_argument("--netc_momentum", type=float, default=0.9)

    # scheduler
    scheduler_group = parser.add_argument_group(title='Scheduler options')
    scheduler_group.add_argument("--scheduler", type=str, default="cosine")
    scheduler_group.add_argument("--T_max", type=int, default=50)
    scheduler_group.add_argument("--warmup_epochs", type=int, default=2)
    scheduler_group.add_argument("--decay_epochs", type=int, default=2)
    scheduler_group.add_argument("--min_lr", type=float, default=1e-7)

    # training, valid, evaluation
    default_group.add_argument("--epochs", type=int, default=50)
    default_group.add_argument("--clip_grad", type=float, default=5.0)
    default_group.add_argument("--batch_size", type=int, default=2, help="batch size per gpu")
    default_group.add_argument("--num_workers", type=int, default=5)
    default_group.add_argument("--pin_memory", type=str2bool, default=True)
    default_group.add_argument("--non_blocking", type=str2bool, default=True)
    default_group.add_argument("--valid_freq", type=int, default=1)
    default_group.add_argument("--valid_index", type=str, default="f1")
    default_group.add_argument("--early_stop", type=str2bool, default=True)
    default_group.add_argument("--patience", type=int, default=10)

    # other
    default_group.add_argument("--train_print_freq", type=int, default=100)
    default_group.add_argument("--train_log_freq", type=int, default=20)
    default_group.add_argument("--valid_print_freq", type=int, default=20)
    default_group.add_argument("--save_base_model", type=str2bool, default=False)
    default_group.add_argument("--local_rank", type=int, default=0, help="node rank for distributed training")
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
    # if args.linear_probe:
    #     args.exp_name = 'linear_probe,' + args.exp_name
    args.exp_name = getTime() + '_' + args.exp_name + f',seed={args.seed}'
    args.output_dir = os.path.join(args.output_dir, args.experiment, args.input)
    if args.linear_probe:
        args.output_dir = os.path.join(args.output_dir, 'freeze_pretrained')
    if args.save_dir != "":
        args.output_dir = os.path.join(args.output_dir, args.save_dir)
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    ckpt_path = os.path.join(args.output_dir, 'checkpoints')
    event_path = os.path.join(args.output_dir, 'events')
    result_path = os.path.join(args.output_dir, 'results')
    args.ckpt_path = ckpt_path
    args.event_path = event_path
    args.result_path = result_path

    if args.T_max != args.epochs:
        print('Warning: The max step of cosine scheduler is not equal to the max epoches, which means there is '
              'multiple rounds of lr change.')

    if args.resume == '':
        load_dir_keys = [k for k in args.__dict__.keys() if k.endswith('_load_dir')]
        for load_dir_key in load_dir_keys:
            if getattr(args, load_dir_key) == '' and getattr(args, load_dir_key + '_ind') >= 0:
                setattr(args, load_dir_key, _checkpoints[getattr(args, load_dir_key + '_ind')])

    argsDict = args.__dict__
    if dist.get_rank() == 0:
        makedir(args.output_dir)
        makedir(args.ckpt_path)
        makedir(args.event_path)
        makedir(args.result_path)
        with open(os.path.join(args.output_dir, 'train_options.json'), 'w', encoding='utf-8') as f:
            json.dump(argsDict, f)
        show_options = '------------------ training options ------------------' + '\n'
        for eachArg, value in argsDict.items():
            show_options += eachArg + ' : ' + str(value) + '\n'
        show_options += '------------------- end -------------------'
        with open(os.path.join(args.output_dir, 'train_options.txt'), 'w', encoding='utf-8') as f:
            f.write(show_options)
        print(show_options)
        save_code('.', os.path.join(args.output_dir, 'code.zip'))


def main(args):
    # assert args.corr_backbone != 'none'
    ### env initialize ###
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    device = torch.device("cuda", args.local_rank)
    print(f"=====[init] local rank={args.local_rank}, rank={dist.get_rank()}, world size={dist.get_world_size()}, "
          f"device={device}=====")

    ### experiment initialize ###
    init_seeds(args.seed + dist.get_rank())
    initialize(args)


    ### build tokenizer ###
    if dist.get_rank() == 0:
        print(f"=> Build {args.tokenizer} tokenizer")
    tokenizer_path = tokenizer_dict[args.tokenizer]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    ### build model and optimizer ###
    if dist.get_rank() == 0:
        print(f"=> Build ParallelMed {args.model} Model")
    model = ParallelMedModel(args, stage="train")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    if args.save_base_model and dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'base_model.pth'))


    ### build dataset ###
    if dist.get_rank() == 0:
        print(f'=> Build training dataset')
    train_dataset = MedDataset(input=args.input, class_num=args.num_classes, image_size=args.image_size,
                               image_slice_num=args.slice_num, bio_num=args.bio_num,
                               bio_v_null=args.bio_v_null, bio_discrete=args.bio_discrete,
                               bio_normalize=args.bio_normalize, pct=args.train_pct, stage="train",
                               dataset_root=args.dataset_root, buffer_root=args.buffer_root)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)
    train_steps = len(train_loader)

    if dist.get_rank() == 0:
        print(f'=> Build valid dataset')
    valid_dataset = MedDataset(input=args.input, class_num=args.num_classes, image_size=args.image_size,
                               image_slice_num=args.slice_num, bio_num=args.bio_num,
                               bio_v_null=args.bio_v_null, bio_discrete=args.bio_discrete,
                               bio_normalize=args.bio_normalize, pct=args.val_pct, stage="valid",
                               dataset_root=args.dataset_root, buffer_root=args.buffer_root)
    valid_sampler = DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
                              num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)
    valid_steps = len(valid_loader)

    assert args.optimizer == 'adamw', f"Unsupported optimizer type: {args.optimizer}"
    assert args.scheduler == 'cosine', f"Unsupported scheduler type: {args.scheduler}"
    optim = torch.optim.AdamW
    sched = torch.optim.lr_scheduler.CosineAnnealingLR

    
    print(f'=> Use two-stage {args.optimizer} optimizer and {args.scheduler} scheduler')
    netf_optimizer = optim(model.module.get_two_stage_optim(args.netf_lr, args.netf_pretrained_lr,
                                                            stage='feature',
                                                            filter_bias_and_bn=args.filter_bias_and_bn),
                            eps=args.netf_eps, betas=(args.netf_beta0, args.netf_beta1),
                            weight_decay=args.netf_weight_decay)
    netc_optimizer = optim(model.module.get_two_stage_optim(args.netc_lr, args.netc_pretrained_lr,
                                                            stage='correlation',
                                                            filter_bias_and_bn=args.filter_bias_and_bn),
                            eps=args.netc_eps, betas=(args.netc_beta0, args.netc_beta1),
                            weight_decay=args.netc_weight_decay)
    netf_scheduler = sched(optimizer=netf_optimizer, T_max=args.T_max, eta_min=args.min_lr)
    netc_scheduler = sched(optimizer=netc_optimizer, T_max=args.T_max, eta_min=args.min_lr)
    optimizers, schedulers = [netf_optimizer, netc_optimizer], [netf_scheduler, netc_scheduler]

    n_iter = 0
    valid_n_iter = 0
    best_valid_metric_medication = None
    best_epoch_medication = 0

    if dist.get_rank() == 0:
        logger = SummaryWriter(log_dir=args.event_path)
        print('===============START TRAIN================')

    model.train()
    patience = 0
    for epoch in range(1, args.epochs + 1):
        buffer_train_loss = []
        buffer_train_scores, buffer_train_targets = [], []
        buffer_train_ids = []
        train_sampler.set_epoch(epoch)
        for step, data in enumerate(train_loader):
            for optimizer in optimizers:
                optimizer.zero_grad()
            n_iter += 1
            name_id = data["name"]
            buffer_train_ids.append(name_id)
            input = {}
            if 'ct' in args.input:
                input['image'] = data["image"].to(device, non_blocking=args.non_blocking)
            if 'biomarker' in args.input:
                input['biomarker'] = data["biomarker"].to(device, non_blocking=args.non_blocking)
                input['biomarker_missing_mask'] = data["biomarker_missing_mask"].to(device, non_blocking=args.non_blocking)
            if 'report' in args.input:
                enc_tokens = tokenizer(data["caption"], padding="max_length", max_length=args.text_max_length,
                                       truncation=True, return_tensors='pt')
                for k, v in enc_tokens.items():
                    enc_tokens[k] = v.to(device, non_blocking=args.non_blocking)
                input['text'] = enc_tokens

            label = data['one_hot_label'].to(device, non_blocking=args.non_blocking)

            output = model(input, label, stage='train')  # (bs, c)
            loss = output["loss"]
            score = output["medication_output"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            for optimizer in optimizers:
                optimizer.step()

            buffer_train_loss.append(loss.item())
            buffer_train_scores.append(torch.sigmoid(score.detach()))  # (bs, c)
            buffer_train_targets.append(label.detach())

            if dist.get_rank() == 0 and step % args.train_print_freq == 0:
                print(f'Epoch {epoch} ({step}/{train_steps}), '
                      f'Loss={loss.item():.4f}')

            if dist.get_rank() == 0 and n_iter % args.train_log_freq == 0:
                logger.add_scalars('medicine/loss_local',
                                   {'train': loss.item()}, n_iter)

        train_loss = link.AllReduce(np.mean(buffer_train_loss)).item()
        train_scores = torch.cat(buffer_train_scores, dim=0)  # (N, C)
        train_targets = torch.cat(buffer_train_targets, dim=0)  # (N, C)

        with torch.no_grad():
            gather_train_scores = link.AllGather.apply(train_scores)
            gather_train_scores = gather_train_scores.view(-1, *(train_scores.shape[1:]))
            gather_train_targets = link.AllGather.apply(train_targets)
            gather_train_targets = gather_train_targets.view(-1, *(train_targets.shape[1:]))

        train_metric_medication = compute_multi_label_metrics(gather_train_targets, gather_train_scores)

        if dist.get_rank() == 0:
            logger.add_scalars('medicine/loss', {'train': train_loss}, epoch)
            for k in train_metric_medication.keys():
                logger.add_scalar(f'medicine/train_metrics_{k}', np.mean(train_metric_medication[k]).item(), epoch)
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'latest_model.pth'))

            print_info = f'Train Epoch {epoch}/{args.epochs}, Loss = {train_loss:.4f}, '
            for k, v in train_metric_medication.items():
                print_info += f'{k.upper()}={100 * np.mean(v).item():.2f}, '
            print_info = print_info[:-2]
            print(print_info)

        ## valid
        if args.valid_freq > 0 and epoch % args.valid_freq == 0:
            valid_sampler.set_epoch(epoch)
            if dist.get_rank() == 0:
                print("================VALID=================")
            valid_n_iter += 1
            buffer_valid_loss = []
            buffer_valid_scores, buffer_valid_targets = [], []
            buffer_valid_ids = []
            model.eval()
            for step, data in enumerate(valid_loader):
                input = {}
                name_id = data["name"]
                buffer_valid_ids.append(name_id)
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
                    output = model(input, label, stage='valid')  # (bs, c)
                loss = output["loss"]
                score = output["medication_output"]

                buffer_valid_loss.append(loss.item())
                buffer_valid_scores.append(torch.sigmoid(score.detach()))  # (bs, c)
                buffer_valid_targets.append(label.detach())

                if dist.get_rank() == 0 and step % args.valid_print_freq == 0:
                    print(f'(Valid) Epoch {epoch} ({step}/{valid_steps}), '
                          f'Loss={loss.item():.4f}')

            valid_loss = link.AllReduce(np.mean(buffer_valid_loss)).item()
            valid_scores = torch.cat(buffer_valid_scores, dim=0)  # (N, C)
            valid_targets = torch.cat(buffer_valid_targets, dim=0)  # (N, C)

            with torch.no_grad():
                gather_valid_medication_scores = link.AllGather.apply(valid_scores)
                gather_valid_medication_scores = gather_valid_medication_scores.view(-1, *(
                valid_scores.shape[1:]))
                gather_valid_targets = link.AllGather.apply(valid_targets)
                gather_valid_targets = gather_valid_targets.view(-1, *(valid_targets.shape[1:]))

            valid_metric_medication = compute_multi_label_metrics(gather_valid_targets, gather_valid_medication_scores)

            if dist.get_rank() == 0:
                logger.add_scalars('medicine/loss', {'valid': valid_loss}, epoch)
                for k in valid_metric_medication.keys():
                    logger.add_scalar(f'medicine/valid_metrics_{k}',
                                      np.mean(valid_metric_medication[k]).item(), epoch)

                print_info = f'valid Epoch {epoch}/{args.epochs}, Loss={valid_loss:.4f}, '
                for k, v in valid_metric_medication.items():
                    print_info += f'{k.upper()}={100 * np.mean(v).item():.2f}, '
                print_info = print_info[:-2]
                print(print_info)
                print(f"====================End of Epoch {epoch}/{args.epochs}====================")

            if best_valid_metric_medication is None or \
                    np.mean(valid_metric_medication[args.valid_index]) >= \
                    np.mean(best_valid_metric_medication[args.valid_index]):
                best_valid_metric_medication = valid_metric_medication
                best_epoch_medication = epoch
                if args.corr_loss_medication != 0:
                    patience = 0

                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'optimal_model_medication.pth'))
                    with open(os.path.join(args.output_dir, 'optimal_valid_result_medication.txt'), 'w', encoding='utf-8') as f:
                        f.write(f'Medication Output\n'
                                f'Best epoch: {best_epoch_medication}/{args.epochs}')
                        for k, v in best_valid_metric_medication.items():
                            f.write(f'\n{k.upper()}: {100 * np.mean(v).item():.2f}')
            else:
                if args.corr_loss_medication != 0:
                    patience += 1

            if args.early_stop and patience >= args.patience:
                if dist.get_rank() == 0:
                    print("early stop")
                break

        model.train()

        for scheduler in schedulers:
            scheduler.step()

    if dist.get_rank() == 0:
        print('==============Complete train==============')
        print_info = f'save model {args.output_dir}\n'
        print_info += f'\nBest epoch:{best_epoch_medication}/{args.epochs}'
        for k, v in best_valid_metric_medication.items():
            print_info += f', {k.upper()}={100 * np.mean(v).item():.2f}'
        print(print_info)

        try:
            postfix = f',f1={100 * np.mean(best_valid_metric_medication["f1"]):.2f}'
            os.rename(args.output_dir, args.output_dir + postfix)
        except:
            pass


if __name__ == '__main__':
    args = get_parse()
    main(args)
