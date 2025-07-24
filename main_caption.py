# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/30 17:03


import json
import random
import time
import datetime
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
from utils import logger
from utils.basic import *
from tokenizer import tokenizer_dict
from datasets.caption_dataset import CaptionDataset
from models.caption_model import CaptionModel
from metrics.captions import compute_metrics as compute_caption_metrics
from optim.utils import add_lr_weight_decay
import optim.lr_sched as lr_sched
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_checkpoints = {
    "0": "./ckpt/checkpoint_32.pth"
}

def get_parse():
    parser = argparse.ArgumentParser()
    # default
    default_group = parser.add_argument_group(title='Default experiment options')
    default_group.add_argument("--root_dir", type=str, default="/root/maliangdi")
    default_group.add_argument("--experiment", type=str, default="caption")
    default_group.add_argument("--linear_probe", action="store_true", default=False)
    default_group.add_argument("--evaluate", action="store_true", default=False, help="only evaluate if true")
    default_group.add_argument("--exp_name", type=str, default="debug")
    default_group.add_argument("--output_dir", type=str, default="./output")
    default_group.add_argument("--save_dir", type=str, default="")
    default_group.add_argument("--ckpt_path", type=str, default="")
    default_group.add_argument("--event_path", type=str, default="")
    default_group.add_argument("--result_path", type=str, default="")

    # model
    model_group = parser.add_argument_group(title='Model options')
    model_group.add_argument("--model", type=str, default="v5")
    model_group.add_argument("--resume", type=str, default="")
    ## vision
    model_group.add_argument("--vision_load_dir", type=str, default="")
    model_group.add_argument("--vision_load_dir_ind", type=int, default=-1)
    model_group.add_argument("--vision_load_epoch", type=str, default="0")
    model_group.add_argument("--vision_backbone", type=str, default="vit-b/16")
    model_group.add_argument("--vision_pretrained", type=str2bool, default=False)
    model_group.add_argument("--vision_pretrained_weight", type=str, default="imagenet")
    model_group.add_argument("--vision_freeze_pretrained", type=str2bool, default=False)
    model_group.add_argument("--vision_freeze_pretrained_layers", type=str2list, default=[],
                             help="list of layers stated as: 1,2,3")
    model_group.add_argument("--vision_depth", type=int, default=12)
    model_group.add_argument("--vision_embed_dim", type=int, default=768)
    model_group.add_argument("--vision_patch_size", type=int, default=16)
    model_group.add_argument("--vision_num_heads", type=int, default=12)
    model_group.add_argument("--vision_mlp_ratio", type=float, default=4)
    ## caption
    model_group.add_argument("--caption_load_dir", type=str, default="")
    model_group.add_argument("--caption_load_dir_ind", type=int, default=-1)
    model_group.add_argument("--caption_load_epoch", type=str, default="0")
    model_group.add_argument("--caption_backbone", type=str, default="chinesebert")
    model_group.add_argument("--caption_pretrained", type=str2bool, default=False)
    model_group.add_argument("--caption_pretrained_weight", type=str, default="chinesebert")
    model_group.add_argument("--caption_freeze_pretrained", type=str2bool, default=False)
    model_group.add_argument("--caption_freeze_pretrained_layers", type=str2list, default=[],
                             help="list of layers stated as: 1,2,3")
    model_group.add_argument("--caption_num_hidden_layers", type=int, default=6)
    model_group.add_argument("--caption_output_hidden_states", type=str2bool, default=True)
    model_group.add_argument("--caption_output_attentions", type=str2bool, default=True)
    model_group.add_argument("--caption_dropout", type=float, default=0.1)
    model_group.add_argument("--caption_beam_size", type=int, default=2)

    # dataset
    dataset_group = parser.add_argument_group(title='Dataset options')
    dataset_group.add_argument("--dataset_root", type=str, default="/root/maliangdi/datasets/medicine")
    dataset_group.add_argument("--buffer_root", type=str, default="/buffer/maliangdi/medicine")
    dataset_group.add_argument("--train_pct", type=float, default=1.0, help="precentage of training dataset")
    dataset_group.add_argument("--val_pct", type=float, default=1.0, help="percentage of valid dataset")
    dataset_group.add_argument("--test_pct", type=float, default=1.0, help="percentage of test dataset")
    ## vision
    dataset_group.add_argument("--slice_num", type=int, default=32)
    dataset_group.add_argument("--image_size", type=int, default=224)
    ## text
    # tokenizer
    tokenizer_group = parser.add_argument_group(title='Tokenizer options')
    tokenizer_group.add_argument("--tokenizer", type=str, default='chinesebert')
    tokenizer_group.add_argument("--vocab_size", type=int, default=21128)
    tokenizer_group.add_argument("--encode_max_length", type=int, default=120)
    tokenizer_group.add_argument("--decode_max_length", type=int, default=120)
    tokenizer_group.add_argument("--text_pad_token_id", type=int, default=0)
    tokenizer_group.add_argument("--text_sos_token_id", type=int, default=101)
    tokenizer_group.add_argument("--text_eos_token_id", type=int, default=102)
    tokenizer_group.add_argument("--text_unk_token_id", type=int, default=100)

    # loss
    loss_group = parser.add_argument_group(title='Loss options')
    loss_group.add_argument("--loss", type=str, default="ce")

    # optimizer
    optimizer_group = parser.add_argument_group(title='Optimizer options')
    optimizer_group.add_argument("--optimizer", type=str, default='adamw')
    optimizer_group.add_argument("--lr", type=float, default=0.00001)
    optimizer_group.add_argument("--weight_decay", type=float, default=0.05)
    optimizer_group.add_argument("--eps", type=float, default=1e-8)
    optimizer_group.add_argument("--beta0", type=float, default=0.9)
    optimizer_group.add_argument("--beta1", type=float, default=0.999)

    # scheduler
    scheduler_group = parser.add_argument_group(title='Scheduler options')
    scheduler_group.add_argument("--scheduler", type=str, default="cosine")
    scheduler_group.add_argument("--warmup_steps", type=int, default=10000)
    scheduler_group.add_argument("--schedule_in_epoch", type=str2bool, default=False)
    scheduler_group.add_argument("--min_lr", type=float, default=1e-8)

    # training, valid, evaluation
    default_group.add_argument("--epochs", type=int, default=50)
    default_group.add_argument("--clip_grad", type=float, default=3.0)
    default_group.add_argument("--batch_size", type=int, default=2, help="batch size per gpu")
    default_group.add_argument("--num_workers", type=int, default=5)
    default_group.add_argument("--pin_memory", type=str2bool, default=True)
    default_group.add_argument("--non_blocking", type=str2bool, default=True)
    default_group.add_argument("--validation", type=str2bool, default=False)
    default_group.add_argument("--valid_freq", type=int, default=1)
    default_group.add_argument("--valid_index", type=str, default="bleu4")
    default_group.add_argument("--valid_mask_ratio", type=float, default=0.2)
    default_group.add_argument("--early_stop", type=str2bool, default=True)
    default_group.add_argument("--patience", type=int, default=5)

    # other
    default_group.add_argument("--train_print_freq", type=int, default=100)
    default_group.add_argument("--valid_print_freq", type=int, default=20)
    default_group.add_argument("--valid_save_image_freq", type=int, default=-1)
    default_group.add_argument("--save_base_model", type=str2bool, default=False)
    default_group.add_argument("--save_freq", default=10)
    default_group.add_argument("--distributed", type=str2bool, default=True)
    default_group.add_argument("--local_rank", type=int, default=0, help="node rank for distributed training")
    default_group.add_argument("--dist_url", type=str, default="env://")
    default_group.add_argument("--fp16", type=str2bool, default=False)
    default_group.add_argument("--seed", type=int, default=0, help="random seed")
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
    args.output_dir = os.path.join(args.output_dir, args.experiment)
    if args.linear_probe:
        args.output_dir = os.path.join(args.output_dir, 'zero_shot')
    if args.save_dir != "":
        args.output_dir = os.path.join(args.output_dir, args.save_dir)
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    ckpt_path = os.path.join(args.output_dir, 'checkpoints')
    event_path = os.path.join(args.output_dir, 'events')
    result_path = os.path.join(args.output_dir, 'results')
    args.ckpt_path = ckpt_path
    args.event_path = event_path
    args.result_path = result_path

    if args.resume == '':
        load_dir_keys = [k for k in args.__dict__.keys() if k.endswith('_load_dir')]
        for load_dir_key in load_dir_keys:
            if getattr(args, load_dir_key) == '' and getattr(args, load_dir_key + '_ind') >= 0:
                setattr(args, load_dir_key, _checkpoints[getattr(args, load_dir_key + '_ind')])

    argsDict = args.__dict__
    if link.is_main_process():
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


def train_one_epoch(model, data_loader, tokenizer, optimizer, scheduler, epoch, device, args, scaler=None):
    # train
    iter_per_epoch = len(data_loader)
    iter_already = len(data_loader) * (epoch - 1)
    model.train()
    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', logger.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]/[{}]'.format(epoch, args.epochs)
    buffer_input_ids, buffer_pred_ids, buffer_img_paths = [], [], []
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, data in enumerate(metric_logger.log_every(data_loader, args.train_print_freq, header)):
        if args.schedule_in_epoch:
            scheduler.adjust_learning_rate(optimizer, epoch, args.epochs, args)
        else:
            scheduler.adjust_learning_rate(optimizer, iter_already + i + 1, args.epochs * iter_per_epoch, args)
        optimizer.zero_grad()

        images = data['image']
        images = images.to(device, non_blocking=True)
        texts = data['caption']
        decode_tokens = tokenizer(texts, padding="max_length", max_length=args.decode_max_length, truncation=True,
                                  return_tensors='pt')
        for k, v in decode_tokens.items():
            decode_tokens[k] = v.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                # complete feature gather and loss computation in model
                output = model(im=images, text=decode_tokens, mode="train")
            loss = output['loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(im=images, text=decode_tokens, mode="train")
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=output['loss'].item())

        buffer_input_ids.append(decode_tokens['input_ids'].detach())
        buffer_pred_ids.append(output['pred_caption_ids'].detach())
        buffer_img_paths.extend(data['image_path'])

    _, train_caption_record = compute_caption_metrics(
        torch.cat(buffer_input_ids, dim=0), torch.cat(buffer_pred_ids, dim=0),
        tokenizer=tokenizer, img_path=buffer_img_paths)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    with open(os.path.join(args.result_path, f'train_caption_epoch{epoch}_rank{link.get_rank()}.txt'),
              'w', encoding='utf-8') as f:
        f.write(train_caption_record)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, epoch, device, args, mode):
    # valid
    model.eval()

    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('bleu1', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bleu2', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bleu3', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bleu4', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('meteor', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('cider', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('rouge_l', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = '{} Epoch: [{}]/[{}]'.format(mode.capitalize(), epoch, args.epochs)

    caption_records = ""

    for i, data in enumerate(metric_logger.log_every(data_loader, args.valid_print_freq, header)):
        images = data['image']  # len = 1
        texts = data['caption']  # len = 1
        images = images.to(device, non_blocking=True)
        decode_tokens = tokenizer(texts, padding='max_length', max_length=args.decode_max_length, truncation=True,
                                  return_tensors='pt')
        for k, v in decode_tokens.items():
            decode_tokens[k] = v.to(device)

        with torch.no_grad():
            output = model(im=images, text=decode_tokens, mode='valid')

        valid_caption_metrics, valid_caption_record = compute_caption_metrics(
            decode_tokens['input_ids'].detach(), output['pred_caption_ids'].detach(),
            tokenizer=tokenizer, img_path=data['image_path'])
        caption_records += valid_caption_record

        metric_logger.update(bleu1=valid_caption_metrics['BLEU_1'])
        metric_logger.update(bleu2=valid_caption_metrics['BLEU_2'])
        metric_logger.update(bleu3=valid_caption_metrics['BLEU_3'])
        metric_logger.update(bleu4=valid_caption_metrics['BLEU_4'])
        metric_logger.update(meteor=valid_caption_metrics['METEOR'])
        metric_logger.update(cider=valid_caption_metrics['CIDER'])
        metric_logger.update(rouge_l=valid_caption_metrics['ROUGE_L'])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    with open(os.path.join(args.result_path, f'{mode}_caption_epoch{epoch}_rank{link.get_rank()}.txt'),
              'w', encoding='utf-8') as f:
        f.write(caption_records)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    ### env initialize ###
    link.init_distributed_mode(args)
    device = torch.device("cuda", args.local_rank)

    ### experiment initialize ###
    init_seeds(args.seed + link.get_rank())
    initialize(args)

    ### build tokenizer ###
    print(f"=> Build {args.tokenizer} tokenizer")
    tokenizer_path = tokenizer_dict[args.tokenizer]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    assert tokenizer.pad_token_id == args.text_pad_token_id, "pad token id not match."
    assert tokenizer.cls_token_id == args.text_sos_token_id, "sos(cls) token id not match."
    assert tokenizer.sep_token_id == args.text_eos_token_id, "eos(sep) token id not match."
    assert tokenizer.unk_token_id == args.text_unk_token_id, "unk token id not match."

    ### build model and optimizer ###
    print(f"=> Build Caption Model")
    model = CaptionModel(args)

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        for k, v in checkpoint.items():
            if k.startswith("module."):
                checkpoint[k[len("module."):]] = v
                del checkpoint[k]
        model.load_state_dict(checkpoint)
        print("+ Resume checkpoint %s" % args.resume)

    ### build dataset ###
    print(f'=> Build Caption Dataset')
    if not args.evaluate:
        train_dataset = CaptionDataset(args, pct=args.train_pct, stage="train")
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  shuffle=train_sampler is None, num_workers=args.num_workers,
                                  drop_last=True, pin_memory=args.pin_memory)

    valid_dataset = CaptionDataset(args, pct=args.val_pct, stage="valid")
    if args.distributed:
        valid_sampler = DistributedSampler(valid_dataset)
    else:
        valid_sampler = None
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
                              num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)

    test_dataset = CaptionDataset(args, pct=args.test_pct, stage="test")
    if args.distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)

    param_groups = add_lr_weight_decay(model, args)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=args.eps, betas=(args.beta0, args.beta1))
    for p_groups in optimizer.param_groups:
        p_groups["base_lr"] = p_groups["lr"]

    model = model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    if args.save_base_model and link.is_main_process() == 0:
        torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'checkpoint_base.pth'))

    if link.is_main_process():
        writer = SummaryWriter(log_dir=args.event_path)

    print("Start training")
    start_time = time.time()
    best = 0
    best_epoch = 0
    best_test = 0
    patient = 0

    for epoch in range(1, args.epochs + 1):
        if not args.evaluate:
            train_stats = train_one_epoch(model, train_loader, tokenizer, optimizer, lr_sched, epoch, device, args)

        val_stats = evaluate(model, valid_loader, tokenizer, epoch, device, args, "valid")
        test_stats = evaluate(model, test_loader, tokenizer, epoch, device, args, "test")

        if args.distributed:
            dist.barrier()

        if link.is_main_process():
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': args,
                'epoch': epoch,
            }

            if args.evaluate:  # no train, only evaluate
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                torch.save(save_obj, os.path.join(args.ckpt_path, 'checkpoint_test.pth'))

            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             }

                torch.save(save_obj, os.path.join(args.ckpt_path, 'checkpoint_latest.pth'))
                if epoch % args.save_freq == 0 or epoch == args.epochs or epoch == 1:
                    torch.save(save_obj, os.path.join(args.ckpt_path, 'checkpoint_%02d.pth' % epoch))

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                for k, v in log_stats.items():
                    if k == 'epoch':
                        continue
                    writer.add_scalar(f'caption/{k}', v, epoch)

                if float(val_stats[args.valid_index]) > best:
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

            caption_dir = args.result_path
            for prefix in ["train", "valid", "test"]:
                record_files = [p for p in os.listdir(caption_dir) if
                                p.startswith(f'{prefix}_caption_epoch{epoch}_rank') and p.endswith('.txt')]
                p_rank = [int(os.path.splitext(p)[0].split('_rank')[-1]) for p in record_files]
                record_files, _ = [list(p) for p in zip(*sorted(zip(record_files, p_rank), key=lambda x: x[-1]))]
                record_files = [os.path.join(caption_dir, p) for p in record_files]
                assert len(record_files) == link.get_world_size()

                with open(os.path.join(caption_dir, f'{prefix}_caption_epoch{epoch}.txt'), 'w', encoding='utf-8') as f:
                    for p in record_files:
                        for line in open(p, encoding='utf8'):
                            f.writelines(line)
                        f.write('\n')

                for p in record_files:
                    os.remove(p)

        if float(val_stats[args.valid_index]) > best:
            best = float(val_stats[args.valid_index])
            best_epoch = epoch
            best_test = float(test_stats[args.valid_index])
            patient = 0
        else:
            patient += 1

        if args.distributed:
            dist.barrier()

        if args.evaluate or (args.early_stop and patient >= args.patience):
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if link.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d\n" % best_epoch)
            f.write(f"best {args.valid_index}: val:{best:.4f}/test:{best_test:.4f}")
        try:
            os.rename(args.output_dir, args.output_dir + f',{args.valid_index}={best_test:.4f}')
        except:
            pass


if __name__ == '__main__':
    args = get_parse()
    main(args)
