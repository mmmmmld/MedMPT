# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/30 17:03


import json
import random
import time
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
from datasets.pretrain_dataset import PretrainDataset
from models.pretrain_model import PretrainModel
from metrics.captions import compute_metrics as compute_caption_metrics
from optim.utils import add_lr_weight_decay
import optim.lr_sched as lr_sched
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_parse():
    parser = argparse.ArgumentParser()
    # default
    default_group = parser.add_argument_group(title='Default experiment options')
    default_group.add_argument("--root_dir", type=str, default="/root/maliangdi")
    default_group.add_argument("--experiment", type=str, default="pretrain")
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
    model_group.add_argument("--queue_size_slice", type=int, default=65536)
    model_group.add_argument("--queue_size_scan", type=int, default=2048)
    ## vision
    model_group.add_argument("--vision_load_dir", type=str, default="")
    model_group.add_argument("--vision_load_prefix", type=str, default="optimal")
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
    model_group.add_argument("--vision_mask_ratio_q", type=float, default=0.2)
    model_group.add_argument("--vision_mask_ratio_k", type=float, default=0)
    model_group.add_argument("--vision_decoder_embed_dim", type=int, default=512)
    model_group.add_argument("--vision_decoder_depth", type=int, default=4)
    model_group.add_argument("--vision_decoder_num_heads", type=int, default=8)
    model_group.add_argument("--vision_fusion", type=str, default='pool')
    ## text
    model_group.add_argument("--text_load_dir", type=str,default="")
    model_group.add_argument("--text_load_prefix", type=str, default="optimal")
    model_group.add_argument("--text_backbone", type=str, default="chinesebert")
    model_group.add_argument("--text_num_hidden_layers", type=int, default=6)
    model_group.add_argument("--text_output_hidden_states", type=str2bool, default=True)
    model_group.add_argument("--text_output_attentions", type=str2bool, default=True)
    model_group.add_argument("--text_freeze_pretrained", type=str2bool, default=False)
    model_group.add_argument("--text_freeze_pretrained_layers", type=str2list, default=[])
    model_group.add_argument("--text_pooling", type=str, default="mean")
    model_group.add_argument("--text_dropout", type=float, default=0.1)
    model_group.add_argument("--tie_text_encoder_decoder", type=str2bool, default=True)

    ## caption
    model_group.add_argument("--caption_load_dir", type=str, default="")
    model_group.add_argument("--caption_load_prefix", type=str, default="optimal")
    model_group.add_argument("--caption_backbone", type=str, default="chinesebert")
    model_group.add_argument("--caption_num_hidden_layers", type=int, default=6)
    model_group.add_argument("--caption_output_hidden_states", type=str2bool, default=True)
    model_group.add_argument("--caption_output_attentions", type=str2bool, default=True)
    model_group.add_argument("--caption_freeze_pretrained", type=str2bool, default=False)
    model_group.add_argument("--caption_freeze_pretrained_layers", type=str2list, default=[])
    model_group.add_argument("--caption_dropout", type=float, default=0.1)
    model_group.add_argument("--caption_beam_size", type=int, default=2)
    ## options
    model_group.add_argument("--global_feature_size", type=int, default=512)
    model_group.add_argument("--momentum", type=float, default=0.996)
    model_group.add_argument("--temperature_multimodal", type=float, default=0.07)
    model_group.add_argument("--temperature_image", type=float, default=0.07)
    model_group.add_argument("--temperature_text", type=float, default=0.07)

    # dataset
    dataset_group = parser.add_argument_group(title='Dataset options')
    dataset_group.add_argument("--dataset_root", type=str, default="/data/dir")
    dataset_group.add_argument("--buffer_root", type=str, default="/buffer/dir")
    dataset_group.add_argument("--train_pct", type=float, default=1.0, help="precentage of training dataset")
    dataset_group.add_argument("--val_pct", type=float, default=1.0, help="percentage of valid dataset")
    ## vision
    dataset_group.add_argument("--slice_num", type=int, default=32)
    dataset_group.add_argument("--image_size", type=int, default=224)
    ## text
    dataset_group.add_argument("--sentence_shuffle", type=str2bool, default=False)

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
    loss_group.add_argument("--weight_iic_slice", type=float, default=1)
    loss_group.add_argument("--weight_iic_scan", type=float, default=1)
    loss_group.add_argument("--weight_mim", type=float, default=1)
    loss_group.add_argument("--weight_ttc", type=float, default=0)
    loss_group.add_argument("--weight_itc", type=float, default=1)
    loss_group.add_argument("--weight_lm", type=float, default=1)

    # optimizer
    optimizer_group = parser.add_argument_group(title='Optimizer options')
    optimizer_group.add_argument("--optimizer", type=str, default='adamw')
    optimizer_group.add_argument("--lr", type=float, default=0.000001)
    optimizer_group.add_argument("--weight_decay", type=float, default=0.05)
    optimizer_group.add_argument("--eps", type=float, default=1e-8)
    optimizer_group.add_argument("--beta0", type=float, default=0.9)
    optimizer_group.add_argument("--beta1", type=float, default=0.999)
    optimizer_group.add_argument("--optimize_in_modality", type=str2bool, default=False)
    optimizer_group.add_argument("--vision_encoder_lr", type=float, default=0.000001)
    optimizer_group.add_argument("--vision_decoder_lr", type=float, default=0.00003)
    optimizer_group.add_argument("--vision_fusion_lr", type=float, default=0.00003)
    optimizer_group.add_argument("--text_encoder_lr", type=float, default=0.00001)
    optimizer_group.add_argument("--text_decoder_lr", type=float, default=0.00001)

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
    default_group.add_argument("--valid_index", type=str, default="auc")
    default_group.add_argument("--valid_mask_ratio", type=float, default=0.2)
    default_group.add_argument("--early_stop", type=str2bool, default=False)
    default_group.add_argument("--patience", type=int, default=10)

    # other
    default_group.add_argument("--train_print_freq", type=int, default=500)
    default_group.add_argument("--valid_print_freq", type=int, default=100)
    default_group.add_argument("--valid_save_image_freq", type=int, default=-1)
    default_group.add_argument("--save_base_model", type=str2bool, default=False)
    default_group.add_argument("--distributed", type=str2bool, default=True)
    default_group.add_argument("--local_rank", type=int, default=0, help="node rank for distributed training")
    default_group.add_argument("--dist_url", type=str, default="env://")
    default_group.add_argument("--fp16", type=str2bool, default=True)
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
    args.exp_name = getTime() + '_' + args.exp_name + f',seed={args.seed}'
    args.output_dir = os.path.join(args.output_dir, 'pretrain')
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
    if dist.get_rank() == 0:
        makedir(args.output_dir)
        makedir(args.ckpt_path)
        makedir(args.event_path)
        makedir(args.result_path)
        makedir(os.path.join(args.result_path, 'caption'))
        makedir(os.path.join(args.result_path, 'mim'))
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
    metric_logger.add_meter('loss_iic_slice', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_iic_scan', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mim', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))  # only in query image
    metric_logger.add_meter('loss_itc', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))  # only in query image
    metric_logger.add_meter('loss_ttc', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_lm', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]/[{}]'.format(epoch, args.epochs)
    data_loader.sampler.set_epoch(epoch)

    for i, data in enumerate(metric_logger.log_every(data_loader, args.train_print_freq, header)):
        if args.schedule_in_epoch:
            scheduler.adjust_learning_rate(optimizer, epoch, args.epochs, args)
        else:
            scheduler.adjust_learning_rate(optimizer, iter_already + i + 1, args.epochs * iter_per_epoch, args)
        optimizer.zero_grad()
        images = data['image']
        texts = data['caption']
        encode_tokens = []
        assert len(images) == 1 or len(images) == 2 or len(texts) == 1 or len(texts) == 2
        for img in images:
            img = img.to(device, non_blocking=True)

        for t in texts:
            temp = tokenizer(t, padding="max_length", max_length=args.encode_max_length, truncation=True,
                             return_tensors='pt')
            for k, v in temp.items():
                temp[k] = v.to(device)
            encode_tokens.append(temp)
        decode_tokens = tokenizer(texts[0], padding="max_length", max_length=args.decode_max_length, truncation=True,
                                  return_tensors='pt')
        for k, v in decode_tokens.items():
            decode_tokens[k] = v.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                # complete feature gather and loss computation in model
                output = model(im_q=images[0], im_k=images[-1], text_q=encode_tokens[0], text_k=encode_tokens[-1],
                               text_c=decode_tokens)

            loss = args.weight_iic_slice * output['loss_iic_slice'] + args.weight_iic_scan * output['loss_iic_scan'] \
                + args.weight_mim * output['loss_mim'] + args.weight_itc * output['loss_itc'] \
                + args.weight_lm * output['loss_lm'] + args.weight_ttc * output['loss_ttc']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(im_q=images[0], im_k=images[-1], text_q=encode_tokens[0], text_k=encode_tokens[-1],
                           text_c=decode_tokens)
            loss = args.weight_iic_slice * output['loss_iic_slice'] + args.weight_iic_scan * output['loss_iic_scan'] \
                + args.weight_mim * output['loss_mim'] + args.weight_itc * output['loss_itc'] \
                + args.weight_lm * output['loss_lm'] + args.weight_ttc * output['loss_ttc']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_iic_slice=output['loss_iic_slice'].item())
        metric_logger.update(loss_iic_scan=output['loss_iic_scan'].item())
        metric_logger.update(loss_mim=output['loss_mim'].item())
        metric_logger.update(loss_itc=output['loss_itc'].item())
        metric_logger.update(loss_ttc=output['loss_ttc'].item())
        metric_logger.update(loss_lm=output['loss_lm'].item())

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def valid_one_epoch(model, data_loader, tokenizer, epoch, device, args):
    # valid
    model.eval()

    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_mim', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bleu1', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bleu2', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bleu3', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bleu4', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('meteor', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('rouge_l', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Valid Epoch: [{}]/[{}]'.format(epoch, args.epochs)
    # data_loader.sampler.set_epoch(epoch)
    caption_records = ''

    for i, data in enumerate(metric_logger.log_every(data_loader, args.valid_print_freq, header)):
        images = data['image']  # len = 1
        texts = data['caption']  # len = 1
        encode_tokens = []
        images[0] = images[0].to(device, non_blocking=True)
        for t in texts:
            temp = tokenizer(t, padding='max_length', max_length=args.encode_max_length, truncation=True,
                             return_tensors='pt')
            for k, v in temp.items():
                temp[k] = v.to(device)
            encode_tokens.append(temp)
        decode_tokens = tokenizer(texts[0], padding='max_length', max_length=args.decode_max_length, truncation=True,
                                  return_tensors='pt')
        for k, v in decode_tokens.items():
            decode_tokens[k] = v.to(device)

        with torch.no_grad():
            output = model(im_q=images[0], text_q=encode_tokens[0], text_c=decode_tokens, mode='valid')

        metric_logger.update(loss_mim=output['loss_mim'].item())

        # record output result
        valid_caption_metrics, valid_caption_record = compute_caption_metrics(
            decode_tokens['input_ids'].detach(), output['pred_caption_ids'].detach(),
            tokenizer=tokenizer, img_path=data['image_path'])
        caption_records += valid_caption_record
        metric_logger.update(bleu1=valid_caption_metrics['BLEU_1'])
        metric_logger.update(bleu2=valid_caption_metrics['BLEU_2'])
        metric_logger.update(bleu3=valid_caption_metrics['BLEU_3'])
        metric_logger.update(bleu4=valid_caption_metrics['BLEU_4'])
        metric_logger.update(meteor=valid_caption_metrics['METEOR'])
        metric_logger.update(rouge_l=valid_caption_metrics['ROUGE_L'])

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    print("")

    with open(os.path.join(args.result_path, 'caption', f'valid_caption_epoch{epoch}_rank{link.get_rank()}.txt'),
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
    print(f"=> Build Pretrain Model")
    model = PretrainModel(args)

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        for k, v in checkpoint.items():
            checkpoint[k[len("module."):]] = v
            del checkpoint[k]
        model.load_state_dict(checkpoint)
        print("+ Resume checkpoint %s" % args.resume)

    ### build dataset ###
    print(f'=> Build Pretrain Dataset')
    train_dataset = PretrainDataset(args, stage="train")
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory)

    if args.validation:
        valid_dataset = PretrainDataset(args, stage="valid")
        valid_sampler = DistributedSampler(valid_dataset)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
                                  num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory)

    param_groups = add_lr_weight_decay(model, args)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=args.eps, betas=(args.beta0, args.beta1))
    for p_groups in optimizer.param_groups:
        p_groups["base_lr"] = p_groups["lr"]

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    if args.save_base_model and link.is_main_process() == 0:
        torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'checkpoint_base.pth'))

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if link.is_main_process():
        writer = SummaryWriter(log_dir=args.event_path)

    print('===============START TRAIN================')
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, tokenizer, optimizer, lr_sched, epoch, device, args, scaler=scaler)
        dist.barrier()
        if link.is_main_process():
            log_stats = {**{f'train_{k}': f'{v:.4f}' for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': args,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.ckpt_path, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            for k, v in train_stats.items():
                try:
                    writer.add_scalar(f'pretrain/{k}', v, epoch)
                except:
                    pass
        dist.barrier()

        if args.validation and epoch % args.valid_freq == 0:
            valid_stats = valid_one_epoch(model, valid_loader, tokenizer, epoch, device, args)
            dist.barrier()
            if link.is_main_process():
                log_stats = {**{f'valid_{k}': f'{v:.4f}' for k, v in valid_stats.items()},
                             'epoch': epoch,
                             }

                with open(os.path.join(args.output_dir, "log_val.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                for k, v in valid_stats.items():
                    try:
                        writer.add_scalar(f'pretrain/valid_{k}', v, epoch)
                    except:
                        pass

                caption_dir = os.path.join(args.result_path, 'caption')
                record_files = [p for p in os.listdir(caption_dir) if
                                p.startswith(f'valid_caption_epoch{epoch}_rank') and p.endswith('.txt')]
                p_rank = [int(os.path.splitext(p)[0].split('_rank')[-1]) for p in record_files]
                record_files, _ = [list(p) for p in zip(*sorted(zip(record_files, p_rank), key=lambda x: x[-1]))]
                record_files = [os.path.join(caption_dir, p) for p in record_files]
                assert len(record_files) == link.get_world_size()

                with open(os.path.join(caption_dir, f'valid_caption_epoch{epoch}.txt'), 'w', encoding='utf-8') as f:
                    for p in record_files:
                        for line in open(p, encoding='utf8'):
                            f.writelines(line)
                        f.write('\n')

                for p in record_files:
                    os.remove(p)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_parse()
    main(args)
