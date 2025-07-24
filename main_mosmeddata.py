# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/30 17:03
import json
import random
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold, StratifiedKFold

from utils import link
from utils import logger
from utils.basic import *
from datasets.mosmed_dataset import MosMedDataset, build_cv_dataset
from models.multiclass_model import MultiClassModel
from optim.utils import add_lr_weight_decay
from metrics.multiclass import compute_multi_class_metrics
import optim.lr_sched as lr_sched
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
Cross validation training
"""

_checkpoints = {
    0: "./ckpt/checkpoint_32.pth",
}

def get_parse():
    parser = argparse.ArgumentParser()
    # default
    default_group = parser.add_argument_group(title='Default experiment options')
    default_group.add_argument("--root_dir", type=str, default="/root/maliangdi")
    default_group.add_argument("--experiment", type=str, default="mosmeddata")
    default_group.add_argument("--task", type=str, default="diagnosis")
    default_group.add_argument("--linear_probe", action="store_true", default=False)
    default_group.add_argument("--cv_fold", type=int, default=3)
    default_group.add_argument("--target_fold", type=int, default=-1)
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
    model_group.add_argument("--vision_load_epoch", type=str, default=0)
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
    model_group.add_argument("--vision_dropout", type=float, default=0.1)

    # dataset
    dataset_group = parser.add_argument_group(title='Dataset options')
    dataset_group.add_argument("--class_num", type=int, default=2)
    ## vision
    dataset_group.add_argument("--slice_num", type=int, default=32)
    dataset_group.add_argument("--image_size", type=int, default=224)
    dataset_group.add_argument("--resample", type=str2bool, default=False)

    # loss
    loss_group = parser.add_argument_group(title='Loss options')
    loss_group.add_argument("--loss", type=str, default="ce", help='ce|bce')
    

    # optimizer
    optimizer_group = parser.add_argument_group(title='Optimizer options')
    optimizer_group.add_argument("--optimizer", type=str, default='adamw')
    optimizer_group.add_argument("--lr", type=float, default=0.000001)
    optimizer_group.add_argument("--weight_decay", type=float, default=0.05)
    optimizer_group.add_argument("--eps", type=float, default=1e-8)
    optimizer_group.add_argument("--beta0", type=float, default=0.9)
    optimizer_group.add_argument("--beta1", type=float, default=0.999)

    # scheduler
    scheduler_group = parser.add_argument_group(title='Scheduler options')
    scheduler_group.add_argument("--scheduler", type=str, default="cosine")
    scheduler_group.add_argument("--warmup_steps", type=int, default=2)
    scheduler_group.add_argument("--schedule_in_epoch", type=str2bool, default=True)
    scheduler_group.add_argument("--min_lr", type=float, default=1e-8)

    # training, valid, evaluation
    default_group.add_argument("--epochs", type=int, default=50)
    default_group.add_argument("--clip_grad", type=float, default=3.0)
    default_group.add_argument("--batch_size", type=int, default=8, help="batch size per gpu")
    default_group.add_argument("--num_workers", type=int, default=5)
    default_group.add_argument("--pin_memory", type=str2bool, default=True)
    default_group.add_argument("--non_blocking", type=str2bool, default=True)
    default_group.add_argument("--validation", type=str2bool, default=True)
    default_group.add_argument("--valid_freq", type=int, default=-1)
    default_group.add_argument("--valid_index", type=str, default="auc")
    default_group.add_argument("--early_stop", type=str2bool, default=True)
    default_group.add_argument("--patience", type=int, default=10)

    # other
    default_group.add_argument("--train_pct", type=float, default=1.0)
    default_group.add_argument("--val_pct", type=float, default=1.0)
    default_group.add_argument("--test_pct", type=float, default=1.0)
    default_group.add_argument("--train_print_freq", type=int, default=100)
    default_group.add_argument("--valid_print_freq", type=int, default=50)
    default_group.add_argument("--valid_save_image_freq", type=int, default=-1)
    default_group.add_argument("--save_base_model", type=str2bool, default=False)
    default_group.add_argument("--save_freq", type=int, default=10)
    default_group.add_argument("--distributed", type=str2bool, default=False)
    default_group.add_argument("--local_rank", type=int, default=0, help="node rank for distributed training")
    default_group.add_argument("--dist_url", type=str, default="env://")
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
    args.output_dir = os.path.join(args.output_dir, args.experiment, args.task)
    if args.linear_probe:
        args.output_dir = os.path.join(args.output_dir, 'linear_probe')
    if args.save_dir != "":
        args.output_dir = os.path.join(args.output_dir, args.save_dir)
    args.output_dir = os.path.join(args.output_dir, args.exp_name)

    if args.task == "diagnosis":
        args.class_num = 2
    elif args.task == "stage":
        args.class_num = 4
    else:
        raise ValueError

    if args.resume == '' and args.vision_load_dir == '' and args.vision_load_dir_ind >= 0:
        args.vision_load_dir = _checkpoints[args.vision_load_dir_ind]

    argsDict = args.__dict__

    makedir(args.output_dir)  
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


def train_one_epoch(model, data_loader, optimizer, scheduler, epoch, device, args, k_fold):
    # train
    iter_per_epoch = len(data_loader)
    iter_already = len(data_loader) * (epoch - 1)

    model.train()

    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', logger.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    pred_targets, pred_scores = [], []

    header = 'Fold: [{}]/[{}], Train Epoch: [{}]/[{}]'.format(k_fold, args.cv_fold, epoch, args.epochs)
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, data in enumerate(metric_logger.log_every(data_loader, args.train_print_freq, header)):
        if args.schedule_in_epoch:
            scheduler.adjust_learning_rate(optimizer, epoch, args.epochs, args)
        else:
            scheduler.adjust_learning_rate(optimizer, iter_already + i + 1, args.epochs * iter_per_epoch, args)

        images, labels = data['image'], data['label']
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        output = model(images, labels)
        loss = output['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=output['loss'].item())

        target = labels.detach()  # (bs,)
        prediction = output['prediction'].detach()  # (bs, c)
        pred_scores.append(prediction)
        pred_targets.append(target)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger.global_avg())
    pred_targets = torch.cat(pred_targets, dim=0)
    pred_scores = torch.cat(pred_scores)
    global_metric = compute_multi_class_metrics(pred_targets, pred_scores)
    return_metric = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_metric.update(global_metric)
    print("Averaged stats:", "\t".join([f"{k}: {v:.4f}" for k, v in return_metric.items()]))
    return return_metric


@torch.no_grad()
def evaluate(model, data_loader, device, args, mode, epoch, k_fold):
    # evaluate
    model.eval()

    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', logger.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    pred_targets, pred_scores = [], []
    data_ids = []

    header = 'Fold: [{}]/[{}], Valid Epoch: [{}]/[{}]'.format(k_fold, args.cv_fold, epoch, args.epochs)

    for i, data in enumerate(metric_logger.log_every(data_loader, args.valid_print_freq, header)):
        data_ids.extend(data["data_id"])
        images, labels = data['image'], data['label']
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            output = model(images, labels)

        metric_logger.update(loss=output['loss'].item())

        target = labels.detach()  # (bs,)
        prediction = output['prediction'].detach()  # (bs, c)
        pred_scores.append(prediction)
        pred_targets.append(target)

    pred_targets = torch.cat(pred_targets, dim=0)
    pred_scores = torch.cat(pred_scores)
    pred_results = dict([(did, {"target": pred_targets[idd].item(), "predict": pred_scores[idd].tolist()}) for idd, did in enumerate(data_ids)])
    global_metric = compute_multi_class_metrics(pred_targets, pred_scores)
    return_metric = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_metric.update(global_metric)
    print("Averaged stats:", "\t".join([f"{k}: {v:.4f}" for k, v in return_metric.items()]))
    return return_metric, pred_results


def run_one_fold(args, train_set, valid_set, device, k_fold, exp_start_time):
    fold_dir = os.path.join(args.output_dir, f"fold_{k_fold}")
    ckpt_path = os.path.join(fold_dir, 'checkpoints')
    event_path = os.path.join(fold_dir, 'events')
    makedir(fold_dir)
    makedir(ckpt_path)
    makedir(event_path)
    
    with open(os.path.join(fold_dir, "train_set.json"), "w", encoding="utf-8") as f:
        json.dump(train_set, f)
    with open(os.path.join(fold_dir, "valid_set.json"), "w", encoding="utf-8") as f:
        json.dump(valid_set, f)

    train_dataset = MosMedDataset(args, data_list=train_set, pct=args.train_pct, stage="train")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=None, shuffle=True, 
        num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory
        )
        
    valid_dataset = MosMedDataset(args, data_list=valid_set, pct=args.val_pct, stage="valid")
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, sampler=None, shuffle=True, 
        num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory
        )

    ### build model and optimizer ###
    print(f"=> Build Multi-Class Classification Model")
    model = MultiClassModel(args)

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        for k, v in checkpoint.items():
            if k.startswith("module."):
                checkpoint[k[len("module."):]] = v
                del checkpoint[k]
        model.load_state_dict(checkpoint)
        print("+ Resume checkpoint %s" % args.resume)
    
    param_groups = add_lr_weight_decay(model, args)
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=args.eps, betas=(args.beta0, args.beta1))
    for p_groups in optimizer.param_groups:
        p_groups["base_lr"] = p_groups["lr"]

    model = model.to(device)
    if args.save_base_model:
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'checkpoint_base.pth'))

    writer = SummaryWriter(log_dir=event_path)
    
    fold_start_time = time.time()
    best_metric_val = None
    best_result_val = None
    patient = 0

    for epoch in range(1, args.epochs + 1):  
        train_stats = train_one_epoch(model, train_loader, optimizer, lr_sched, epoch, device, args, k_fold)
        val_stats, val_results = evaluate(model, valid_loader, device, args, "valid", epoch, k_fold)
       
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': args,
            'epoch': epoch,
        }

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
        }

        torch.save(save_obj, os.path.join(ckpt_path, 'checkpoint_latest.pth'))
        if epoch % args.save_freq == 0 or epoch == args.epochs or epoch == 1:
            torch.save(save_obj, os.path.join(ckpt_path, 'checkpoint_%02d.pth' % epoch))

        with open(os.path.join(fold_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        for k, v in log_stats.items():
            if k == 'epoch':
                continue
            writer.add_scalar(f'mosmeddata/{k}', v, epoch)

        if best_metric_val is None or float(val_stats['auc']) >= best_metric_val['auc']:
            torch.save(save_obj, os.path.join(ckpt_path, 'checkpoint_best.pth'))
            best_metric_val = val_stats
            best_result_val = val_results
            best_epoch = epoch
            patient = 0
            with open(os.path.join(fold_dir, "best_result.txt"), "w") as f:
                f.write("best epoch: %d\n" % best_epoch)
                f.write("best valid metric:")
                for k, v in best_metric_val.items():
                    f.write(f"\n{k}: {v:.4f}")
            with open(os.path.join(fold_dir, "best_result.json"), "w", encoding="utf-8") as f:
                json.dump(best_result_val, f)
        else:
            patient += 1

        if args.early_stop and patient >= args.patience:
            break

    
    print(f'Fold {k_fold} Training time: {str(datetime.timedelta(seconds=int(time.time() - fold_start_time)))}, Total Time: {str(datetime.timedelta(seconds=int(time.time() - exp_start_time)))}')
        
    try:
        os.rename(fold_dir, fold_dir + f',acc={best_metric_val["acc"]:.4f},auc={best_metric_val["auc"]:.4f}')
    except:
        pass

    return best_result_val


def main(args):
    ### env initialize ###
    link.init_distributed_mode(args)
    device = torch.device("cuda", args.local_rank)

    ### experiment initialize ###
    init_seeds(args.seed + link.get_rank())
    initialize(args)

    ### build dataset ###
    print(f'=> Build Cross Validation MosMedDataset')
    floder = StratifiedKFold(n_splits=args.cv_fold, random_state=args.seed, shuffle=True)
    all_data, all_labels = build_cv_dataset(args.root_dir, args.task)
    all_results = {}
    for fold_index, (Trindex, Tsindex) in enumerate(floder.split(all_data, all_labels)):
        if args.target_fold > -1 and fold_index != args.target_fold:
            continue  # unless you want to run 1 fold only

        print(f"============== [{fold_index}]/[{args.cv_fold}] fold ==============")

        train_set = [all_data[i] for i in Trindex]
        val_set = [all_data[i] for i in Tsindex]

        fold_results = run_one_fold(args, train_set, val_set, device, fold_index, exp_start_time=time.time())
        all_results.update(fold_results)
    
    
    all_targets = np.array([val["target"] for val in all_results.values()])
    all_predicts = np.array([val["predict"] for val in all_results.values()])
    all_metric = compute_multi_class_metrics(all_targets, all_predicts)
    print("Optimal stats:", "\t".join([f"{k}: {v:.4f}" for k, v in all_metric.items()]))

    with open(os.path.join(args.output_dir, "best_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f)
    
    with open(os.path.join(args.output_dir, "best_result.txt"), "a") as f:
        for k, v in all_metric.items():
            f.write(f"\n{k}: {v:.4f}")
    
    try:
        os.rename(args.output_dir, args.output_dir + f',acc={all_metric["acc"]:.4f},auc={all_metric["auc"]:.4f}')
    except:
        pass


if __name__ == '__main__':
    args = get_parse()
    main(args)
