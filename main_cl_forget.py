import argparse
import datetime
import json
import random
import time
from pathlib import Path
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.incremental import generate_cls_order
from engine import evaluate, train_one_epoch, train_one_epoch_incremental, train_one_epoch_forget_remain, train_one_epoch_forget_cls
from models import build_model
import wandb
import os
import loralib as lora
from collections import OrderedDict
import util.cal_norm as cal_norm
import torch.distributed as dist
from baselines.random_drop import random_drop_weights

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector',
                                     add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names',
                        default=["backbone.0"],
                        type=str,
                        nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    # parser.add_argument('--lr_beta', default=0.01, type=float)
    parser.add_argument('--lr_linear_proj_names',
                        default=['reference_points', 'sampling_offsets'],
                        type=str,
                        nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--lr_drop_balanced', default=10, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm',
                        default=0.1,
                        type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine',
                        default=False,
                        action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument(
        '--frozen_weights',
        type=str,
        default=None,
        help=
        "Path to the pretrained model. If set, only the mask head will be trained"
    )

    # * Backbone
    parser.add_argument('--backbone',
                        default='resnet50',
                        type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument(
        '--dilation',
        action='store_true',
        help=
        "If true, we replace stride with dilation in the last convolutional block (DC5)"
    )
    parser.add_argument(
        '--position_embedding',
        default='sine',
        type=str,
        choices=('sine', 'learned'),
        help="Type of positional embedding to use on top of the image features"
    )
    parser.add_argument('--position_embedding_scale',
                        default=2 * np.pi,
                        type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels',
                        default=4,
                        type=int,
                        help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers',
                        default=6,
                        type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers',
                        default=6,
                        type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument(
        '--dim_feedforward',
        default=1024,
        type=int,
        help=
        "Intermediate size of the feedforward layers in the transformer blocks"
    )
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout',
                        default=0.1,
                        type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument(
        '--nheads',
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries',
                        default=300,
                        type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks',
                        action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument(
        '--no_aux_loss',
        dest='aux_loss',
        action='store_false',
        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class',
                        default=2,
                        type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox',
                        default=5,
                        type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou',
                        default=2,
                        type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # forget loss coefficients
    parser.add_argument('--forget_cls_loss_coef', default=-2, type=float)
    parser.add_argument('--forget_bbox_loss_coef', default=5, type=float)
    parser.add_argument('--forget_giou_loss_coef', default=2, type=float)

    # ref coefficients are only for this CL-DETR method
    parser.add_argument('--ref_cls_loss_coef', default=2, type=float)
    parser.add_argument('--ref_bbox_loss_coef', default=5, type=float)
    parser.add_argument('--ref_giou_loss_coef', default=2, type=float)
    parser.add_argument('--ref_loss_overall_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode',
                        default=False,
                        action='store_true',
                        help='whether to cache images on memory')

    # incremental parameters
    parser.add_argument('--num_of_phases', default=2, type=int)
    parser.add_argument('--cls_per_phase', default=40, type=int)
    parser.add_argument('--data_setting',
                        default='tfh',
                        choices=['tfs', 'tfh'])
    parser.add_argument('--num_of_first_cls', default=40, type=int)

    #parser.add_argument('--cls_per_phase', default=3, type=int)
    #parser.add_argument('--data_setting', default='tfs', choices=['tfs', 'tfh'])

    parser.add_argument('--seed_cls', default=123, type=int)
    parser.add_argument('--seed_data', default=123, type=int)
    parser.add_argument('--method',
                        default='icarl',
                        choices=['baseline', 'icarl'])
    parser.add_argument('--mem_rate',
                        default=0.1,
                        type=float,
                        help='memory rate of the rehearsal set')

    parser.add_argument('--debug_mode', default=False, action='store_true')
    parser.add_argument('--balanced_ft', default=False, action='store_true')
    parser.add_argument('--eval', action='store_true')

    # rehearsal rate and forget rate of the corresponding dataset
    parser.add_argument('--rehearsal_rate',
                        default=0.1,
                        type=float,
                        help='rehearsal rate of the rehearsal set')
    parser.add_argument('--forget_rate',
                        default=0.1,
                        type=float,
                        help='memory rate of the forget set')
    parser.add_argument('--finetune_rehearsal_epoch',
                        default=50,
                        type=int,
                        help='rehearsal buffer epoch, rehearsal training as a baseline method which only use remain dataset to fineturn the model')
    parser.add_argument('--finetune_forget_epoch',
                        default=10,
                        type=int,
                        help='use forget dataset to fineturn the model with gradient ascent')
    parser.add_argument('--lr_drop_rehearsal', default=40, type=int)
    # LoRA rank
    parser.add_argument('--lora_rank',
                        default=0,
                        type=int,
                        help='rank of the LoRA model')
    parser.add_argument('--lora_reg_rank',
                        default=0,
                        type=int,
                        help='rank of the LoRA regression head')
    parser.add_argument('--lora_cls_rank',
                        default=0,
                        type=int,
                        help='rank of the LoRA classification head')   
    # LoRA position
    parser.add_argument('--lora_pos',
                        default=['None'],
                        nargs='+',
                        choices=['encoder', 'decoder', 'proj_layer', 'None'],
                        help='position of the LoRA model')
    # LoRA checkpoint
    parser.add_argument('--resume_lora',
                        default=None,
                        type=str,
                        help='checkpoint of the LoRA model')

    # baseline rehearsal training directly, not using LoRA, directly train DETR from scratch  using rehearsal set
    parser.add_argument('--rehearsal_training',
                        default=False,
                        action='store_true',
                        help='whether to use rehearsal training directly')
    parser.add_argument('--finetune_gradient_ascent',
                        default=False,
                        action='store_true',
                        help='whether to use gradient ascent to finetune the model using forget set')
    parser.add_argument('--random_drop',
                        default=False,
                        action='store_true',
                        help='whether to use random drop to the model as a baseline')
    parser.add_argument('--random_drop_rate',
                        default=0.0,
                        type=float,
                        help='random drop rate of pretrained model')
    ## one stage training
    parser.add_argument('--one_stage',
                        default=True,
                        action='store_false',
                        help='whether to use one stage training')
    parser.add_argument('--beta', default=0.04, type=float, help='beta for balancing the remain and forget loss')
    parser.add_argument('--alpha', default=0.04, type=float, help='alpha for group sparse loss')
    # wandb
    parser.add_argument(
        '--wandb_offline',
        default=False,
        action='store_true',
    )
    # specify the encoder layer to use LoRA
    parser.add_argument('--lora_encoder_layers', nargs='*', default=[], type=int)
    # specify the decoder layer to use LoRA
    parser.add_argument('--lora_decoder_layers', nargs='*', default=[], type=int)
    # manual order for classes
    parser.add_argument('--manual_order', default=False, action='store_true')
    
    # continual learning args
    parser.add_argument('--num_tasks', type=int)
    parser.add_argument('--this_task_start_cls', default=0, type=int)
    parser.add_argument('--this_task_end_cls', default=0, type=int)
    parser.add_argument('--cl_beta_list', nargs='*', default=[], type=float)
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    if args.rehearsal_training:
        args.lora_pos = ['None']
        args.lora_rank = 0

    print(args)

    if args.rank == 0:
        wandb.login(key='808d6ef02f3a9c448c5641c132830eb0c3c83c2a')
        wandb.init(project="forget learning", dir=args.output_dir, \
                    group="cl-forget", mode="offline" if args.wandb_offline else "online")
        wandb.config.update(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if 'None' not in args.lora_pos:
        print("Training", args.lora_pos)
        lora.mark_only_lora_as_trainable(model)  # only train LoRA
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)
    print('ratio of params:', n_parameters / 39847265)
    if args.rank == 0:
        wandb.log({
            'number of params': n_parameters,
            'ratio of params': n_parameters / 39847265
        })

    cls_order = generate_cls_order(seed=args.seed_cls, manual=args.manual_order)

    if args.data_setting == 'tfs':
        total_phase_num = args.num_of_phases
    elif args.data_setting == 'tfh':
        total_phase_num = args.num_of_phases
    else:
        raise ValueError('Please set the correct data setting.')

    origin_first_task_cls = args.num_of_first_cls
    for task_i in range(args.num_tasks): # start from 0
        # modify num_of_first_cls according to task id
        print('\n')
        print('*****************task_i:', task_i,'***********************')
        print('\n')
        args.num_of_first_cls = origin_first_task_cls - task_i * args.cls_per_phase
    
        dataset_remain = build_dataset(image_set='train', args=args, cls_order=cls_order, \
            phase_idx=0, incremental=True, incremental_val=False, val_each_phase=False, is_rehearsal=True) # is_rehearal will use args.rehearsal_rate
        args.this_task_start_cls = origin_first_task_cls - task_i * args.cls_per_phase
        args.this_task_end_cls = args.this_task_start_cls + args.cls_per_phase
        dataset_forget = build_dataset(image_set='train', args=args, cls_order=cls_order, \
                phase_idx=1, incremental=True, incremental_val=False, val_each_phase=False)
        dataset_val_remain = build_dataset(image_set='val', args=args, cls_order=cls_order, \
                    phase_idx=0, incremental=True, incremental_val=True, val_each_phase=True)
        dataset_val_forget = build_dataset(image_set='val', args=args, cls_order=cls_order, \
                    phase_idx=1, incremental=True, incremental_val=True, val_each_phase=True)    
        if task_i > 0:
            # val already forget dataset in the previous tasks: 60-10-10(last 10 is forget in the previous tasks)
            args.this_task_start_cls = origin_first_task_cls - (task_i - 1) * args.cls_per_phase 
            dataset_val_old = build_dataset(image_set='val', args=args, cls_order=cls_order, \
                        phase_idx=1, incremental=True, incremental_val=True, val_each_phase=False, is_rehearsal=True)
        
        if args.distributed:
            if args.cache_mode:
                sampler_forget = samplers.NodeDistributedSampler(dataset_forget)
                sampler_remain = samplers.NodeDistributedSampler(
                    dataset_remain)  # shuffle=True by default
                sampler_val_forget = samplers.NodeDistributedSampler(
                    dataset_val_forget, shuffle=False)
                sampler_val_remain = samplers.NodeDistributedSampler(
                    dataset_val_remain, shuffle=False)
                if task_i > 0:
                    sampler_val_old = samplers.NodeDistributedSampler(
                        dataset_val_old, shuffle=False)
            else:
                sampler_forget = samplers.DistributedSampler(dataset_forget)
                sampler_remain = samplers.DistributedSampler(dataset_remain)
                sampler_val_forget = samplers.DistributedSampler(
                    dataset_val_forget, shuffle=False)
                sampler_val_remain = samplers.DistributedSampler(
                    dataset_val_remain, shuffle=False)
                if task_i > 0:
                    sampler_val_old = samplers.DistributedSampler(
                        dataset_val_old, shuffle=False)
        else:
            sampler_forget = torch.utils.data.RandomSampler(dataset_forget)
            sampler_remain = torch.utils.data.RandomSampler(dataset_remain)
            sampler_val_forget = torch.utils.data.SequentialSampler(
                dataset_val_forget)
            sampler_val_remain = torch.utils.data.SequentialSampler(
                dataset_val_remain)
            if task_i > 0:
                sampler_val_old = torch.utils.data.SequentialSampler(
                    dataset_val_old)

        batch_sampler_forget = torch.utils.data.BatchSampler(sampler_forget,
                                                            args.batch_size,
                                                            drop_last=True)
        batch_sampler_remain = torch.utils.data.BatchSampler(sampler_remain,
                                                            args.batch_size,
                                                            drop_last=True)

        data_loader_forget = DataLoader(dataset_forget,
                                        batch_sampler=batch_sampler_forget,
                                        collate_fn=utils.collate_fn,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
        data_loader_remain = DataLoader(dataset_remain,
                                        batch_sampler=batch_sampler_remain,
                                        collate_fn=utils.collate_fn,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
        data_loader_val_forget = DataLoader(dataset_val_forget, batch_size=args.batch_size, \
                                            sampler=sampler_val_forget, drop_last=False,\
                                            collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
        data_loader_val_remain = DataLoader(dataset_val_remain, batch_size=args.batch_size, \
                                            sampler=sampler_val_remain, drop_last=False,\
                                            collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
        if task_i > 0:
            data_loader_val_old = DataLoader(dataset_val_old, batch_size=args.batch_size, \
                                            sampler=sampler_val_old, drop_last=False,\
                                            collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        for n, p in model_without_ddp.named_parameters():
            print(n)

        param_dicts = [{
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if p.requires_grad
            ],
            "lr":
            args.lr,
        }]

        if args.rehearsal_training or args.finetune_gradient_ascent:
            param_dicts = [{
                "params": [
                    p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names)
                    and not match_name_keywords(n, args.lr_linear_proj_names)
                    and p.requires_grad
                ],
                "lr":
                args.lr,
            }, {
                "params": [
                    p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_backbone_names)
                    and p.requires_grad
                ],
                "lr":
                args.lr_backbone,
            }, {
                "params": [
                    p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_linear_proj_names)
                    and p.requires_grad
                ],
                "lr":
                args.lr * args.lr_linear_proj_mult,
            }]

        print('setting the optimizer...')
        if args.sgd:
            optimizer_forget = torch.optim.SGD(param_dicts,
                                            lr=args.lr,
                                            momentum=0.9,
                                            weight_decay=args.weight_decay)
            optimizer_remain = torch.optim.SGD(param_dicts,
                                            lr=args.lr,
                                            momentum=0.9,
                                            weight_decay=args.weight_decay)
        else:
            optimizer_forget = torch.optim.AdamW(param_dicts,
                                                lr=args.lr,
                                                weight_decay=args.weight_decay)
            optimizer_remain = torch.optim.AdamW(param_dicts,
                                                lr=args.lr,
                                                weight_decay=args.weight_decay)
        lr_scheduler_forget = torch.optim.lr_scheduler.StepLR(
            optimizer_forget, args.lr_drop, gamma=args.lr_gamma)
        lr_scheduler_remain = torch.optim.lr_scheduler.StepLR(
            optimizer_remain, args.lr_drop_rehearsal, gamma=args.lr_gamma)
        
        if task_i == 0:
            print('pytorch model distributed...')
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu])
                model_without_ddp = model.module

        base_ds_forget = get_coco_api_from_dataset(dataset_val_forget)
        base_ds_remain = get_coco_api_from_dataset(dataset_val_remain)
        if task_i > 0:
            base_ds_old = get_coco_api_from_dataset(dataset_val_old)

        if args.frozen_weights is not None:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
            model_without_ddp.detr.load_state_dict(checkpoint['model'])

        output_dir = Path(args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # 如果args.resume为True，且rehearsal_training为False，则加载预训练模型
        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                                map_location='cpu',
                                                                check_hash=True)
            else:
                # load DETR weights
                if task_i == 0:
                    checkpoint = torch.load(args.resume, map_location='cpu')
                    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
                        checkpoint['model'], strict=False)
                    unexpected_keys = [
                        k for k in unexpected_keys
                        if not (k.endswith('total_params') or k.endswith('total_ops'))
                    ]

                    if len(missing_keys) > 0:
                        print('Missing Keys: {}'.format(missing_keys))
                    if len(unexpected_keys) > 0:
                        print('Unexpected Keys: {}'.format(unexpected_keys))

                    # load lora weights
                    if args.resume_lora is not None:
                        lora_checkpoint = torch.load(args.resume_lora,
                                                    map_location='cpu')
                        model_without_ddp.load_state_dict(lora_checkpoint['lora'],
                                                    strict=False)
                else: # task_i > 0
                    if args.one_stage:    
                        last_checkpoint_path = os.path.join(args.output_dir , ('checkpoint_task_' + str(task_i-1)+'.pth'))
                        checkpoint = torch.load(last_checkpoint_path, map_location='cpu')
                        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
                            checkpoint['model'], strict=False)
                        unexpected_keys = [
                            k for k in unexpected_keys
                            if not (k.endswith('total_params') or k.endswith('total_ops'))
                        ]

                        if len(missing_keys) > 0:
                            print('Missing Keys: {}'.format(missing_keys))
                        if len(unexpected_keys) > 0:
                            print('Unexpected Keys: {}'.format(unexpected_keys))
                        
                        # reintialize lora weights
                        utils.reinitialize_lora_parameters(model_without_ddp)
                        # import pdb; pdb.set_trace()
                    else:
                        pass
            # check the resumed model
            if not args.eval:
                print("Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_forget,
                    base_ds_forget, device, args.output_dir)
                utils.log_wandb(args=args,array=forget_maps,task_i='resumed-eval',name='forget', epoch=0)
                print("Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_remain,
                    base_ds_remain, device, args.output_dir)
                utils.log_wandb(args=args,array=remain_maps,task_i='resumed-eval', name='remain',epoch=0)
                if task_i > 0:
                    print("Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model, criterion, postprocessors, data_loader_val_old,
                        base_ds_old, device, args.output_dir)
                    utils.log_wandb(args=args,array=old_maps,name='old',task_i='resumed-eval', epoch=0)
        if args.eval:
            print("Testing for forget classes")
            test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                model, criterion, postprocessors, data_loader_val_forget,
                base_ds_forget, device, args.output_dir)
            utils.log_wandb(args=args,array=forget_maps,name='forget', task_i='eval', epoch=0)

            print("Testing for remain classes")
            test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                model, criterion, postprocessors, data_loader_val_remain,
                base_ds_remain, device, args.output_dir)
            utils.log_wandb(args=args,array=remain_maps,name='remain',task_i='eval', epoch=0)
            return
        
        if args.one_stage:
            cl_beta = args.cl_beta_list[task_i]
            print("start one stage training")
            start_time = time.time()
            epoch = 0
            for epoch in range(args.epochs):
                if args.distributed:
                    sampler_forget.set_epoch(epoch)

                train_stats = train_one_epoch_forget_cls(model,
                                                        criterion,
                                                        data_loader_forget=data_loader_forget,
                                                        data_loader_remain=data_loader_remain,
                                                        optimizer=optimizer_forget, device=device,
                                                        epoch=epoch, max_norm=args.clip_max_norm,
                                                        beta=cl_beta, alpha=args.alpha)
                lr_scheduler_forget.step()

                print("One stage training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_forget,
                    base_ds_forget, device, args.output_dir)
                utils.log_wandb(args=args,array=forget_maps,name='forget',task_i=task_i, epoch=epoch+1)
                print("One stage training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_remain,
                    base_ds_remain, device, args.output_dir)
                utils.log_wandb(args=args,array=remain_maps,name='remain',task_i=task_i, epoch=epoch+1)

                if task_i > 0:
                    print("One stage training. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model, criterion, postprocessors, data_loader_val_old,
                        base_ds_old, device, args.output_dir)
                    utils.log_wandb(args=args,array=old_maps,name='old',task_i=task_i, epoch=epoch+1)
                    
                if args.output_dir:
                    checkpoint_paths = [output_dir / ('checkpoint_task_'+ str(task_i) + '.pth')]
                    checkpoint_paths_lora = [output_dir / ('checkpoint_lora_task_' + str(task_i) + '.pth')]

                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master(
                            {
                                'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer_forget.state_dict(),
                                'lr_scheduler': lr_scheduler_forget.state_dict(),
                                'epoch': epoch,
                                'args': args
                            }, checkpoint_path)

                    for checkpoint_path_lora in checkpoint_paths_lora:
                        utils.save_on_master(
                            {'lora': OrderedDict(lora.lora_state_dict(model_without_ddp))},
                            checkpoint_path_lora)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            if dist.get_rank() == 0:
                wandb.log({'time': total_time_str})
            # get the norm of the lora model for visualization
            norm_list = cal_norm.get_norm_of_lora(model_without_ddp)
            if args.rank == 0:
                wandb.log({'lora_norm_list-task'+str(task_i): norm_list})
            

        elif args.rehearsal_training:
            print("start directly rehearsal training")
            start_time = time.time()
            remain_epoch = 0
            for remain_epoch in range(args.finetune_rehearsal_epoch):
                if args.distributed:
                    sampler_remain.set_epoch(remain_epoch)
                train_stats = train_one_epoch(model,
                                            criterion,
                                            data_loader_remain,
                                            optimizer_remain,
                                            device,
                                            remain_epoch,
                                            args.clip_max_norm,
                                            ascent=False)
                lr_scheduler_remain.step()

                print("Directly Rehearsal training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_forget,
                    base_ds_forget, device, args.output_dir)
                utils.log_wandb(args=args,array=forget_maps,name='forget', task_i=task_i, epoch=remain_epoch+1)
                print("Directly Rehearsal training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_remain,
                    base_ds_remain, device, args.output_dir)
                utils.log_wandb(args=args,array=remain_maps,name='remain',task_i=task_i, epoch=remain_epoch+1)
                if task_i > 0:
                    print("Directly Rehearsal training. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model, criterion, postprocessors, data_loader_val_old,
                        base_ds_old, device, args.output_dir)
                    utils.log_wandb(args=args,array=old_maps,name='old',task_i=task_i, epoch=remain_epoch+1)
            
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            if dist.get_rank() == 0:
                wandb.log({'time': total_time_str})


        elif args.finetune_gradient_ascent:
            print("start gradient ascent training")
            start_time = time.time()
            forget_epoch = 0
            for forget_epoch in range(args.finetune_forget_epoch):
                if args.distributed:
                    sampler_forget.set_epoch(forget_epoch)
                train_stats = train_one_epoch(model,
                                            criterion,
                                            data_loader_forget,
                                            optimizer_forget,
                                            device,
                                            forget_epoch,
                                            args.clip_max_norm,
                                            ascent=True)
                lr_scheduler_forget.step()

                print("Gradient ascent training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_forget,
                    base_ds_forget, device, args.output_dir)
                utils.log_wandb(args=args,array=forget_maps,name='forget',task_i=task_i,  epoch=forget_epoch+1)
                print("Gradient ascent training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_remain,
                    base_ds_remain, device, args.output_dir)
                utils.log_wandb(args=args,array=remain_maps,name='remain',task_i=task_i, epoch=forget_epoch+1)

                if task_i > 0:
                    print("Gradient ascent training. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model, criterion, postprocessors, data_loader_val_old,
                        base_ds_old, device, args.output_dir)
                    utils.log_wandb(args=args,array=old_maps,name='old',task_i=task_i, epoch=forget_epoch+1)
            
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            if dist.get_rank() == 0:
                wandb.log({'time': total_time_str})
        
        elif args.random_drop:
            dropped_model = random_drop_weights(model_without_ddp)
        
            if args.distributed:
                dropped_ddp_model = torch.nn.parallel.DistributedDataParallel(
                dropped_model, device_ids=[args.gpu])
            
            print("Random drop. Testing for forget classes")
            test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                dropped_ddp_model, criterion, postprocessors, data_loader_val_forget,
                base_ds_forget, device, args.output_dir)
            utils.log_wandb(args=args,array=forget_maps,name='forget',task_i=task_i, epoch=0)
            print("Random drop. Testing for remain classes")
            test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                dropped_ddp_model, criterion, postprocessors, data_loader_val_remain,
                base_ds_remain, device, args.output_dir)
            utils.log_wandb(args=args,array=remain_maps,name='remain',task_i=task_i, epoch=0)
            if task_i > 0:
                print("Random drop. Testing for old classes")
                test_stats_old, coco_evaluator_old, old_maps = evaluate(
                    dropped_ddp_model, criterion, postprocessors, data_loader_val_old,
                    base_ds_old, device, args.output_dir)
                utils.log_wandb(args=args,array=old_maps,name='old',task_i=task_i, epoch=0)
            
        
        else: # two stage training
            print("start two stage training")
            start_time = time.time()

            forget_epoch = 0

            for forget_epoch in range(args.epochs):
                if args.distributed:
                    sampler_forget.set_epoch(forget_epoch)
                train_stats = train_one_epoch(model,
                                            criterion,
                                            data_loader_forget,
                                            optimizer_forget,
                                            device,
                                            forget_epoch,
                                            args.clip_max_norm,
                                            ascent=True)
                lr_scheduler_forget.step()

                print("Forget training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_forget,
                    base_ds_forget, device, args.output_dir)
                utils.log_wandb(args=args,array=forget_maps,name='forget', task_i=task_i, epoch=forget_epoch+1)
                print("Forget training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_remain,
                    base_ds_remain, device, args.output_dir)
                utils.log_wandb(args=args,array=remain_maps,name='remain',task_i=task_i, epoch=forget_epoch+1)

            # rehearsal training
            remain_epoch = 0
            for remain_epoch in range(args.rehearsal_epoch):
                if args.distributed:
                    sampler_remain.set_epoch(remain_epoch)
                train_stats = train_one_epoch(model,
                                            criterion,
                                            data_loader_remain,
                                            optimizer_remain,
                                            device,
                                            remain_epoch,
                                            args.clip_max_norm,
                                            ascent=False)
                lr_scheduler_remain.step()

                print("Rehearsal training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_forget,
                    base_ds_forget, device, args.output_dir)
                utils.log_wandb(args=args,array=forget_maps,name='forget', task_i=task_i, epoch=remain_epoch+1)
                print("Rehearsal training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model, criterion, postprocessors, data_loader_val_remain,
                    base_ds_remain, device, args.output_dir)
                utils.log_wandb(args=args,array=remain_maps,name='remain',task_i=task_i, epoch=remain_epoch+1)

            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                checkpoint_paths_lora = [output_dir / 'checkpoint_lora.pth']
                # print(checkpoint_paths, checkpoint_paths_lora)
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            'model': model_without_ddp.state_dict(),
                            'optimizer_forget': optimizer_forget.state_dict(),
                            'lr_scheduler_forget': lr_scheduler_forget.state_dict(),
                            'optimizer_remain': optimizer_remain.state_dict(),
                            'lr_scheduler_remain': lr_scheduler_remain.state_dict(),
                            'epoch': forget_epoch,
                            'remain_epoch': remain_epoch,
                            'args': args,
                        }, checkpoint_path)

                for checkpoint_path_lora in checkpoint_paths_lora:
                    utils.save_on_master(
                        {'lora': OrderedDict(lora.lora_state_dict(model_without_ddp))},
                        checkpoint_path_lora)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))

    return
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script',
        parents=[get_args_parser()])
    args = parser.parse_args()

    if args.debug_mode:
        args.epochs = 1
        # args.cls_per_phase = 1
        args.batch_size = 1
        args.rehearsal_epoch = 1

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    if args.rank == 0:
        # wandb.run.name = 'forget' + str(args.cls_per_phase) + '-'.join(
        #     args.lora_pos) + str(args.lora_rank) + 'encoderL' + '-'.join(
        #         str(i)
        #         for i in args.lora_encoder_layers) + 'decoderL' + '-'.join(
        #             str(i) for i in args.lora_decoder_layers)
        if args.one_stage:
            wandb.run.name = 'forget-start' + str(args.num_of_first_cls) + '-per' + str(args.cls_per_phase) + '-one-stage'
        if args.eval:
            wandb.run.name = 'eval-' + wandb.run.name
        if args.rehearsal_training:
            wandb.run.name = 'rehearsal-forget' + str(args.cls_per_phase)
        if args.finetune_gradient_ascent:
            wandb.run.name = 'gradient-ascent-forget' + str(args.cls_per_phase)
        if args.random_drop:
            wandb.run.name = 'random-drop-' + str(args.random_drop_rate) + 'forget-' + str(args.cls_per_phase)
        wandb.finish()