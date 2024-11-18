import argparse
import copy
import datetime
import json
import os
import random
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader

import datasets
import datasets.samplers as samplers
import util.cal_norm as cal_norm
import util.misc as utils
from baselines.DERtrain import train_one_epoch_DER
from baselines.FDRtrain import train_one_epoch_FDR
from baselines.SCRUBtrain import train_one_superepoch_SCRUB
from datasets import CLDatasetWrapper, build_dataset, get_coco_api_from_dataset
from datasets.incremental import generate_cls_order
from engine import evaluate
from engine_cl import train_one_epoch_ewc, train_one_epoch_l2, train_one_epoch_mas
from models import build_model
import torch.nn as nn

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    # parser.add_argument('--lr_beta', default=0.01, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--lr_gamma", default=0.1, type=float)
    parser.add_argument("--lr_drop_balanced", default=10, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument("--sgd", action="store_true")

    # Variants of Deformable DETR
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )
    parser.add_argument(
        "--num_feature_levels", default=4, type=int, help="number of feature levels"
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=300, type=int, help="Number of query slots"
    )
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # forget loss coefficients
    parser.add_argument("--forget_cls_loss_coef", default=-2, type=float)
    parser.add_argument("--forget_bbox_loss_coef", default=5, type=float)
    parser.add_argument("--forget_giou_loss_coef", default=2, type=float)

    # ref coefficients are only for this CL-DETR method
    parser.add_argument("--ref_cls_loss_coef", default=2, type=float)
    parser.add_argument("--ref_bbox_loss_coef", default=5, type=float)
    parser.add_argument("--ref_giou_loss_coef", default=2, type=float)
    parser.add_argument("--ref_loss_overall_coef", default=1, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", default="./data/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--cache_mode",
        default=False,
        action="store_true",
        help="whether to cache images on memory",
    )

    # incremental parameters
    parser.add_argument("--num_of_phases", default=2, type=int)
    parser.add_argument("--cls_per_phase", default=40, type=int)
    parser.add_argument("--data_setting", default="tfh", choices=["tfs", "tfh"])
    parser.add_argument("--num_of_first_cls", default=40, type=int)

    # parser.add_argument('--cls_per_phase', default=3, type=int)
    # parser.add_argument('--data_setting', default='tfs', choices=['tfs', 'tfh'])

    parser.add_argument("--seed_cls", default=123, type=int)
    parser.add_argument("--seed_data", default=123, type=int)
    parser.add_argument("--method", default="icarl", choices=["baseline", "icarl"])

    parser.add_argument("--debug_mode", default=False, action="store_true")
    parser.add_argument("--balanced_ft", default=False, action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--debug_flag", default=False, action="store_true")

    # remain(memory) and forget rate of the corresponding dataset
    parser.add_argument(
        "--mem_rate", default=0.1, type=float, help="memory rate of the rehearsal set"
    )
    parser.add_argument(
        "--forget_rate", default=0.1, type=float, help="memory rate of the forget set"
    )
    # LoRA rank
    parser.add_argument(
        "--lora_rank", default=0, type=int, help="rank of the LoRA model"
    )
    parser.add_argument(
        "--lora_reg_rank", default=0, type=int, help="rank of the LoRA regression head"
    )
    parser.add_argument(
        "--lora_cls_rank",
        default=0,
        type=int,
        help="rank of the LoRA classification head",
    )
    # LoRA position
    parser.add_argument(
        "--lora_pos",
        default=["None"],
        nargs="+",
        choices=["encoder", "decoder", "proj_layer", "None"],
        help="position of the LoRA model",
    )
    # LoRA checkpoint
    parser.add_argument(
        "--resume_lora", default=None, type=str, help="checkpoint of the LoRA model"
    )

    # baseline CL method
    parser.add_argument(
        "--l2",
        default=False,
        action="store_true",
        help="whether to use l2 regularization",
    )
    parser.add_argument(
        "--l2_lambda", default=0.01, type=float, help="lambda for l2 regularization"
    )
    parser.add_argument(
        "--ewc", default=False, action="store_true", help="whether to use ewc"
    )
    parser.add_argument("--ewc_lambda", default=0.01, type=float, help="lambda for ewc")
    parser.add_argument(
        "--MAS", default=False, action="store_true", help="whether to use mas"
    )
    parser.add_argument("--mas_lambda", default=0.01, type=float, help="lambda for mas")
    parser.add_argument(
        "--si", default=False, action="store_true", help="whether to use si"
    )
    parser.add_argument("--si_lambda", default=0.01, type=float, help="lambda for si")
    parser.add_argument(
        "--replay", default=False, action="store_true", help="whether to use replay"
    )
    parser.add_argument(
        "--online",
        default=False,
        action="store_true",
        help="whether to use online regularization",
    )
    parser.add_argument(
        "--retrain",
        default=False,
        action="store_true",
        help="whether to retrain the model",
    )
    # ewc args
    parser.add_argument(
        "--n_fisher_sample",
        default=None,
        type=int,
        help="number of samples to estimate fisher",
    )
    # forget learning factor
    parser.add_argument(
        "--beta",
        default=0.04,
        type=float,
        help="beta for balancing the remain and forget loss",
    )
    parser.add_argument(
        "--alpha", default=0.04, type=float, help="alpha for group sparse loss"
    )
    # wandb
    parser.add_argument(
        "--wandb_offline",
        default=False,
        action="store_true",
    )
    # specify the encoder layer to use LoRA
    parser.add_argument("--lora_encoder_layers", nargs="*", default=[], type=int)
    # specify the decoder layer to use LoRA
    parser.add_argument("--lora_decoder_layers", nargs="*", default=[], type=int)
    # manual order for classes
    parser.add_argument("--manual_order", default=False, action="store_true")

    # continual learning args
    parser.add_argument("--num_tasks", type=int)
    parser.add_argument("--this_task_start_cls", default=0, type=int)
    parser.add_argument("--this_task_end_cls", default=0, type=int)
    parser.add_argument("--cl_beta_list", nargs="*", default=[], type=float)

    # baselines args
    parser.add_argument("--SCRUB", default=False, action="store_true")
    parser.add_argument(
        "--sgda_smoothing", default=0.0, type=float, help="smoothing factor for SGDA"
    )
    parser.add_argument("--sgda_gamma", default=0.99, type=float, help="gamma for sgda")
    parser.add_argument(
        "--sgda_alpha", default=0.001, type=float, help="alpha for sgda"
    )
    parser.add_argument(
        "--sgda_learning_rate", default=1e-4, type=float, help="lr for sgda"
    )
    parser.add_argument(
        "--sgda_momentum", default=0.9, type=float, help="momentum for sgda"
    )
    parser.add_argument(
        "--sgda_weight_decay", default=5e-4, type=float, help="weight_decay for sgda"
    )
    parser.add_argument(
        "--SCRUB_superepoch", default=3, type=int, help="superepoch for sgda"
    )
    parser.add_argument(
        "--kd_T", default=2.0, type=float, help="temperature for kd loss"
    )
    parser.add_argument(
        "--scrub_decay_epoch", default=30, type=int, help="decay epoch for sgda"
    )

    # DER method
    parser.add_argument(
        "--Der", default=False, action="store_true", help="whether to use DER"
    )
    parser.add_argument("--DER_lambda", default=0.1, type=float, help="lambda for DER")
    parser.add_argument(
        "--DER_plus", default=False, action="store_true", help="whether to use DER_plus"
    )
    parser.add_argument(
        "--DER_plus_lambda", default=0.1, type=float, help="lambda for DER_plus"
    )
    # FDR method
    parser.add_argument(
        "--FDR", default=False, action="store_true", help="whether to use FDR"
    )
    parser.add_argument("--FDR_lambda", default=0.1, type=float, help="lambda for FDR")

    # few shot setting
    parser.add_argument("--few_shot", default=False, action="store_true")
    parser.add_argument("--few_shot_num", default=8, type=int)
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    print(args)

    if args.rank == 0:
        wandb.login(key="808d6ef02f3a9c448c5641c132830eb0c3c83c2a")
        wandb.init(
            project="forget learning",
            dir=args.output_dir,
            group="cl-baseline",
            mode="offline" if args.wandb_offline else "online",
        )
        wandb.config.update(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # freeze the last regression head and classification head
    for name, param in model.named_parameters():
        if "bbox_embed" in name and not args.retrain:
            param.requires_grad = False
        if "class_embed" in name and not args.retrain:
            param.requires_grad = False

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    print("ratio of params:", n_parameters / 39847265)
    if args.rank == 0:
        wandb.log(
            {
                "number of params": n_parameters,
                "ratio of params": n_parameters / 39847265,
            }
        )

    cls_order = generate_cls_order(seed=args.seed_cls, manual=args.manual_order)

    if args.data_setting == "tfs":
        total_phase_num = args.num_of_phases
    elif args.data_setting == "tfh":
        total_phase_num = args.num_of_phases
    else:
        raise ValueError("Please set the correct data setting.")

    origin_first_task_cls = args.num_of_first_cls

    regularization_terms = {}  # for regularization terms

    if args.Der:
        teacher_model = copy.deepcopy(model)
        teacher_model.to(device)
        teacher_model.eval()
        # freeze the teacher model
        for param in teacher_model.parameters():
            param.requires_grad = False

    if args.FDR:
        teacher_model = copy.deepcopy(model)
        teacher_model.to(device)
        teacher_model.eval()
        # freeze the teacher model
        for param in teacher_model.parameters():
            param.requires_grad = False

    if args.SCRUB:
        teacher_model = copy.deepcopy(model)
        teacher_model.to(device)
        teacher_model.eval()

        beta = 0.1

        def avg_fn(averaged_model_parameter, model_parameter):
            return (1 - beta) * averaged_model_parameter + beta * model_parameter

        swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=avg_fn)
        swa_model = swa_model.to(device)

    # TODO: need to add origin dataset for the first task
    # origin_dataset =
    for task_i in range(args.num_tasks):  # start from 0
        # modify num_of_first_cls according to task id
        print("\n")
        print("*****************task_i:", task_i, "***********************")
        print("\n")

        args.num_of_first_cls = origin_first_task_cls - task_i * args.cls_per_phase
        print("\033[91mFor remain dataset:\033[0m")
        print("num_of_first_cls:", args.num_of_first_cls, "select [: num_of_first_cls]")
        print("")
        dataset_remain = build_dataset(
            image_set="train",
            args=args,
            cls_order=cls_order,
            phase_idx=0,
            incremental=True,
            incremental_val=False,
            val_each_phase=False,
            is_rehearsal=True,
        )

        args.this_task_start_cls = origin_first_task_cls - task_i * args.cls_per_phase
        args.this_task_end_cls = args.this_task_start_cls + args.cls_per_phase
        print("\033[91mFor forget dataset:\033[0m")
        print("this task start cls:", args.this_task_start_cls)
        print("this task end cls:", args.this_task_end_cls)
        print("select [this_task_start_cls: this_task_end_cls]")
        print("")
        dataset_forget = build_dataset(
            image_set="train",
            args=args,
            cls_order=cls_order,
            phase_idx=1,
            incremental=True,
            incremental_val=False,
            val_each_phase=False,
        )
        print("\033[91mFor val dataset:\033[0m")
        dataset_val_remain = build_dataset(
            image_set="val",
            args=args,
            cls_order=cls_order,
            phase_idx=0,
            incremental=True,
            incremental_val=True,
            val_each_phase=True,
        )
        dataset_val_forget = build_dataset(
            image_set="val",
            args=args,
            cls_order=cls_order,
            phase_idx=1,
            incremental=True,
            incremental_val=True,
            val_each_phase=True,
        )
        dataset_cl_forget = CLDatasetWrapper(dataset_forget)

        if task_i > 0:
            # val already forget dataset in the previous tasks: 60-10-10(last 10 is forget in the previous tasks)
            args.this_task_start_cls = (
                origin_first_task_cls - (task_i - 1) * args.cls_per_phase
            )
            dataset_val_old = build_dataset(
                image_set="val",
                args=args,
                cls_order=cls_order,
                phase_idx=1,
                incremental=True,
                incremental_val=True,
                val_each_phase=False,
                is_rehearsal=True,
            )
        if args.replay:
            print("concat the dataset...")
            dataset_total = torch.utils.data.ConcatDataset(
                [dataset_remain, dataset_cl_forget]
            )

        if args.distributed:
            if args.cache_mode:
                sampler_forget = samplers.NodeDistributedSampler(dataset_forget)
                sampler_remain = samplers.NodeDistributedSampler(
                    dataset_remain
                )  # shuffle=True by default
                sampler_val_forget = samplers.NodeDistributedSampler(
                    dataset_val_forget, shuffle=False
                )
                sampler_val_remain = samplers.NodeDistributedSampler(
                    dataset_val_remain, shuffle=False
                )
                sampler_cl_forget = samplers.NodeDistributedSampler(dataset_cl_forget)

                if args.replay:
                    sampler_total = samplers.NodeDistributedSampler(dataset_total)

                if task_i > 0:
                    sampler_val_old = samplers.NodeDistributedSampler(
                        dataset_val_old, shuffle=False
                    )
            else:
                sampler_forget = samplers.DistributedSampler(dataset_forget)
                sampler_remain = samplers.DistributedSampler(dataset_remain)
                sampler_val_forget = samplers.DistributedSampler(
                    dataset_val_forget, shuffle=False
                )
                sampler_val_remain = samplers.DistributedSampler(
                    dataset_val_remain, shuffle=False
                )
                sampler_cl_forget = samplers.DistributedSampler(dataset_cl_forget)

                if args.replay:
                    sampler_total = samplers.DistributedSampler(dataset_total)

                if task_i > 0:
                    sampler_val_old = samplers.DistributedSampler(
                        dataset_val_old, shuffle=False
                    )
        else:
            sampler_forget = torch.utils.data.RandomSampler(dataset_forget)
            sampler_remain = torch.utils.data.RandomSampler(dataset_remain)
            sampler_val_forget = torch.utils.data.SequentialSampler(dataset_val_forget)
            sampler_val_remain = torch.utils.data.SequentialSampler(dataset_val_remain)
            sampler_cl_forget = torch.utils.data.RandomSampler(dataset_cl_forget)

            if args.replay:
                sampler_total = torch.utils.data.RandomSampler(dataset_total)

            if task_i > 0:
                sampler_val_old = torch.utils.data.SequentialSampler(dataset_val_old)

        batch_sampler_forget = torch.utils.data.BatchSampler(
            sampler_forget, args.batch_size, drop_last=True
        )
        batch_sampler_remain = torch.utils.data.BatchSampler(
            sampler_remain, args.batch_size, drop_last=True
        )
        batch_sampler_cl_forget = torch.utils.data.BatchSampler(
            sampler_cl_forget, args.batch_size, drop_last=True
        )
        if args.replay:
            batch_sampler_total = torch.utils.data.BatchSampler(
                sampler_total, args.batch_size, drop_last=True
            )

        data_loader_forget = DataLoader(
            dataset_forget,
            batch_sampler=batch_sampler_forget,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        data_loader_remain = DataLoader(
            dataset_remain,
            batch_sampler=batch_sampler_remain,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        data_loader_val_forget = DataLoader(
            dataset_val_forget,
            batch_size=args.batch_size,
            sampler=sampler_val_forget,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        data_loader_val_remain = DataLoader(
            dataset_val_remain,
            batch_size=args.batch_size,
            sampler=sampler_val_remain,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        data_loader_cl_forget = DataLoader(
            dataset_cl_forget,
            batch_sampler=batch_sampler_cl_forget,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if task_i == 0:
            # we need a dataloader that combines all the classes to calculate the importance matrix
            dataset_importance = dataset_remain
            if args.distributed:
                if args.cache_mode:
                    sampler_importance = samplers.NodeDistributedSampler(
                        dataset_importance
                    )
                else:
                    sampler_importance = samplers.DistributedSampler(dataset_importance)
            else:
                sampler_importance = torch.utils.data.RandomSampler(dataset_importance)
            batch_sampler_importance = torch.utils.data.BatchSampler(
                sampler_importance, args.batch_size, drop_last=False
            )
            data_loader_importance = DataLoader(
                dataset_importance,
                batch_sampler=batch_sampler_importance,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        elif task_i < args.num_tasks - 1:
            args_importance = copy.deepcopy(args)
            args_importance.num_of_first_cls = (
                origin_first_task_cls - (task_i + 1) * args.cls_per_phase
            )
            dataset_importance = build_dataset(
                image_set="train",
                args=args_importance,
                cls_order=cls_order,
                phase_idx=0,
                incremental=True,
                incremental_val=False,
                val_each_phase=False,
                is_rehearsal=True,
            )
            if args.distributed:
                if args.cache_mode:
                    sampler_importance = samplers.NodeDistributedSampler(
                        dataset_importance
                    )
                else:
                    sampler_importance = samplers.DistributedSampler(dataset_importance)
            else:
                sampler_importance = torch.utils.data.RandomSampler(dataset_importance)
            batch_sampler_importance = torch.utils.data.BatchSampler(
                sampler_importance, args.batch_size, drop_last=False
            )
            data_loader_importance = DataLoader(
                dataset_importance,
                batch_sampler=batch_sampler_importance,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        else:
            print("This is the last task, no need to calculate the importance matrix")

        if args.replay:
            data_loader_total = DataLoader(
                dataset_total,
                batch_sampler=batch_sampler_total,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
                pin_memory=True,
            )

        if task_i > 0:
            data_loader_val_old = DataLoader(
                dataset_val_old,
                batch_size=args.batch_size,
                sampler=sampler_val_old,
                drop_last=False,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
                pin_memory=True,
            )

        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        for n, p in model_without_ddp.named_parameters():
            print(n, p.requires_grad)

        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names)
                    and not match_name_keywords(n, args.lr_linear_proj_names)
                    and p.requires_grad
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_backbone_names)
                    and p.requires_grad
                ],
                "lr": args.lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_linear_proj_names)
                    and p.requires_grad
                ],
                "lr": args.lr * args.lr_linear_proj_mult,
            },
        ]

        print("setting the optimizer...")
        if args.sgd:
            optimizer = torch.optim.SGD(
                param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                param_dicts, lr=args.lr, weight_decay=args.weight_decay
            )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.lr_drop, gamma=args.lr_gamma
        )

        if task_i == 0:
            print("pytorch model distributed...")
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu]
                )
                model_without_ddp = model.module

        # eval时候才用base_ds，所以训练时候dataset不需要
        base_ds_forget = get_coco_api_from_dataset(dataset_val_forget)
        base_ds_remain = get_coco_api_from_dataset(dataset_val_remain)

        if task_i > 0:
            base_ds_old = get_coco_api_from_dataset(dataset_val_old)

        if args.frozen_weights is not None:
            checkpoint = torch.load(args.frozen_weights, map_location="cpu")
            model_without_ddp.detr.load_state_dict(checkpoint["model"])

        output_dir = Path(args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # 如果args.resume为True，且为第一个任务，则加载预训练模型
        if args.resume and task_i == 0:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                # load DETR weights
                checkpoint = torch.load(args.resume, map_location="cpu")
                missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
                    checkpoint["model"], strict=False
                )
                unexpected_keys = [
                    k
                    for k in unexpected_keys
                    if not (k.endswith("total_params") or k.endswith("total_ops"))
                ]

                if len(missing_keys) > 0:
                    print("Missing Keys: {}".format(missing_keys))
                if len(unexpected_keys) > 0:
                    print("Unexpected Keys: {}".format(unexpected_keys))

            # check the resumed model
            if (not args.eval) and (not args.debug_flag):
                print("Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_forget,
                    base_ds_forget,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=forget_maps,
                    task_i="resumed-eval",
                    name="forget",
                    epoch=0,
                )
                print("Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_remain,
                    base_ds_remain,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=remain_maps,
                    task_i="resumed-eval",
                    name="remain",
                    epoch=0,
                )
                if task_i > 0:
                    print("Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model,
                        criterion,
                        postprocessors,
                        data_loader_val_old,
                        base_ds_old,
                        device,
                        args.output_dir,
                    )
                    utils.log_wandb(
                        args=args,
                        array=old_maps,
                        name="old",
                        task_i="resumed-eval",
                        epoch=0,
                    )

        if args.eval:
            print("Testing for forget classes")
            test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val_forget,
                base_ds_forget,
                device,
                args.output_dir,
            )
            utils.log_wandb(
                args=args, array=forget_maps, name="forget", task_i="eval", epoch=0
            )

            print("Testing for remain classes")
            test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val_remain,
                base_ds_remain,
                device,
                args.output_dir,
            )
            utils.log_wandb(
                args=args, array=remain_maps, name="remain", task_i="eval", epoch=0
            )
            return

        parms_without_ddp = {
            n: p for n, p in model_without_ddp.named_parameters() if p.requires_grad
        }  # for convenience

        # choose the training method, l2, ewc, mas, si, replay+ewc, replay+mas, replay+si, replay+l2
        if args.retrain:
            print("retraining baseline", task_i)
            start_time = time.time()
            if args.sgd:
                optimizer = torch.optim.SGD(
                    param_dicts,
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    param_dicts, lr=args.lr, weight_decay=args.weight_decay
                )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

            # reinitialize the model
            model_without_ddp = build_model(args)[0]
            model_without_ddp.to(device)

            model_without_ddp = model.module

            retrain_epoch = 0
            for retrain_epoch in range(args.epochs):
                sampler_cl_forget.set_epoch(retrain_epoch)

                train_stats = train_one_epoch_l2(
                    model=model,
                    criterion=criterion,
                    data_loader_cl_forget=data_loader_remain,
                    device=device,
                    optimizer=optimizer,
                    epoch=retrain_epoch,
                    clip_max_norm=args.clip_max_norm,
                    l2_lambda=0,
                    regularization_terms=None,
                )
                lr_scheduler.step()

                print("retraining baseline. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_forget,
                    base_ds_forget,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=forget_maps,
                    name="forget",
                    task_i=task_i,
                    epoch=retrain_epoch + 1,
                )
                print("retraining baseline. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_remain,
                    base_ds_remain,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=remain_maps,
                    name="remain",
                    task_i=task_i,
                    epoch=retrain_epoch + 1,
                )

                if task_i > 0:
                    print("retraining baseline. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model,
                        criterion,
                        postprocessors,
                        data_loader_val_old,
                        base_ds_old,
                        device,
                        args.output_dir,
                    )
                    utils.log_wandb(
                        args=args,
                        array=old_maps,
                        name="old",
                        task_i=task_i,
                        epoch=retrain_epoch + 1,
                    )
            end_time = time.time()
            total_time = end_time - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            if args.rank == 0:
                wandb.log({"time": total_time_str})

        if args.l2:
            print("start l2 training in", task_i)
            start_time = time.time()

            # reinitialize the optimizer
            if args.sgd:
                optimizer = torch.optim.SGD(
                    param_dicts,
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    param_dicts, lr=args.lr, weight_decay=args.weight_decay
                )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

            if (
                task_i == 0
            ):  # the pretrained task, we don't need to train, only calculate the importance matrix
                # 1.Backup the weight of current task
                task_param = {}
                model_without_ddp = model.module
                for n, p in model_without_ddp.named_parameters():
                    if p.requires_grad:
                        task_param[n] = p.clone().detach()

                def calculate_importance_l2(model, dataloader):
                    # Use an identity importance so it is an L2 regularization.
                    importance = {}
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            importance[n] = p.clone().detach().fill_(1)  # Identity
                    return importance

                # 2. calculate the importance matrix and get the regularization terms
                importance = calculate_importance_l2(
                    model_without_ddp, data_loader_importance
                )
                regularization_terms[0] = {
                    "importance": importance,
                    "task_param": task_param,
                }

            # 1. learn the current task
            l2_epoch = 0
            for l2_epoch in range(args.epochs):
                sampler_cl_forget.set_epoch(l2_epoch)
                if args.replay:
                    train_stats = train_one_epoch_l2(
                        model=model,
                        criterion=criterion,
                        data_loader_cl_forget=data_loader_total,
                        device=device,
                        optimizer=optimizer,
                        epoch=l2_epoch,
                        clip_max_norm=args.clip_max_norm,
                        l2_lambda=args.l2_lambda,
                        regularization_terms=regularization_terms,
                    )
                else:  # no replay
                    train_stats = train_one_epoch_l2(
                        model=model,
                        criterion=criterion,
                        data_loader_cl_forget=data_loader_cl_forget,
                        device=device,
                        optimizer=optimizer,
                        epoch=l2_epoch,
                        clip_max_norm=args.clip_max_norm,
                        l2_lambda=args.l2_lambda,
                        regularization_terms=regularization_terms,
                    )

                lr_scheduler.step()

                print("l2 regularization training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_forget,
                    base_ds_forget,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=forget_maps,
                    name="forget",
                    task_i=task_i,
                    epoch=l2_epoch + 1,
                )
                print("l2 regularization training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_remain,
                    base_ds_remain,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=remain_maps,
                    name="remain",
                    task_i=task_i,
                    epoch=l2_epoch + 1,
                )

                if task_i > 0:
                    print("l2 regularization training. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model,
                        criterion,
                        postprocessors,
                        data_loader_val_old,
                        base_ds_old,
                        device,
                        args.output_dir,
                    )
                    utils.log_wandb(
                        args=args,
                        array=old_maps,
                        name="old",
                        task_i=task_i,
                        epoch=l2_epoch + 1,
                    )
            # 2. Backup the weight of current task
            task_param = {}
            model_without_ddp = model.module
            for n, p in model_without_ddp.named_parameters():
                if p.requires_grad:
                    task_param[n] = p.clone().detach()

            if task_i < args.num_tasks - 1:
                # 3. calculate the importance matrix and get the regularization terms
                importance = calculate_importance_l2(
                    model_without_ddp, data_loader_remain
                )
                if (
                    args.online and len(regularization_terms) > 0
                ):  # only have one importance matrix
                    regularization_terms[0] = {
                        "importance": importance,
                        "task_param": task_param,
                    }
                else:
                    regularization_terms[task_i + 1] = {
                        "importance": importance,
                        "task_param": task_param,
                    }
            else:
                print(
                    "This is the last task, no need to calculate the importance matrix"
                )

            end_time = time.time()
            total_time = end_time - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            if args.rank == 0:
                wandb.log({"time": total_time_str})

        if args.ewc:
            print("start ewc training in task", task_i)
            start_time = time.time()

            # reinitialize the optimizer
            if args.sgd:
                optimizer = torch.optim.SGD(
                    param_dicts,
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    param_dicts, lr=args.lr, weight_decay=args.weight_decay
                )

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

            if (
                task_i == 0
            ):  # the pretrained task, we don't need to train, only calculate the importance matrix
                # 1.Backup the weight of current task
                task_param = {}
                model_without_ddp = model.module
                for n, p in model_without_ddp.named_parameters():
                    if p.requires_grad:
                        task_param[n] = p.clone().detach()

                def calculate_importance_ewc(model, dataloader):
                    print("calculate importance of ewc...")

                    # Initialize the importance matrix
                    if args.online and len(regularization_terms) > 0:
                        importance = regularization_terms[0]["importance"]
                    else:
                        importance = {}
                        for n, p in model_without_ddp.named_parameters():
                            if p.requires_grad:
                                importance[n] = (
                                    p.clone().detach().fill_(0)
                                )  # zero initialization

                    # Sample a subset (n_fisher_sample) of data to estimate the importance matrix (batch size is 1)
                    # Otherwise it uses mini-batch to estimate the importance matrix. This speeds up the process a lot with similar performance.

                    if args.n_fisher_sample is not None:
                        # FIXME: there is a bug RuntimeError:  Trying to resize storage that is not resizable
                        n_sample = min(args.n_fisher_sample, len(dataloader.dataset))
                        print(
                            "Sample",
                            args.n_fisher_sample,
                            "data to estimate the importance matrix",
                        )
                        rand_ind = random.sample(
                            list(range(len(dataloader.dataset))), n_sample
                        )
                        subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
                        subdataloader = DataLoader(
                            subdata,
                            batch_size=2,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                        )
                    else:
                        subdataloader = dataloader

                    model.eval()
                    # Accumulate the square of gradients
                    for i, (samples, targets) in enumerate(subdataloader):
                        samples = samples.to(device)

                        targets = [
                            {k: v.to(device) for k, v in t.items()} for t in targets
                        ]
                        # if dist.get_rank()==0:
                        #     import pdb; pdb.set_trace()
                        outputs = model(samples)  # outputs is a list of dicts
                        loss_dict = criterion(outputs, targets)  # loss_dict is a dict
                        weight_dict = criterion.weight_dict
                        losses = sum(
                            loss_dict[k] * weight_dict[k]
                            for k in loss_dict.keys()
                            if k in weight_dict
                        )
                        model.zero_grad()
                        losses.backward()
                        for n, p in importance.items():
                            if (
                                parms_without_ddp[n].grad is not None
                            ):  # some parameters may not have grad
                                p += (
                                    (parms_without_ddp[n].grad ** 2)
                                    * len(samples.tensors)
                                    / len(subdataloader)
                                )

                    model.train()
                    return importance

                importance = calculate_importance_ewc(
                    model_without_ddp, data_loader_importance
                )
                regularization_terms[0] = {
                    "importance": importance,
                    "task_param": task_param,
                }

            # 1. learn the current task
            ewc_epoch = 0
            for ewc_epoch in range(args.epochs):
                sampler_cl_forget.set_epoch(ewc_epoch)
                if args.replay:
                    train_stats = train_one_epoch_ewc(
                        model=model,
                        criterion=criterion,
                        data_loader_cl_forget=data_loader_total,
                        device=device,
                        optimizer=optimizer,
                        epoch=ewc_epoch,
                        clip_max_norm=args.clip_max_norm,
                        ewc_lambda=args.ewc_lambda,
                        regularization_terms=regularization_terms,
                    )

                else:  # no replay
                    train_stats = train_one_epoch_ewc(
                        model=model,
                        criterion=criterion,
                        data_loader_cl_forget=data_loader_cl_forget,
                        device=device,
                        optimizer=optimizer,
                        epoch=ewc_epoch,
                        clip_max_norm=args.clip_max_norm,
                        ewc_lambda=args.ewc_lambda,
                        regularization_terms=regularization_terms,
                    )

                lr_scheduler.step()

                print("ewc regularization training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_forget,
                    base_ds_forget,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=forget_maps,
                    name="forget",
                    task_i=task_i,
                    epoch=ewc_epoch + 1,
                )
                print("ewc regularization training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_remain,
                    base_ds_remain,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=remain_maps,
                    name="remain",
                    task_i=task_i,
                    epoch=ewc_epoch + 1,
                )

                if task_i > 0:
                    print("ewc regularization training. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model,
                        criterion,
                        postprocessors,
                        data_loader_val_old,
                        base_ds_old,
                        device,
                        args.output_dir,
                    )
                    utils.log_wandb(
                        args=args,
                        array=old_maps,
                        name="old",
                        task_i=task_i,
                        epoch=ewc_epoch + 1,
                    )
            # 2. Backup the weight of current task
            task_param = {}
            model_without_ddp = model.module
            for n, p in model_without_ddp.named_parameters():
                if p.requires_grad:
                    task_param[n] = p.clone().detach()

            if task_i < args.num_tasks - 1:
                # 3. calculate the importance matrix and get the regularization terms
                importance = calculate_importance_ewc(
                    model_without_ddp, data_loader_remain
                )
                if (
                    args.online and len(regularization_terms) > 0
                ):  # only have one importance matrix
                    regularization_terms[0] = {
                        "importance": importance,
                        "task_param": task_param,
                    }
                else:
                    regularization_terms[task_i + 1] = {
                        "importance": importance,
                        "task_param": task_param,
                    }
            else:
                print(
                    "This is the last task, no need to calculate the importance matrix"
                )

            end_time = time.time()
            total_time = end_time - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            if args.rank == 0:
                wandb.log({"time": total_time_str})

        if args.Der:
            #  如果模型只用于推理，不需要 DDP，teacher model不需要
            if args.DER_plus:
                print("start DER++ training in task", task_i)
            else:
                print("start DER training in task", task_i)

            start_time = time.time()

            # reinitialize the optimizer
            if args.sgd:
                optimizer = torch.optim.SGD(
                    param_dicts,
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    param_dicts, lr=args.lr, weight_decay=args.weight_decay
                )

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

            epoch = 0
            for epoch in range(args.epochs):
                if args.distributed:
                    sampler_forget.set_epoch(epoch)
                    sampler_remain.set_epoch(epoch)

                train_stats = train_one_epoch_DER(
                    student_model=model,
                    teacher_model=teacher_model,
                    criterion=criterion,
                    data_loader_forget=data_loader_forget,
                    data_loader_remain=data_loader_remain,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    max_norm=args.clip_max_norm,
                    lambda_der=args.DER_lambda,
                    plus=args.DER_plus,
                    lambda_der_plus=args.DER_plus_lambda,
                )
                lr_scheduler.step()

                print("DER training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_forget,
                    base_ds_forget,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=forget_maps,
                    name="forget",
                    task_i=task_i,
                    epoch=epoch + 1,
                )

                print("DER training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_remain,
                    base_ds_remain,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=remain_maps,
                    name="remain",
                    task_i=task_i,
                    epoch=epoch + 1,
                )

                if task_i > 0:
                    print("DER training. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model,
                        criterion,
                        postprocessors,
                        data_loader_val_old,
                        base_ds_old,
                        device,
                        args.output_dir,
                    )
                    utils.log_wandb(
                        args=args,
                        array=old_maps,
                        name="old",
                        task_i=task_i,
                        epoch=epoch + 1,
                    )

            end_time = time.time()
            total_time = end_time - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            if args.rank == 0:
                wandb.log({"time": total_time_str})

        if args.FDR:
            print("start FDR training in task", task_i)
            start_time = time.time()

            # reinitialize the optimizer
            if args.sgd:
                optimizer = torch.optim.SGD(
                    param_dicts,
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    param_dicts, lr=args.lr, weight_decay=args.weight_decay
                )

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

            epoch = 0
            for epoch in range(args.epochs):
                if args.distributed:
                    sampler_forget.set_epoch(epoch)
                    sampler_remain.set_epoch(epoch)

                train_stats = train_one_epoch_FDR(
                    student_model=model,
                    teacher_model=teacher_model,
                    criterion=criterion,
                    data_loader_forget=data_loader_forget,
                    data_loader_remain=data_loader_remain,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    max_norm=args.clip_max_norm,
                    reg_lambda=args.FDR_lambda,
                )
                lr_scheduler.step()

                print("FDR training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_forget,
                    base_ds_forget,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=forget_maps,
                    name="forget",
                    task_i=task_i,
                    epoch=epoch + 1,
                )

                print("FDR training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_remain,
                    base_ds_remain,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=remain_maps,
                    name="remain",
                    task_i=task_i,
                    epoch=epoch + 1,
                )

                if task_i > 0:
                    print("FDR training. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model,
                        criterion,
                        postprocessors,
                        data_loader_val_old,
                        base_ds_old,
                        device,
                        args.output_dir,
                    )
                    utils.log_wandb(
                        args=args,
                        array=old_maps,
                        name="old",
                        task_i=task_i,
                        epoch=epoch + 1,
                    )

            end_time = time.time()
            total_time = end_time - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            if args.rank == 0:
                wandb.log({"time": total_time_str})

        if args.SCRUB:
            print("start SCRUB training in task", task_i)
            if args.sgda_smoothing > 0:
                print("smoothing with SGDA")
            start_time = time.time()

            # reinitialize the optimizer
            if args.sgd:
                optimizer = torch.optim.SGD(
                    param_dicts,
                    lr=args.sgda_learning_rate,
                    momentum=args.sgda_momentum,
                    weight_decay=args.sgda_weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    param_dicts,
                    lr=args.sgda_learning_rate,
                    weight_decay=args.sgda_weight_decay,
                )

            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop) # do not use lr_scheduler for SCRUB

            epoch = 0
            for epoch in range(args.SCRUB_superepoch):
                if args.distributed:
                    sampler_forget.set_epoch(epoch)
                    sampler_remain.set_epoch(epoch)

                train_stats = train_one_superepoch_SCRUB(
                    student_model=model,
                    teacher_model=teacher_model,
                    swa_model=swa_model,
                    criterion=criterion,
                    data_loader_forget=data_loader_forget,
                    data_loader_remain=data_loader_remain,
                    optimizer=optimizer,
                    device=device,
                    superepoch=epoch,
                    max_norm=args.clip_max_norm,
                    kd_T=args.kd_T,
                    sgda_smoothing=args.sgda_smoothing,
                    sgda_alpha=args.sgda_alpha,
                    sgda_gamma=args.sgda_gamma,
                    sgda_lr=args.sgda_learning_rate,
                    lr_decay_epochs=args.scrub_decay_epoch,
                )

                print("SCRUB training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_forget,
                    base_ds_forget,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=forget_maps,
                    name="forget",
                    task_i=task_i,
                    epoch=epoch + 1,
                )

                print("SCRUB training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_remain,
                    base_ds_remain,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=remain_maps,
                    name="remain",
                    task_i=task_i,
                    epoch=epoch + 1,
                )

                if task_i > 0:
                    print("SCRUB training. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model,
                        criterion,
                        postprocessors,
                        data_loader_val_old,
                        base_ds_old,
                        device,
                        args.output_dir,
                    )
                    utils.log_wandb(
                        args=args,
                        array=old_maps,
                        name="old",
                        task_i=task_i,
                        epoch=epoch + 1,
                    )

            end_time = time.time()
            total_time = end_time - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            if args.rank == 0:
                wandb.log({"time": total_time_str})

        if args.MAS:
            print("start mas training in task", task_i)
            start_time = time.time()

            # reinitialize the optimizer
            if args.sgd:
                optimizer = torch.optim.SGD(
                    param_dicts,
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    param_dicts, lr=args.lr, weight_decay=args.weight_decay
                )

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

            if (
                task_i == 0
            ):  # the pretrained task, we don't need to train, only calculate the importance matrix
                # 1.Backup the weight of current task
                task_param = {}
                model_without_ddp = model.module
                for n, p in model_without_ddp.named_parameters():
                    if p.requires_grad:
                        task_param[n] = p.clone().detach()

                def calculate_importance_mas(model, dataloader):
                    print("calculate importance of mas...")

                    # Initialize the importance matrix
                    if args.online and len(regularization_terms) > 0:
                        importance = regularization_terms[0]["importance"]
                    else:
                        importance = {}
                        for n, p in model_without_ddp.named_parameters():
                            if p.requires_grad:
                                importance[n] = (
                                    p.clone().detach().fill_(0)
                                )  # zero initialization

                    model.eval()

                    # 网络输出logits的L2范数的平方作为loss，对其求偏导，得到梯度
                    for i, (samples, targets) in enumerate(dataloader):
                        samples = samples.to(device)
                        targets = [
                            {k: v.to(device) for k, v in t.items()} for t in targets
                        ]

                        outputs = model(samples)  # outputs is a list of dicts
                        outputs_logit = outputs["pred_logits"]
                        outputs_logit.pow_(2)
                        loss = outputs_logit.mean()

                        model.zero_grad()
                        loss.backward()

                        for n, p in importance.items():
                            if parms_without_ddp[n].grad is not None:
                                p += parms_without_ddp[n].grad.abs() / len(dataloader)

                    model.train()
                    return importance

                importance = calculate_importance_mas(
                    model_without_ddp, data_loader_importance
                )
                regularization_terms[0] = {
                    "importance": importance,
                    "task_param": task_param,
                }

            # 1. learn the current task
            mas_epoch = 0
            for mas_epoch in range(args.epochs):
                sampler_cl_forget.set_epoch(mas_epoch)
                if args.replay:
                    train_stats = train_one_epoch_mas(
                        model=model,
                        criterion=criterion,
                        data_loader_cl_forget=data_loader_total,
                        device=device,
                        optimizer=optimizer,
                        epoch=mas_epoch,
                        clip_max_norm=args.clip_max_norm,
                        mas_lambda=args.mas_lambda,
                        regularization_terms=regularization_terms,
                    )

                else:
                    train_stats = train_one_epoch_mas(
                        model=model,
                        criterion=criterion,
                        data_loader_cl_forget=data_loader_cl_forget,
                        device=device,
                        optimizer=optimizer,
                        epoch=mas_epoch,
                        clip_max_norm=args.clip_max_norm,
                        mas_lambda=args.mas_lambda,
                        regularization_terms=regularization_terms,
                    )

                lr_scheduler.step()

                print("mas regularization training. Testing for forget classes")
                test_stats_forget, coco_evaluator_forget, forget_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_forget,
                    base_ds_forget,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=forget_maps,
                    name="forget",
                    task_i=task_i,
                    epoch=mas_epoch + 1,
                )
                print("mas regularization training. Testing for remain classes")
                test_stats_remain, coco_evaluator_remain, remain_maps = evaluate(
                    model,
                    criterion,
                    postprocessors,
                    data_loader_val_remain,
                    base_ds_remain,
                    device,
                    args.output_dir,
                )
                utils.log_wandb(
                    args=args,
                    array=remain_maps,
                    name="remain",
                    task_i=task_i,
                    epoch=mas_epoch + 1,
                )

                if task_i > 0:
                    print("mas regularization training. Testing for old classes")
                    test_stats_old, coco_evaluator_old, old_maps = evaluate(
                        model,
                        criterion,
                        postprocessors,
                        data_loader_val_old,
                        base_ds_old,
                        device,
                        args.output_dir,
                    )
                    utils.log_wandb(
                        args=args,
                        array=old_maps,
                        name="old",
                        task_i=task_i,
                        epoch=mas_epoch + 1,
                    )

            # 2. Backup the weight of current task
            task_param = {}
            model_without_ddp = model.module
            for n, p in model_without_ddp.named_parameters():
                if p.requires_grad:
                    task_param[n] = p.clone().detach()

            if task_i < args.num_tasks - 1:
                # 3. calculate the importance matrix and get the regularization terms
                importance = calculate_importance_mas(
                    model_without_ddp, data_loader_remain
                )
                if args.online and len(regularization_terms) > 0:
                    regularization_terms[0] = {
                        "importance": importance,
                        "task_param": task_param,
                    }
                else:
                    regularization_terms[task_i + 1] = {
                        "importance": importance,
                        "task_param": task_param,
                    }
            else:
                print(
                    "This is the last task, no need to calculate the importance matrix"
                )

            end_time = time.time()
            total_time = end_time - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))
            if args.rank == 0:
                wandb.log({"time": total_time_str})

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deformable DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    if args.debug_flag:
        args.wandb_offline = True

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    if args.rank == 0:
        if args.eval:
            wandb.run.name = "eval-" + args.resume
        if args.l2:
            wandb.run.name = (
                "forget-start"
                + str(args.num_of_first_cls)
                + "-per"
                + str(args.cls_per_phase)
                + "-l2"
            )
        if args.ewc:
            wandb.run.name = (
                "forget-start"
                + str(args.num_of_first_cls)
                + "-per"
                + str(args.cls_per_phase)
                + "-ewc"
            )
        if args.MAS:
            wandb.run.name = (
                "forget-start"
                + str(args.num_of_first_cls)
                + "-per"
                + str(args.cls_per_phase)
                + "-mas"
            )
        if args.FDR:
            wandb.run.name = (
                "forget-start"
                + str(args.num_of_first_cls)
                + "-per"
                + str(args.cls_per_phase)
                + "-FDR"
            )
        if args.Der:
            wandb.run.name = (
                "forget-start"
                + str(args.num_of_first_cls)
                + "-per"
                + str(args.cls_per_phase)
                + "-DER"
            )
            if args.DER_plus:
                wandb.run.name += "++"
        if args.SCRUB:
            wandb.run.name = (
                "forget-start"
                + str(args.num_of_first_cls)
                + "-per"
                + str(args.cls_per_phase)
                + "-SCRUB"
            )
            if args.sgda_smoothing > 0:
                wandb.run.name += "-smooth" + str(args.sgda_smoothing)
        if args.retrain:
            wandb.run.name = (
                "forget-start"
                + str(args.num_of_first_cls)
                + "-per"
                + str(args.cls_per_phase)
                + "-retrain"
            )
        if args.replay:
            wandb.run.name += "-replay"
        if args.online:
            wandb.run.name += "-online"
        wandb.finish()
