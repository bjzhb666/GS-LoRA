"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from util import box_ops
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import numpy as np
import torch.distributed as dist
import wandb

def get_l2_loss(model, regularization_terms, l2_lambda=0.01):
    if regularization_terms is None:
        return torch.tensor(0.0, device=model.device)
    l2_loss = torch.tensor(0.0, device=model.device)
    model_without_ddp = model.module
    reg_loss = torch.tensor(0.0, device=model.device)
    for i, reg_term in regularization_terms.items():
        task_reg_loss = torch.tensor(0.0, device=model.device)
        importance = reg_term['importance']
        task_param = reg_term['task_param']
        
        for n, p in model_without_ddp.named_parameters():
            if p.requires_grad:
                task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
        reg_loss += task_reg_loss 
    l2_loss += l2_lambda * reg_loss
    return l2_loss


def train_one_epoch_l2(model:torch.nn.Module,
                    criterion:torch.nn.Module,
                    data_loader_cl_forget:Iterable,
                    optimizer:torch.optim.Optimizer,
                    device:torch.device,
                    epoch:int,
                    clip_max_norm:float=0,
                    l2_lambda:float=0.01,
                    regularization_terms:dict=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # metric_logger.update(forget_class_error=0.0)
    # metric_logger.update(remain_class_error=0.0)

    prefetcher = data_prefetcher(data_loader_cl_forget, device, prefetch=True)
    samples, targets = prefetcher.next()
    # if dist.get_rank() == 0:
    #     import pdb; pdb.set_trace()
    for _ in metric_logger.log_every(range(len(data_loader_cl_forget)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # L2 regularization loss
        l2_loss = get_l2_loss(model, regularization_terms,l2_lambda=l2_lambda)
        
        losses += l2_loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item() + l2_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(l2_loss=l2_loss.item())
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        samples, targets = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_ewc(model:torch.nn.Module,
                    criterion:torch.nn.Module,
                    data_loader_cl_forget:Iterable,
                    optimizer:torch.optim.Optimizer,
                    device:torch.device,
                    epoch:int,
                    clip_max_norm:float=0,
                    ewc_lambda:float=0.01,
                    regularization_terms:dict=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    prefetcher = data_prefetcher(data_loader_cl_forget, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader_cl_forget)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # EWC regularization loss
        ewc_loss = get_l2_loss(model, regularization_terms, l2_lambda=ewc_lambda)

        losses += ewc_loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item() + ewc_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(ewc_loss=ewc_loss.item())
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        samples, targets = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_mas(model:torch.nn.Module,criterion:torch.nn.Module,
                        data_loader_cl_forget:Iterable,
                        optimizer:torch.optim.Optimizer,
                        device:torch.device,
                        epoch:int,
                        clip_max_norm:float=0,
                        mas_lambda:float=0.01,
                        regularization_terms:dict=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    prefetcher = data_prefetcher(data_loader_cl_forget, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader_cl_forget)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # MAS regularization loss
        mas_loss = get_l2_loss(model, regularization_terms, l2_lambda=mas_lambda)

        losses += mas_loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item() + mas_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(mas_loss=mas_loss.item())
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        samples, targets = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} 


def train_one_epoch_si():
    pass




