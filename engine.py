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

def train_one_epoch_incremental(model: torch.nn.Module, old_model: torch.nn.Module, ref_loss_overall_coef,
                    criterion: torch.nn.Module, postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    old_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        ref_outputs = old_model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        ref_results = postprocessors['bbox'](ref_outputs, orig_target_sizes, topk = 5, distillation=True)
        ref_loss_dict = criterion(outputs, ref_results, enable_aux=False)
        ref_weight_dict = criterion.ref_weight_dict
        ref_losses = sum(ref_loss_dict[k] * ref_weight_dict[k] for k in ref_loss_dict.keys() if k in ref_weight_dict)

        # 蒸馏策略的改进，先用旧模型产生伪标签，然后将伪标签加入到新标签中，再用新模型训练（样本层面的蒸馏，而不再是logits的蒸馏）
        for img_idx in range(len(targets)):
            include_list = []
            for ref_box_idx in range(len(ref_results[img_idx]['boxes'])):
                this_ref_box = ref_results[img_idx]['boxes'][ref_box_idx]
                this_ref_box = torch.reshape(this_ref_box, (1, -1))
                include_this_pseudo_label = True
                for target_box_idx in range(len(targets[img_idx]['boxes'])):
                    this_target_box = targets[img_idx]['boxes'][target_box_idx]
                    this_target_box = torch.reshape(this_target_box, (1, -1))
                    iou, union = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(this_ref_box), box_ops.box_cxcywh_to_xyxy(this_target_box))
                    if iou >= 0.7: # 伪标签与新标签的iou大于0.7，不加入
                        include_this_pseudo_label = False
                include_list.append(include_this_pseudo_label)

            targets[img_idx]['boxes'] = torch.cat((targets[img_idx]['boxes'], ref_results[img_idx]['boxes'][include_list]), 0)
            targets[img_idx]['labels'] = torch.cat((targets[img_idx]['labels'], ref_results[img_idx]['labels'][include_list]), 0)
            # 该代码将具有include_list中为True的目标框和标签连接起来，从而创建一个包含伪标签的新targets列表。
        loss_dict = criterion(outputs, targets) # 直接将新旧标签混合后的targets传入，计算损失，相当于Ldetr+Lkd
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes, below are the same as train_one_epoch
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        #losses += ref_loss_overall_coef*ref_losses

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, ascent: bool = False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if ascent: # gradient ascent
            losses = -losses
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_forget_remain(model:torch.nn.Module, criterion:torch.nn.Module,
                                  data_loader_forget:Iterable, data_loader_remain:Iterable,
                                  optimizer:torch.optim.Optimizer, device:torch.device,
                                  epoch: int, max_norm: float = 0, beta: float = 0.5):
    '''
    :param beta: the weight of the forget data loss $L_{total} = \beta L_{forget} + L_{remain}$
    '''
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('forget_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('remain_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    prefetcher_forget = data_prefetcher(data_loader_forget, device, prefetch=True)
    samples_forget, targets_forget = prefetcher_forget.next()
    prefetcher_remain = data_prefetcher(data_loader_remain, device, prefetch=True)
    samples_remain, targets_remain = prefetcher_remain.next()
    for _ in metric_logger.log_every(range(len(data_loader_remain)), print_freq, header):
        outputs_remain = model(samples_remain)
        loss_dict_remain = criterion(outputs_remain, targets_remain)
        weight_dict_remain = criterion.weight_dict
        losses_remain = sum(loss_dict_remain[k] * weight_dict_remain[k] for k in loss_dict_remain.keys() if k in weight_dict_remain)

        outputs_forget = model(samples_forget)
        loss_dict_forget = criterion(outputs_forget, targets_forget)
        weight_dict_forget = criterion.weight_dict
        losses_forget = sum(loss_dict_forget[k] * weight_dict_forget[k] for k in loss_dict_forget.keys() if k in weight_dict_forget)
        losses_forget = -losses_forget # gradient ascent
        # structure_loss = group_sparse_loss(model)
        losses_total = beta * losses_forget + losses_remain

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_forget = utils.reduce_dict(loss_dict_forget)
        loss_dict_reduced_unscaled_forget = {f'{k}_forget_unscaled': v
                                        for k, v in loss_dict_reduced_forget.items()}
        loss_dict_reduced_scaled_forget = {f'{k}_forget_scaled': v * weight_dict_forget[k]
                                        for k, v in loss_dict_reduced_forget.items() if k in weight_dict_forget}
        losses_reduced_scaled_forget = sum(loss_dict_reduced_scaled_forget.values())

        loss_dict_reduced_remain = utils.reduce_dict(loss_dict_remain)
        loss_dict_reduced_unscaled_remain = {f'{k}_remain_unscaled': v
                                        for k, v in loss_dict_reduced_remain.items()}
        loss_dict_reduced_scaled_remain = {f'{k}_remain_scaled': v * weight_dict_remain[k]
                                        for k, v in loss_dict_reduced_remain.items() if k in weight_dict_remain}
        losses_reduced_scaled_remain = sum(loss_dict_reduced_scaled_remain.values())



        loss_value = losses_reduced_scaled_forget.item() * -1.0 * beta + losses_reduced_scaled_remain.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced_forget)
            sys.exit(1)

        optimizer.zero_grad()
        losses_total.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             loss_forget=losses_reduced_scaled_forget.item(),
                             loss_remain = losses_reduced_scaled_remain.item(),
                             **loss_dict_reduced_scaled_forget,
                             **loss_dict_reduced_unscaled_forget,
                             **loss_dict_reduced_scaled_remain,
                             **loss_dict_reduced_unscaled_remain)
        metric_logger.update(forget_class_error=loss_dict_reduced_forget['class_error'])
        metric_logger.update(remain_class_error=loss_dict_reduced_remain['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        if dist.get_rank() == 0:
            wandb.log({"loss": loss_value, "loss_forget": losses_reduced_scaled_forget.item(),
                       "loss_remain": losses_reduced_scaled_remain.item(),
                       "forget_class_error": loss_dict_reduced_forget['class_error'],
                       "remain_class_error": loss_dict_reduced_remain['class_error'],})

        samples_forget, targets_forget = prefetcher_forget.next()
        samples_remain, targets_remain = prefetcher_remain.next()

        if samples_remain is None:
            prefetcher_remain = data_prefetcher(data_loader_remain, device, prefetch=True)
            samples_remain, targets_remain = prefetcher_remain.next()
        elif samples_forget is None:
            prefetcher_forget = data_prefetcher(data_loader_forget, device, prefetch=True)
            samples_forget, targets_forget = prefetcher_forget.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
'''
    for _ in metric_logger.log_every(range(len(data_loader_forget)), print_freq, header):
        outputs_forget = model(samples_forget)
        loss_dict_forget = criterion(outputs_forget, targets_forget)
        weight_dict_forget = criterion.weight_dict
        losses_forget = sum(loss_dict_forget[k] * weight_dict_forget[k] for k in loss_dict_forget.keys() if k in weight_dict_forget)   
        losses_forget = -losses_forget # gradient ascent

        outputs_remain = model(samples_remain)
        loss_dict_remain = criterion(outputs_remain, targets_remain)
        weight_dict_remain = criterion.weight_dict
        losses_remain = sum(loss_dict_remain[k] * weight_dict_remain[k] for k in loss_dict_remain.keys() if k in weight_dict_remain)

        losses_total = beta * losses_forget + losses_remain
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_forget = utils.reduce_dict(loss_dict_forget)
        loss_dict_reduced_unscaled_forget = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced_forget.items()}
        loss_dict_reduced_scaled_forget = {k: v * weight_dict_forget[k]
                                        for k, v in loss_dict_reduced_forget.items() if k in weight_dict_forget}
        losses_reduced_scaled_forget = sum(loss_dict_reduced_scaled_forget.values())

        loss_dict_reduced_remain = utils.reduce_dict(loss_dict_remain)
        loss_dict_reduced_unscaled_remain = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced_remain.items()}
        loss_dict_reduced_scaled_remain = {k: v * weight_dict_remain[k]
                                        for k, v in loss_dict_reduced_remain.items() if k in weight_dict_remain}
        losses_reduced_scaled_remain = sum(loss_dict_reduced_scaled_remain.values())

        loss_value = losses_reduced_scaled_forget.item() * beta + losses_reduced_scaled_remain.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced_forget)
            print(loss_dict_reduced_remain)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses_total.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled_forget, **loss_dict_reduced_scaled_remain)
        metric_logger.update(forget_class_error=loss_dict_reduced_forget['class_error'])
        metric_logger.update(remain_class_error=loss_dict_reduced_remain['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples_forget, targets_forget = prefetcher_forget.next()
        samples_remain, targets_remain = prefetcher_remain.next()
        
        if samples_remain is None:
            prefetcher_remain = data_prefetcher(data_loader_remain, device, prefetch=True)
            samples_remain, targets_remain = prefetcher_remain.next()
        elif samples_forget is None:
            prefetcher_forget = data_prefetcher(data_loader_forget, device, prefetch=True)
            samples_forget, targets_forget = prefetcher_forget.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
'''

def train_one_epoch_forget_cls(model:torch.nn.Module, criterion:torch.nn.Module,
                               data_loader_forget:Iterable, data_loader_remain:Iterable,
                               optimizer:torch.optim.Optimizer, device:torch.device,
                               epoch: int, max_norm: float = 0, beta: float = 0.5, alpha: float = 0.5):
    '''
    only use negative cls loss to train the model when forgetting, do not negative the bbox loss
    '''
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('forget_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('remain_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    prefetcher_forget = data_prefetcher(data_loader_forget, device, prefetch=True)
    samples_forget, targets_forget = prefetcher_forget.next()

    prefetcher_remain = data_prefetcher(data_loader_remain, device, prefetch=True)
    samples_remain, targets_remain = prefetcher_remain.next()
    for _ in metric_logger.log_every(range(len(data_loader_remain)), print_freq, header):
        outputs_remain = model(samples_remain)
        loss_dict_remain = criterion(outputs_remain, targets_remain)
        weight_dict_remain = criterion.weight_dict
        losses_remain = sum(loss_dict_remain[k] * weight_dict_remain[k] for k in loss_dict_remain.keys() if k in weight_dict_remain)

        outputs_forget = model(samples_forget)
        loss_dict_forget = criterion(outputs_forget, targets_forget)
        weight_dict_forget = criterion.forget_weight_dict
        losses_forget = sum(loss_dict_forget[k] * weight_dict_forget[k] for k in loss_dict_forget.keys() if k in weight_dict_forget)
        structure_loss = group_sparse_loss(model)
        losses_total = beta * losses_forget + losses_remain + alpha * structure_loss

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_forget = utils.reduce_dict(loss_dict_forget)
        loss_dict_reduced_unscaled_forget = {f'{k}_forget_unscaled': v
                                        for k, v in loss_dict_reduced_forget.items()}
        loss_dict_reduced_scaled_forget = {f'{k}_forget_scaled': v * weight_dict_forget[k]
                                        for k, v in loss_dict_reduced_forget.items() if k in weight_dict_forget}
        losses_reduced_scaled_forget = sum(loss_dict_reduced_scaled_forget.values())

        loss_dict_reduced_remain = utils.reduce_dict(loss_dict_remain)
        loss_dict_reduced_unscaled_remain = {f'{k}_remain_unscaled': v
                                        for k, v in loss_dict_reduced_remain.items()}
        loss_dict_reduced_scaled_remain = {f'{k}_remain_scaled': v * weight_dict_remain[k]
                                        for k, v in loss_dict_reduced_remain.items() if k in weight_dict_remain}
        losses_reduced_scaled_remain = sum(loss_dict_reduced_scaled_remain.values())

        
        # print('structure_loss: ', structure_loss)
        # loss_dict_reduced_structure = utils.reduce_dict(structure_loss)

        loss_value = losses_reduced_scaled_forget.item() * beta + losses_reduced_scaled_remain.item() + alpha * structure_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced_forget)
            print(loss_dict_reduced_remain)
            print(structure_loss.item())
            sys.exit(1)
            
        optimizer.zero_grad()
        losses_total.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                                loss_forget=losses_reduced_scaled_forget.item(),
                                loss_remain = losses_reduced_scaled_remain.item(),
                                loss_structure = structure_loss.item(),
                                **loss_dict_reduced_scaled_forget,
                                **loss_dict_reduced_unscaled_forget,
                                **loss_dict_reduced_scaled_remain,
                                **loss_dict_reduced_unscaled_remain)
        metric_logger.update(forget_class_error=loss_dict_reduced_forget['class_error'])
        metric_logger.update(remain_class_error=loss_dict_reduced_remain['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        if dist.get_rank() == 0:
            wandb.log({"loss": loss_value, "loss_forget": losses_reduced_scaled_forget.item(),
                       "loss_remain": losses_reduced_scaled_remain.item(),
                       "forget_class_error": loss_dict_reduced_forget['class_error'],
                       "remain_class_error": loss_dict_reduced_remain['class_error'],
                       "loss_structure": structure_loss.item()})

        samples_forget, targets_forget = prefetcher_forget.next()
        samples_remain, targets_remain = prefetcher_remain.next()
        
        if samples_remain is None:
            prefetcher_remain = data_prefetcher(data_loader_remain, device, prefetch=True)
            samples_remain, targets_remain = prefetcher_remain.next()
        elif samples_forget is None:
            prefetcher_forget = data_prefetcher(data_loader_forget, device, prefetch=True)
            samples_forget, targets_forget = prefetcher_forget.next()
        # if dist.get_rank() == 0:
        #     print("samples_forget: ", samples_forget)
        #     print("targets_forget: ", targets_forget)
        #     import pdb; pdb.set_trace()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

   
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 20, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        eval_result_array = coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # TODO: add eval_result_array to return value and modify the caller
    return stats, coco_evaluator, eval_result_array


def group_sparse_loss(model: torch.nn.Module, type='layer'):
    model_without_ddp = model.module
    learnable_params_name = [
        n for n, p in model_without_ddp.named_parameters() if p.requires_grad
    ]

    # grouping the parameters
    '''
    'transformer.encoder.layers.0.linear1.lora_A', 'transformer.encoder.layers.0.linear1.lora_B', 
    'transformer.encoder.layers.0.linear2.lora_A', 'transformer.encoder.layers.0.linear2.lora_B', 
    'transformer.encoder.layers.1.linear1.lora_A', 'transformer.encoder.layers.1.linear1.lora_B', 
    'transformer.encoder.layers.1.linear2.lora_A', 'transformer.encoder.layers.1.linear2.lora_B', 
    'transformer.encoder.layers.2.linear1.lora_A', 'transformer.encoder.layers.2.linear1.lora_B', 
    'transformer.encoder.layers.2.linear2.lora_A', 'transformer.encoder.layers.2.linear2.lora_B', 
    'transformer.encoder.layers.3.linear1.lora_A', 'transformer.encoder.layers.3.linear1.lora_B', 
    'transformer.encoder.layers.3.linear2.lora_A', 'transformer.encoder.layers.3.linear2.lora_B', 
    'transformer.encoder.layers.4.linear1.lora_A', 'transformer.encoder.layers.4.linear1.lora_B', 
    'transformer.encoder.layers.4.linear2.lora_A', 'transformer.encoder.layers.4.linear2.lora_B', 
    'transformer.encoder.layers.5.linear1.lora_A', 'transformer.encoder.layers.5.linear1.lora_B', 
    'transformer.encoder.layers.5.linear2.lora_A', 'transformer.encoder.layers.5.linear2.lora_B', 
    'transformer.decoder.layers.0.linear1.lora_A', 'transformer.decoder.layers.0.linear1.lora_B', 
    'transformer.decoder.layers.0.linear2.lora_A', 'transformer.decoder.layers.0.linear2.lora_B', 
    'transformer.decoder.layers.1.linear1.lora_A', 'transformer.decoder.layers.1.linear1.lora_B', 
    'transformer.decoder.layers.1.linear2.lora_A', 'transformer.decoder.layers.1.linear2.lora_B', 
    'transformer.decoder.layers.2.linear1.lora_A', 'transformer.decoder.layers.2.linear1.lora_B', 
    'transformer.decoder.layers.2.linear2.lora_A', 'transformer.decoder.layers.2.linear2.lora_B', 
    'transformer.decoder.layers.3.linear1.lora_A', 'transformer.decoder.layers.3.linear1.lora_B', 
    'transformer.decoder.layers.3.linear2.lora_A', 'transformer.decoder.layers.3.linear2.lora_B', 
    'transformer.decoder.layers.4.linear1.lora_A', 'transformer.decoder.layers.4.linear1.lora_B', 
    'transformer.decoder.layers.4.linear2.lora_A', 'transformer.decoder.layers.4.linear2.lora_B', 
    'transformer.decoder.layers.5.linear1.lora_A', 'transformer.decoder.layers.5.linear1.lora_B', 
    'transformer.decoder.layers.5.linear2.lora_A', 'transformer.decoder.layers.5.linear2.lora_B' '''

    # group the parameters into 12 groups
    # TODO: other group method can be used
    group_layers = []
    for i in range(12):
        group_item = []
        if i < 6:
            group_item.append('transformer.encoder.layers.' + str(i) +
                              '.linear1.lora_A')
            group_item.append('transformer.encoder.layers.' + str(i) +
                              '.linear1.lora_B')
            group_item.append('transformer.encoder.layers.' + str(i) +
                              '.linear2.lora_A')
            group_item.append('transformer.encoder.layers.' + str(i) +
                              '.linear2.lora_B')
        else:
            group_item.append('transformer.decoder.layers.' + str(i - 6) +
                              '.linear1.lora_A')
            group_item.append('transformer.decoder.layers.' + str(i - 6) +
                              '.linear1.lora_B')
            group_item.append('transformer.decoder.layers.' + str(i - 6) +
                              '.linear2.lora_A')
            group_item.append('transformer.decoder.layers.' + str(i - 6) +
                              '.linear2.lora_B')
        group_layers.append(group_item)
    # print(group_layers)
    # get the parameters
    group_params = []
    for group_item in group_layers:
        group_param = []
        for item in group_item:
            group_param.append(
                model_without_ddp.get_parameter(item) if item in
                learnable_params_name else None)
        group_params.append(group_param)
    # group_params is a list of list of parameters
    # print('group_params', len(group_params))
    # print(len(group_params[0]))
    # print(group_params[0][0].shape, group_params[0][1].shape, group_params[0][2].shape, group_params[0][3].shape)
    # calculate the loss
    def group_sparse_multi_module(group_param):
        # group_param is a list of parameters
        # calculate the loss for a single group of parameters
        def l2_loss(param_group):
            return torch.sum(param_group**2)

        lasso_sum = 0
        for param in group_param:
            lasso_sum += l2_loss(param)
        return torch.sqrt(lasso_sum)

    group_sparse_loss = 0
    # calculate the loss for all groups of parameters
    for group_param in group_params:
        group_sparse_loss += group_sparse_multi_module(group_param)
    # print('group_sparse_loss', group_sparse_loss)
    return group_sparse_loss