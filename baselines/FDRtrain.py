import torch
from typing import Iterable
import util.misc as utils
import math
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.data_prefetcher import data_prefetcher

"""
Implementation of Hilbert-constrained gradient descent.
From MEASURING AND REGULARIZING NETWORKS IN FUNCTION SPACE (ICLR 2019)
"""


def regularization_loss(outputs, target_logits):
    loss = torch.norm(outputs - target_logits, 2, 1).mean()
    return loss


def train_one_epoch_FDR(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader_forget: Iterable,
    data_loader_remain: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    reg_lambda: float = 0.0,
):
    student_model.train()
    criterion.train()
    teacher_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    prefetcher_remain = data_prefetcher(data_loader_remain, device, prefetch=True)
    inputs_remain, targets_remain = prefetcher_remain.next()
    prefetcher_forget = data_prefetcher(data_loader_forget, device, prefetch=True)
    inputs_forget, targets_forget = prefetcher_forget.next()

    for _ in metric_logger.log_every(
        range(len(data_loader_forget)), print_freq, header
    ):
        outputs_forget = student_model(inputs_forget)
        loss_dict_forget = criterion(outputs_forget, targets_forget)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict_forget[k] * weight_dict[k]
            for k in loss_dict_forget.keys()
            if k in weight_dict
        )
        outputs_remain = student_model(inputs_remain)

        with torch.no_grad():
            teacher_outputs_remain = teacher_model(inputs_remain)
            teacher_outputs_remain["pred_logits"] = teacher_outputs_remain[
                "pred_logits"
            ].detach()

        loss_FDR = regularization_loss(
            outputs_remain["pred_logits"], teacher_outputs_remain["pred_logits"]
        )

        loss_total = losses + reg_lambda * loss_FDR

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_forget = utils.reduce_dict(loss_dict_forget)
        loss_dict_reduced_forget_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced_forget.items()
        }
        loss_dict_reduced_forget_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced_forget.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_forget_scaled.values())
        loss_value = losses_reduced_scaled.item() + reg_lambda * loss_FDR.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss_total.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(), max_norm
            )
        else:
            grad_total_norm = utils.get_total_grad_norm(
                student_model.parameters(), max_norm
            )

        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_forget_scaled)
        metric_logger.update(loss_FDR=loss_FDR.item() * reg_lambda)
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        inputs_remain, targets_remain = prefetcher_remain.next()
        inputs_forget, targets_forget = prefetcher_forget.next()

        if inputs_forget is None:
            prefetcher_forget = data_prefetcher(
                data_loader_forget, device, prefetch=True
            )
            inputs_forget, targets_forget = prefetcher_forget.next()
        if inputs_remain is None:
            prefetcher_remain = data_prefetcher(
                data_loader_remain, device, prefetch=True
            )
            inputs_remain, targets_remain = prefetcher_remain.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
