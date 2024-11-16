import torch
from typing import Iterable
import util.misc as utils
import math
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.data_prefetcher import data_prefetcher


def DER_regularzaion_loss(preds, gts):
    diff = preds - gts
    norm = torch.norm(diff, p=2)
    squared_l2_norm = norm * norm
    return squared_l2_norm


def train_one_epoch_DER(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader_forget: Iterable,
    data_loader_remain: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    lambda_der: float = 0.0,
    plus: bool = False,
    lambda_der_plus: float = 0.0,
):
    student_model.train()
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

        loss_der = DER_regularzaion_loss(
            outputs_remain["pred_logits"], teacher_outputs_remain["pred_logits"]
        )

        losses_CE_next = torch.tensor(0.0)

        if plus:
            inputs_remain_next, targets_remain_next = prefetcher_remain.next()
            if inputs_remain_next is None:
                prefetcher_remain = data_prefetcher(
                    data_loader_remain, device, prefetch=True
                )
                inputs_remain_next, targets_remain_next = prefetcher_remain.next()
            outputs_remain_next = student_model(inputs_remain_next)
            losses_CE_next = criterion(outputs_remain_next, targets_remain_next)
            weight_dict_CE_next = criterion.weight_dict
            losses_CE_next = sum(
                losses_CE_next[k] * weight_dict_CE_next[k]
                for k in losses_CE_next.keys()
                if k in weight_dict_CE_next
            )

        loss_total = losses + lambda_der * loss_der + lambda_der_plus * losses_CE_next

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
        loss_value = (
            losses_reduced_scaled.item()
            + lambda_der * loss_der.item()
            + lambda_der_plus * losses_CE_next.item()
        )
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced_forget)
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
        metric_logger.update(loss_der=loss_der.item() * lambda_der)
        if plus:
            metric_logger.update(loss_CE_next=losses_CE_next.item())
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        inputs_forget, targets_forget = prefetcher_forget.next()
        inputs_remain, targets_remain = prefetcher_remain.next()

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
