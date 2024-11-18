import torch
from typing import Iterable
import util.misc as utils
import math
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.data_prefetcher import data_prefetcher
from util.sgda_utils import adjust_learning_rate as sgda_adjust_learning_rate
from util.sgda_utils import DistillKL, param_dist


def train_one_superepoch_SCRUB(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    swa_model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader_forget: Iterable,
    data_loader_remain: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    superepoch: int,
    max_norm: float = 0,
    kd_T: float = 2.0,
    sgda_smoothing: float = 0.0,
    sgda_gamma: float = 0.99,
    sgda_alpha: float = 0.001,
    sgda_lr: float = 0.1,
    lr_decay_epochs: int = 30,
    lr_decay_rate: float = 0.1,
):
    student_model.train()
    teacher_model.eval()
    criterion.train()

    criterionKD = DistillKL(T=kd_T)
    criterionKD.train()

    forget_prefetcher = data_prefetcher(data_loader_forget, device, prefetch=True)
    remain_prefetcher = data_prefetcher(data_loader_remain, device, prefetch=True)

    inputs_forget, targets_forget = forget_prefetcher.next()
    inputs_remain, targets_remain = remain_prefetcher.next()

    # metric_logger = utils.MetricLogger(delimiter=" ")
    # metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # header = "Epoch: [{}]".format(superepoch*15)
    print_freq = 20
    # import pdb; pdb.set_trace()
    # create option for sgda adjust learning rate
    opt = {}
    opt["sgda_learning_rate"] = sgda_lr
    opt["lr_decay_rate"] = lr_decay_rate
    opt["lr_decay_epochs"] = lr_decay_epochs

    print("len(data_loader_forget):", len(data_loader_forget))
    print("len(data_loader_remain):", len(data_loader_remain))

    for i in range(10):

        if i < 5:  # max steps + min steps
            print("\033[0;31;40mmax steps + min steps\033[0m")
            epoch = superepoch * 15 + i
            lr = sgda_adjust_learning_rate(epoch, opt, optimizer)

            for iter in range(len(data_loader_forget)):

                outputs_forget = student_model(inputs_forget)

                with torch.no_grad():
                    teacher_outputs_forget = teacher_model(inputs_forget)
                    teacher_outputs_forget["pred_logits"] = teacher_outputs_forget[
                        "pred_logits"
                    ].detach()

                # compute KD loss
                loss_kd = criterionKD(
                    outputs_forget["pred_logits"], teacher_outputs_forget["pred_logits"]
                )
                # compute SGDA loss
                loss_sgda = param_dist(student_model, swa_model, sgda_smoothing)
                # total loss
                loss = -loss_kd + loss_sgda

                if iter % print_freq == 0:
                    print(
                        f"forget epoch:{epoch}, iter: {iter}, loss_kd: {loss_kd.item()}, loss_sgda: {loss_sgda.item()}, loss: {loss.item()}"
                    )

                # ===================backward=====================
                optimizer.zero_grad()
                loss.backward()
                if max_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), max_norm
                    )
                else:
                    grad_total_norm = utils.get_total_grad_norm(
                        student_model.parameters(), max_norm
                    )
                optimizer.step()

                inputs_forget, targets_forget = forget_prefetcher.next()
                if inputs_forget is None:
                    prefetcher_forget = data_prefetcher(
                        data_loader_forget, device, prefetch=True
                    )
                    inputs_forget, targets_forget = prefetcher_forget.next()

            for iter in range(len(data_loader_remain)):
                outputs_remain = student_model(inputs_remain)

                with torch.no_grad():
                    teacher_outputs_remain = teacher_model(inputs_remain)
                    teacher_outputs_remain["pred_logits"] = teacher_outputs_remain[
                        "pred_logits"
                    ].detach()

                # compute KD loss
                loss_kd = criterionKD(
                    outputs_remain["pred_logits"], teacher_outputs_remain["pred_logits"]
                )
                # compute SGDA loss
                loss_sgda = param_dist(student_model, swa_model, sgda_smoothing)
                # compute original loss
                loss_dict_remain = criterion(outputs_remain, targets_remain)
                weight_dict = criterion.weight_dict
                losses_original = sum(
                    loss_dict_remain[k] * weight_dict[k]
                    for k in loss_dict_remain.keys()
                    if k in weight_dict
                )
                # total loss
                loss_remain_total = (
                    sgda_gamma * losses_original + sgda_alpha * loss_kd + loss_sgda
                )

                # metric_logger.update(loss_remain_total=loss_remain_total.item())
                # metric_logger.update(loss_kd_remain=loss_kd.item()*sgda_alpha)
                # metric_logger.update(loss_sgda_remain=loss_sgda.item())
                if iter % print_freq == 0:
                    print(
                        f"remain epoch:{epoch}, iter: {iter}, loss_original:{losses_original.item()*sgda_gamma}, loss_kd: {loss_kd.item()*sgda_alpha}, loss_sgda: {loss_sgda.item()}, loss: {loss_remain_total.item()}"
                    )
                # backward
                optimizer.zero_grad()
                loss_remain_total.backward()
                if max_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), max_norm
                    )
                else:
                    grad_total_norm = utils.get_total_grad_norm(
                        student_model.parameters(), max_norm
                    )
                optimizer.step()

                inputs_remain, targets_remain = remain_prefetcher.next()
                if inputs_remain is None:
                    prefetcher_remain = data_prefetcher(
                        data_loader_remain, device, prefetch=True
                    )
                    inputs_remain, targets_remain = prefetcher_remain.next()

        else:
            # min steps
            print("\033[0;31;40mmin steps\033[0m")
            epoch = superepoch * 15 + i
            lr = sgda_adjust_learning_rate(epoch, opt, optimizer)

            for iter in range(len(data_loader_remain)):
                outputs_remain = student_model(inputs_remain)

                with torch.no_grad():
                    teacher_outputs_remain = teacher_model(inputs_remain)
                    teacher_outputs_remain["pred_logits"] = teacher_outputs_remain[
                        "pred_logits"
                    ].detach()

                # compute KD loss
                loss_kd = criterionKD(
                    outputs_remain["pred_logits"], teacher_outputs_remain["pred_logits"]
                )
                # compute SGDA loss
                loss_sgda = param_dist(student_model, swa_model, sgda_smoothing)
                # compute original loss
                loss_dict_remain = criterion(outputs_remain, targets_remain)
                weight_dict = criterion.weight_dict
                losses_original = sum(
                    loss_dict_remain[k] * weight_dict[k]
                    for k in loss_dict_remain.keys()
                    if k in weight_dict
                )
                # total loss
                loss_remain_total = (
                    sgda_gamma * losses_original + sgda_alpha * loss_kd + loss_sgda
                )

                # metric_logger.update(loss_remain_total=loss_remain_total.item())
                # metric_logger.update(loss_kd_remain=loss_kd.item()*sgda_alpha)
                # metric_logger.update(loss_sgda_remain=loss_sgda.item())
                if iter % print_freq == 0:
                    print(
                        f"remain epoch:{epoch}, iter: {iter}, loss_original:{losses_original.item()*sgda_gamma}, loss_kd: {loss_kd.item()*sgda_alpha}, loss_sgda: {loss_sgda.item()}, loss: {loss_remain_total.item()}"
                    )
                # backward
                optimizer.zero_grad()
                loss_remain_total.backward()
                if max_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), max_norm
                    )
                else:
                    grad_total_norm = utils.get_total_grad_norm(
                        student_model.parameters(), max_norm
                    )
                optimizer.step()

                inputs_remain, targets_remain = remain_prefetcher.next()
                if inputs_remain is None:
                    prefetcher_remain = data_prefetcher(
                        data_loader_remain, device, prefetch=True
                    )
                    inputs_remain, targets_remain = prefetcher_remain.next()

    # gather the stats from the remaining part
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    return None
