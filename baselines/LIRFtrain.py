import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import logging
import numpy as np

from util.utils import train_accuracy
import util.utils as util
from util.data_prefetcher import data_prefetcher
import wandb
from util.utils import get_time
from IPython import embed


class AT(nn.Module):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    """

    def __init__(self):
        super(AT, self).__init__()
        self.p = 2

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am


def at(x):
    att = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    att[att < 0.005] = 0
    # print (att)
    return att


def at_loss(x, y):
    attx = at(x)
    atty = at(y)
    # print((attx).size())
    # print(attx,atty)
    # atty[atty<0.2] = 0
    return (attx - atty).pow(2).mean()


def train_one_epoch_LIRF(
    student_low: torch.nn.Module,
    deposit_low: torch.nn.Module,
    teacher_low: torch.nn.Module,
    teacher_up: torch.nn.Module,
    data_loader_cl_forget: torch.utils.data.DataLoader,
    remain_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch: int,  # lambda_kp:float,lambda_at:float,
    losses_CE: util.AverageMeter,
    losses_AT: util.AverageMeter,
    kd_lossesKP: util.AverageMeter,
    losses_pt_re: util.AverageMeter,
    losses_total: util.AverageMeter,
    losses_remain: util.AverageMeter,
    task_i: str,
    testloader_forget: torch.utils.data.DataLoader,
    testloader_remain: torch.utils.data.DataLoader,
    forget_acc_before: float,
    highest_H_mean: float,
    cfg: dict,
    testloader_open: torch.utils.data.DataLoader = None,
):
    student_low.train()
    deposit_low.train()
    criterion.train()

    teacher_low.eval()
    teacher_up.eval()

    criterionAT = AT()
    DISP_FREQ = 5
    VER_FREQ = 5

    split = cfg["PER_FORGET_CLS"]
    T = cfg["LIRF_T"]

    # remain_loader prefetcher
    prefetcher_remain = data_prefetcher(
        remain_loader, device=device, prefetch=True
    )  # data has been put in GPU
    inputs_remain, labels_remain = prefetcher_remain.next()

    for inputs_forget, labels_forget in iter(data_loader_cl_forget):
        inputs_forget = inputs_forget.to(device)
        labels_forget = labels_forget.to(device)

        student_forget_mid = student_low(inputs_forget)
        student_final, _ = teacher_up(student_forget_mid, labels_forget)

        deposit_forget_mid = deposit_low(inputs_forget)
        deposit_final, _ = teacher_up(deposit_forget_mid, labels_forget)

        # compute teacher outputs
        with torch.no_grad():
            teacher_mid = teacher_low(inputs_forget)
            teacher_final, _ = teacher_up(teacher_mid, labels_forget)

        # compute loss
        # 1. CE loss
        loss_CE = criterion(student_final, labels_forget) * (1 - cfg["LIRF_alpha"])
        # 2. AT loss
        # loss_AT = criterionAT(student_forget_mid, teacher_mid.detach())
        loss_AT = at_loss(student_forget_mid, teacher_mid.detach())
        # 3. kd loss in loss KP
        kd_lossKP = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_final[:, split:] / T, dim=1),
            F.softmax(teacher_final[:, split:] / T, dim=1),
        ) * (cfg["LIRF_alpha"] * T * T)

        # 4. Loss pt and Loss re for recovery
        loss_pt_re = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(deposit_final[:, :split] / T, dim=1),
            F.softmax(teacher_final[:, :split] / T, dim=1),
        ) * (cfg["LIRF_alpha"] * T * T) + nn.CrossEntropyLoss()(
            deposit_final, labels_forget
        ) * (
            1 - cfg["LIRF_alpha"]
        )

        # 5. replay CE loss
        student_remain_mid = student_low(inputs_remain)
        student_remain_final, _ = teacher_up(student_remain_mid, labels_remain)
        loss_replay = criterion(student_remain_final, labels_remain)

        # total loss
        loss_total = (
            loss_CE
            - 300 * loss_AT
            + 10 * kd_lossKP
            + 0.05 * loss_pt_re
            + 5 * loss_replay
        )

        losses_total.update(loss_total.item(), inputs_forget.size(0))
        losses_CE.update(loss_CE.item(), inputs_forget.size(0))
        losses_AT.update(loss_AT.item(), inputs_forget.size(0))
        kd_lossesKP.update(kd_lossKP.item(), inputs_forget.size(0))
        losses_pt_re.update(loss_pt_re.item(), inputs_forget.size(0))
        losses_remain.update(loss_replay.item(), inputs_forget.size(0))

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # display training loss & accuarcy every DISP_FREQ iterations
        if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
            epoch_loss_CE = losses_CE.avg
            epoch_loss_AT = losses_AT.avg
            epoch_loss_kdKP = kd_lossesKP.avg
            epoch_loss_pt_re = losses_pt_re.avg
            epoch_loss_total = losses_total.avg
            epoch_loss_remain = losses_remain.avg

            wandb.log(
                {
                    "epoch_loss_CE-{}".format(task_i): epoch_loss_CE,
                    "epoch_loss_AT-{}".format(task_i): epoch_loss_AT,
                    "epoch_loss_kdKP-{}".format(task_i): epoch_loss_kdKP,
                    "epoch_loss_pt_re-{}".format(task_i): epoch_loss_pt_re,
                    "epoch_loss_total-{}".format(task_i): epoch_loss_total,
                    "epoch_loss_remain-{}".format(task_i): epoch_loss_remain,
                }
            )

            print(
                "Task {} Epoch {} Batch {}\t"
                "Training CE Loss {loss_CE.val:.4f} ({loss_CE.avg:.4f})\t"
                "Training AT Loss {loss_AT.val:.4f} ({loss_AT.avg:.4f})\t"
                "Training total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
                "Training remain Loss {loss_remain.val:.4f} ({loss_remain.avg:.4f})\t".format(
                    task_i,
                    epoch + 1,
                    batch + 1,
                    loss_CE=losses_CE,
                    loss_AT=losses_AT,
                    loss_total=losses_total,
                    loss_remain=losses_remain,
                )
            )

            # reset average meters
            losses_CE.reset()
            losses_AT.reset()
            kd_lossesKP.reset()
            losses_pt_re.reset()
            losses_total.reset()
            losses_remain.reset()
        
        with torch.no_grad():
            if ((batch + 1) % VER_FREQ == 0) and batch != 0:
                highest_H_mean = evaluate_LIRF(
                    student_low,
                    teacher_up,
                    testloader_forget,
                    testloader_remain,
                    device,
                    batch,
                    epoch,
                    task_i,
                    forget_acc_before,
                    highest_H_mean,
                    cfg,
                    optimizer,
                    testloader_open=testloader_open,
                )
                student_low.train()

        batch += 1
        # prefetcher next remain batch
        inputs_remain, labels_remain = prefetcher_remain.next()
        if inputs_remain is None:
            prefetcher_remain = data_prefetcher(
                remain_loader, device=device, prefetch=True
            )
            inputs_remain, labels_remain = prefetcher_remain.next()

    return (
        batch,
        highest_H_mean,
        losses_CE,
        losses_AT,
        kd_lossesKP,
        losses_pt_re,
        losses_total,
        losses_remain,
    )


def eval_data_LIRF(
    student_low: torch.nn.Module,
    teacher_up: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    mode: str,
    batch: int = 0,
):
    """

    Evaluate the model on test set for LIRF, return acc (0-100)
    """
    correct = 0
    total = 0
    student_low.eval()
    teacher_up.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs_mid = student_low(images)
            outputs, _ = teacher_up(outputs_mid, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("Test {} Accuracy in LIRF = {:.2f}%".format(mode, accuracy))
    wandb.log({"Test {} Accuracy".format(mode): accuracy})

    return accuracy


def evaluate_LIRF(
    student_low: torch.nn.Module,
    teacher_up: torch.nn.Module,
    testloader_forget: torch.utils.data.DataLoader,
    testloader_remain: torch.utils.data.DataLoader,
    device: torch.device,
    batch: int,
    epoch: int,
    task_i: str,
    forget_acc_before: float,
    highest_H_mean: float,
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    testloader_open: torch.utils.data.DataLoader = None,
):
    student_low.eval()
    teacher_up.eval()

    for params in optimizer.param_groups:
        lr = params["lr"]
        break

    print("current learning rate:{:.7f}".format(lr))
    print("Perfom evaluation on test set and save checkpoints...")

    forget_acc = eval_data_LIRF(
        student_low,
        teacher_up,
        testloader_forget,
        device,
        "forget-{}".format(task_i),
        batch,
    )
    remain_acc = eval_data_LIRF(
        student_low,
        teacher_up,
        testloader_remain,
        device,
        "remain-{}".format(task_i),
        batch,
    )
    if testloader_open is not None:
        open_acc = eval_data_LIRF(
            student_low,
            teacher_up,
            testloader_open,
            device,
            "open-{}".format(task_i),
            batch,
        )
    forget_drop = forget_acc_before - forget_acc
    Hmean = 2 * forget_drop * remain_acc / (forget_drop + remain_acc + 1e-8)

    if Hmean > highest_H_mean:
        highest_H_mean = Hmean

    return highest_H_mean
