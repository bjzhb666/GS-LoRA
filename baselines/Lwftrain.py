import torch
import torch.nn as nn
import torch.nn.functional as F
import util.utils as util

from util.utils import train_accuracy
from util.data_prefetcher import data_prefetcher
import wandb
from engine_cl import evaluate


def L_old_kd_loss(preds, gts, temperature=2.0):
    preds = F.softmax(preds, dim=-1)
    preds = torch.pow(preds, 1.0 / temperature)
    # l_preds = F.softmax(preds, dim=-1)
    l_preds = F.log_softmax(preds, dim=-1)

    gts = F.softmax(gts, dim=-1)
    gts = torch.pow(gts, 1.0 / temperature)
    l_gts = F.log_softmax(gts, -1)

    l_preds = torch.log(l_preds)

    l_preds[l_preds != l_preds] = 0.0  # remove nan
    loss = torch.mean(torch.sum(-l_gts * l_preds, axis=1))

    return loss


def train_one_epoch_Lwf(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    data_loader_cl_forget: torch.utils.data.DataLoader,
    remain_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch: int,
    losses_CE: util.AverageMeter,
    losses_KD: util.AverageMeter,
    losses_total: util.AverageMeter,
    losses_remain: util.AverageMeter,
    task_i: str,
    testloader_forget: torch.utils.data.DataLoader,
    testloader_remain: torch.utils.data.DataLoader,
    forget_acc_before: float,
    highest_H_mean: float,
    cfg: dict,
    lambda_kd: float,
    lambda_remain: float,
    temperature: float,
    testloader_open: torch.utils.data.DataLoader = None,
):
    student_model.train()
    teacher_model.eval()

    prefetcher_remain = data_prefetcher(remain_loader, device=device, prefetch=True)
    inputs_remain, labels_remain = prefetcher_remain.next()

    DISP_FREQ = 5
    VER_FREQ = 100

    for inputs_forget, labels_forget in iter(data_loader_cl_forget):
        inputs_forget, labels_forget = inputs_forget.to(device), labels_forget.to(
            device
        )

        outputs_forget, embeds_forget = student_model(
            inputs_forget.float(), labels_forget
        )
        loss_CE = criterion(outputs_forget, labels_forget)

        outputs_remain, embeds_remain = student_model(
            inputs_remain.float(), labels_remain
        )
        loss_remain = criterion(outputs_remain, labels_remain)

        with torch.no_grad():
            outputs_teacher, embeds_teacher = teacher_model(
                inputs_remain.float(), labels_remain
            )
            outputs_teacher = outputs_teacher.detach()

        loss_KD = L_old_kd_loss(
            outputs_remain, outputs_teacher, temperature=temperature
        )

        loss_total = loss_CE + lambda_kd * loss_KD + lambda_remain * loss_remain

        losses_total.update(loss_total.item(), inputs_forget.size(0))
        losses_CE.update(loss_CE.item(), inputs_forget.size(0))
        losses_KD.update(loss_KD.item(), inputs_forget.size(0))
        losses_remain.update(loss_remain.item(), inputs_forget.size(0))

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # display training loss & acc every DISP_FREQ iterations
        if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
            epoch_loss_CE = losses_CE.avg
            epoch_loss_KD = losses_KD.avg
            epoch_loss_remain = losses_remain.avg
            epoch_loss_total = losses_total.avg

            wandb.log(
                {
                    "epoch_loss_CE-{}".format(task_i): epoch_loss_CE,
                    "epoch_loss_KD-{}".format(task_i): epoch_loss_KD,
                    "epoch_loss_remain-{}".format(task_i): epoch_loss_remain,
                    "epoch_loss_total-{}".format(task_i): epoch_loss_total,
                }
            )

            print(
                "Task {} Epoch {} Batch {}\t"
                "Training CE Loss {loss_CE.val:.4f} ({loss_CE.avg:.4f})\t"
                "Training KD Loss {loss_KD.val:.4f} ({loss_KD.avg:.4f})\t"
                "Training Remain Loss {loss_remain.val:.4f} ({loss_remain.avg:.4f})\t"
                "Training Total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t".format(
                    task_i,
                    epoch + 1,
                    batch + 1,
                    loss_CE=losses_CE,
                    loss_KD=losses_KD,
                    loss_remain=losses_remain,
                    loss_total=losses_total,
                )
            )

            # reset average meters
            losses_CE.reset()
            losses_KD.reset()
            losses_total.reset()
            losses_remain.reset()
        with torch.no_grad():
            if ((batch + 1) % VER_FREQ == 0) and batch != 0:
                highest_H_mean = evaluate(
                    student_model,
                    testloader_forget=testloader_forget,
                    testloader_remain=testloader_remain,
                    device=device,
                    batch=batch,
                    epoch=epoch,
                    task_i=task_i,
                    forget_acc_before=forget_acc_before,
                    highest_H_mean=highest_H_mean,
                    cfg=cfg,
                    optimizer=optimizer,
                    testloader_open=testloader_open,
                )
                student_model.train()

        batch += 1

        # prefetch next batch
        inputs_remain, labels_remain = prefetcher_remain.next()
        if inputs_remain is None:
            prefetcher_remain = data_prefetcher(
                remain_loader, device=device, prefetch=True
            )
            inputs_remain, labels_remain = prefetcher_remain.next()

    return batch, highest_H_mean, losses_CE, losses_KD, losses_remain, losses_total
