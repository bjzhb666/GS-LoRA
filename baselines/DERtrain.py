import torch
import torch.nn as nn
import torch.nn.functional as F
import util.utils as util

from util.utils import train_accuracy
from util.data_prefetcher import data_prefetcher
import wandb
from engine_cl import evaluate


def DER_regularzaion_loss(preds, gts):
    diff = preds - gts
    norm = torch.norm(diff, p=2)
    squared_l2_norm = norm * norm
    return squared_l2_norm


def train_one_epoch_DER(
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
    losses_DER: util.AverageMeter,
    losses_total: util.AverageMeter,
    task_i: str,
    testloader_forget: torch.utils.data.DataLoader,
    testloader_remain: torch.utils.data.DataLoader,
    forget_acc_before: float,
    highest_H_mean: float,
    cfg: dict,
    lambda_der: float,
    plus: bool = False,
    lambda_der_plus: float = 0.0,
    testloader_open: torch.utils.data.DataLoader = None,
):
    student_model.train()
    teacher_model.eval()

    prefetcher_remain = data_prefetcher(remain_loader, device=device, prefetch=True)
    inputs_remain, labels_remain = prefetcher_remain.next()

    if inputs_remain is None:
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

        with torch.no_grad():
            outputs_teacher_remain, embeds_teacher_remain = teacher_model(
                inputs_remain.float(), labels_remain
            )
            outputs_teacher_remain = outputs_teacher_remain.detach()

        loss_der = DER_regularzaion_loss(embeds_remain, embeds_teacher_remain)

        losses_CE_next = torch.tensor(0.0)

        if plus:
            inputs_remain_next, labels_remain_next = prefetcher_remain.next()
            if inputs_remain_next is None:
                prefetcher_remain = data_prefetcher(
                    remain_loader, device=device, prefetch=True
                )
                inputs_remain_next, labels_remain_next = prefetcher_remain.next()
            outputs_remain_next, embeds_remain_next = student_model(
                inputs_remain_next.float(), labels_remain_next
            )
            losses_CE_next = criterion(outputs_remain_next, labels_remain_next)

        loss_total = loss_CE + lambda_der * loss_der + lambda_der_plus * losses_CE_next

        losses_total.update(loss_total.item(), inputs_forget.size(0))
        losses_CE.update(loss_CE.item(), inputs_forget.size(0))
        losses_DER.update(loss_der.item(), inputs_forget.size(0))

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # display training loss & acc every DISP_FREQ iterations
        if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
            epoch_loss_CE = losses_CE.avg
            epoch_loss_DER = losses_DER.avg
            epoch_loss_total = losses_total.avg

            wandb.log(
                {
                    "epoch_loss_CE": epoch_loss_CE,
                    "epoch_loss_DER": epoch_loss_DER,
                    "epoch_loss_total": epoch_loss_total,
                }
            )

            print(
                "Task {} Epoch {} Batch {}\t"
                "Training CE Loss {loss_CE.val:.4f} ({loss_CE.avg:.4f})\t"
                "Training DER Loss {loss_DER.val:.4f} ({loss_DER.avg:.4f})\t"
                "Training Total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t".format(
                    task_i,
                    epoch + 1,
                    batch + 1,
                    loss_CE=losses_CE,
                    loss_DER=losses_DER,
                    loss_total=losses_total,
                )
            )

            losses_DER.reset()
            losses_CE.reset()
            losses_total.reset()

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

        # prefetcher next batch
        inputs_remain, labels_remain = prefetcher_remain.next()
        if inputs_remain is None:
            prefetcher_remain = data_prefetcher(
                remain_loader, device=device, prefetch=True
            )
            inputs_remain, labels_remain = prefetcher_remain.next()
    return batch, highest_H_mean, losses_CE, losses_DER, losses_total
