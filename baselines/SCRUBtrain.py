import torch
import torch.nn as nn
import torch.nn.functional as F
import util.utils as util
from util.sgda_utils import adjust_learning_rate as sgda_adjust_learning_rate
from util.sgda_utils import DistillKL, param_dist
import wandb
from engine_cl import evaluate


def train_one_superepoch_SCRUB(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    swa_model: torch.nn.Module,
    data_loader_forget: torch.utils.data.DataLoader,
    remain_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    superepoch: int,
    batch: int,
    losses_CE: util.AverageMeter,
    losses_total_forget: util.AverageMeter,
    losses_total_remain: util.AverageMeter,
    losses_kd_remain: util.AverageMeter,
    losses_kd_forget: util.AverageMeter,
    task_i: str,
    testloader_forget: torch.utils.data.DataLoader,
    testloader_remain: torch.utils.data.DataLoader,
    forget_acc_before: float,
    highest_H_mean: float,
    cfg: dict,
    kd_T: float = 2.0,
    sgda_smoothing: float = 0.0,
    sgda_gamma: float = 0.99,
    sgda_alpha: float = 0.001,
    testloader_open: torch.utils.data.DataLoader = None,
):
    """
    Note that we should use dataloader forget rather than data_loader_cl_forget
    kd_T = 2 temperature for kd loss
    """

    student.train()
    teacher.eval()
    criterion.train()

    criterionKD = DistillKL(T=kd_T)
    criterionKD.train()

    DISP_FREQ = 5
    VER_FREQ = 100
    # forget data update and remain data update
    for i in range(10):
        if i < 5:  # max steps + min steps
            print("\033[0;31;40mmax steps + min steps\033[0m")
            epoch = superepoch * 15 + i
            # lr_scheduler.step(epoch)
            lr = sgda_adjust_learning_rate(epoch, cfg, optimizer)

            for inputs_forget, labels_forget in iter(data_loader_forget):
                inputs_forget = inputs_forget.to(device)
                labels_forget = labels_forget.to(device)

                outputs_forget, _ = student(inputs_forget.float(), labels_forget)

                # compute teacher outputs
                with torch.no_grad():
                    teacher_outputs, _ = teacher(inputs_forget.float(), labels_forget)

                # compute kd loss
                loss_kd = criterionKD(outputs_forget, teacher_outputs)
                # compute SGDA loss
                loss_sgda = param_dist(student, swa_model, p=sgda_smoothing)
                # total loss
                loss = -loss_kd + loss_sgda

                losses_total_forget.update(loss.item(), inputs_forget.size(0))
                losses_kd_forget.update(loss_kd.item(), inputs_forget.size(0))

                # ===================backward=====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # display training loss and accuracy every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    epoch_loss_kd_forget = losses_kd_forget.avg
                    epoch_loss_total_forget = losses_total_forget.avg

                    wandb.log(
                        {
                            "epoch_loss_kd_forget-{}".format(
                                task_i
                            ): epoch_loss_kd_forget,
                            "epoch_loss_total_forget-{}".format(
                                task_i
                            ): epoch_loss_total_forget,
                        }
                    )

                    print(
                        "Task {} Epoch {} Batch {}\t"
                        "Forget Training kd Loss {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t"
                        "Forget Training total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t".format(
                            task_i,
                            epoch + 1,
                            batch + 1,
                            loss_kd=losses_kd_forget,
                            loss_total=losses_total_forget,
                        )
                    )

                    # reset average meters
                    losses_kd_forget.reset()
                    losses_total_forget.reset()

                batch += 1

            for inputs_remain, labels_remain in iter(remain_loader):
                inputs_remain = inputs_remain.to(device)
                labels_remain = labels_remain.to(device)

                outputs_remain, _ = student(inputs_remain.float(), labels_remain)

                # compute teacher outputs
                with torch.no_grad():
                    teacher_outputs, _ = teacher(inputs_remain.float(), labels_remain)

                # compute kd loss
                loss_kd = criterionKD(outputs_remain, teacher_outputs)
                # compute CE loss
                loss_CE = criterion(outputs_remain, labels_remain)
                # compute SGDA loss
                loss_sgda = param_dist(student, swa_model, p=sgda_smoothing)

                # total loss
                loss = sgda_gamma * loss_CE + sgda_alpha * loss_kd + loss_sgda

                losses_total_remain.update(loss.item(), inputs_remain.size(0))
                losses_kd_remain.update(loss_kd.item(), inputs_remain.size(0))
                losses_CE.update(loss_CE.item(), inputs_remain.size(0))

                # ===================backward=====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # display training loss and accuracy every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    epoch_loss_kd_remain = losses_kd_remain.avg
                    epoch_loss_total_remain = losses_total_remain.avg
                    epoch_loss_CE = losses_CE.avg

                    wandb.log(
                        {
                            "epoch_loss_kd_remain-{}".format(
                                task_i
                            ): epoch_loss_kd_remain,
                            "epoch_loss_total_remain-{}".format(
                                task_i
                            ): epoch_loss_total_remain,
                            "epoch_loss_CE-{}".format(task_i): epoch_loss_CE,
                        }
                    )

                    print(
                        "Task {} Epoch {} Batch {}\t"
                        "Remain Training kd Loss {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t"
                        "Remain Training total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
                        "Remain Training CE Loss {loss_CE.val:.4f} ({loss_CE.avg:.4f})\t".format(
                            task_i,
                            epoch + 1,
                            batch + 1,
                            loss_kd=losses_kd_remain,
                            loss_total=losses_total_remain,
                            loss_CE=losses_CE,
                        )
                    )

                    # reset average meters
                    losses_kd_remain.reset()
                    losses_total_remain.reset()
                    losses_CE.reset()

                batch += 1

        else:
            # min steps
            print("\033[0;31;40mmin steps\033[0m")
            epoch = superepoch * 15 + i
            # lr_scheduler.step(epoch)
            lr = sgda_adjust_learning_rate(epoch, cfg, optimizer)

            for inputs_remain, labels_remain in iter(remain_loader):
                inputs_remain = inputs_remain.to(device)
                labels_remain = labels_remain.to(device)

                outputs_remain, _ = student(inputs_remain.float(), labels_remain)

                # compute teacher outputs
                with torch.no_grad():
                    teacher_outputs, _ = teacher(inputs_remain.float(), labels_remain)

                # compute kd loss
                loss_kd = criterionKD(outputs_remain, teacher_outputs)
                # compute CE loss
                loss_CE = criterion(outputs_remain, labels_remain)
                # compute SGDA loss
                loss_sgda = param_dist(student, swa_model, p=sgda_smoothing)

                # total loss
                loss = sgda_gamma * loss_CE + sgda_alpha * loss_kd + loss_sgda

                losses_total_remain.update(loss.item(), inputs_remain.size(0))
                losses_kd_remain.update(loss_kd.item(), inputs_remain.size(0))
                losses_CE.update(loss_CE.item(), inputs_remain.size(0))

                # ===================backward=====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # display training loss and accuracy every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    epoch_loss_kd_remain = losses_kd_remain.avg
                    epoch_loss_total_remain = losses_total_remain.avg
                    epoch_loss_CE = losses_CE.avg

                    wandb.log(
                        {
                            "epoch_loss_kd_remain-{}".format(
                                task_i
                            ): epoch_loss_kd_remain,
                            "epoch_loss_total_remain-{}".format(
                                task_i
                            ): epoch_loss_total_remain,
                            "epoch_loss_CE-{}".format(task_i): epoch_loss_CE,
                        }
                    )

                    print(
                        "Task {} Epoch {} Batch {}\t"
                        "Remain Training kd Loss {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t"
                        "Remain Training total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
                        "Remain Training CE Loss {loss_CE.val:.4f} ({loss_CE.avg:.4f})\t".format(
                            task_i,
                            epoch + 1,
                            batch + 1,
                            loss_kd=losses_kd_remain,
                            loss_total=losses_total_remain,
                            loss_CE=losses_CE,
                        )
                    )

                    # reset average meters
                    losses_kd_remain.reset()
                    losses_total_remain.reset()
                    losses_CE.reset()
                with torch.no_grad():
                    if ((batch + 1) % VER_FREQ == 0) and batch != 0:
                        highest_H_mean = evaluate(
                            student,
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
                            testloader_open=testloader_open
                        )
                        student.train()

                batch += 1

    swa_model.update_parameters(student)

    return (
        batch,
        highest_H_mean,
        losses_CE,
        losses_total_forget,
        losses_total_remain,
        losses_kd_forget,
        losses_kd_remain,
        swa_model,
    )
