import torch
from util.utils import train_accuracy
import util.utils as util
from util.data_prefetcher import data_prefetcher
import wandb
from util.utils import get_time
import os
from IPython import embed
import torch.nn.functional as F


def train_one_epoch(
    model: torch.nn.Module,
    dataloader_forget: torch.utils.data.DataLoader,
    dataloader_remain: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses_forget: util.AverageMeter,
    losses_remain: util.AverageMeter,
    losses_total: util.AverageMeter,
    losses_structure: util.AverageMeter,
    top1_forget: util.AverageMeter,
    top1_remain: util.AverageMeter,
    beta: float,
    alpha: float,
    BND: float,
    batch: int,
    testloader_forget: torch.utils.data.DataLoader,
    testloader_remain: torch.utils.data.DataLoader,
    forget_acc_before: float,
    highest_H_mean: float,
    cfg: dict,
    task_i: str,
    use_prototype: bool,
    prototype_dict: dict,
    prototype_weight_forget: float,
    prototype_weight_remain: float,
    losses_prototype_forget: util.AverageMeter,
    losses_prototype_remain: util.AverageMeter,
    dataloader_open: torch.utils.data.DataLoader = None,
):
    """
    Train the model for one epoch and evaluate on test set and save checkpoints
    :return: batch(int), highest_H_mean(int)
    """
    model.train()
    criterion.train()
    # print('Create data prefetcher...')
    prefetcher_forget = data_prefetcher(dataloader_forget, device, prefetch=True)
    inputs_forget, labels_forget = (
        prefetcher_forget.next()
    )  # data has already been put on GPU device

    DISP_FREQ = 5
    VER_FREQ = 100
    # import pdb; pdb.set_trace()
    for inputs_remain, labels_remain in iter(dataloader_remain):
        inputs_remain = inputs_remain.to(device)
        labels_remain = labels_remain.to(device)
        outputs_remain, embeds_remain = model(inputs_remain.float(), labels_remain)

        # compute remain loss
        loss_remain = criterion(outputs_remain, labels_remain)
        prec1_remain = train_accuracy(outputs_remain.data, labels_remain, topk=(1,))
        # import pdb; pdb.set_trace()
        losses_remain.update(loss_remain.data.item(), inputs_remain.size(0))
        top1_remain.update(prec1_remain.data.item(), inputs_remain.size(0))

        outputs_forget, embeds_forget = model(inputs_forget.float(), labels_forget)
        # compute forget loss
        loss_forget = criterion(outputs_forget, labels_forget)
        prec1_forget = train_accuracy(outputs_forget.data, labels_forget, topk=(1,))

        # loss_forget = -loss_forget # maximize the loss
        # embed() # debug
        loss_forget = torch.functional.F.relu(BND - loss_forget)  # bounded loss
        losses_forget.update(beta * loss_forget.data.item(), inputs_forget.size(0))
        top1_forget.update(prec1_forget.data.item(), inputs_forget.size(0))

        # compute structure loss
        structure_loss = get_structure_loss(
            model, imagenet=(cfg["DATA_ROOT"] == "./data/imagenet100/")
        )
        losses_structure.update(
            alpha * structure_loss.data.item(), inputs_remain.size(0)
        )
        # compute regularization loss, add prototype distillation loss
        if use_prototype:
            prototype_loss_forget = get_prototype_loss(
                embeds_forget, labels_forget, prototype_dict
            )
            prototype_loss_remain = get_prototype_loss(
                embeds_remain, labels_remain, prototype_dict
            )
            prototype_loss = (
                prototype_weight_forget
                * torch.functional.F.relu(cfg["BND_pro"] - prototype_loss_forget)
                + prototype_weight_remain * prototype_loss_remain
            )
        else:
            prototype_loss_forget = torch.tensor(0.0).to(device)
            prototype_loss_remain = torch.tensor(0.0).to(device)
            prototype_loss = torch.tensor(0.0).to(device)

        losses_prototype_forget.update(
            prototype_weight_forget
            * torch.functional.F.relu(cfg["BND_pro"] - prototype_loss_forget).item(),
            inputs_remain.size(0),
        )
        losses_prototype_remain.update(
            prototype_loss_remain.data.item() * prototype_weight_remain,
            inputs_remain.size(0),
        )

        # compute total loss
        loss_total = (
            loss_forget * beta + loss_remain + structure_loss * alpha + prototype_loss
        )
        losses_total.update(loss_total.data.item(), inputs_remain.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # display training loss & accuracy every DISP_FREQ iterations
        if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
            epoch_loss_forget = losses_forget.avg
            epoch_loss_remain = losses_remain.avg
            epoch_loss_total = losses_total.avg
            epoch_acc_forget = top1_forget.avg
            epoch_acc_remain = top1_remain.avg
            epoch_loss_structure = losses_structure.avg
            epoch_loss_prototype_forget = losses_prototype_forget.avg
            epoch_loss_prototype_remain = losses_prototype_remain.avg

            wandb.log(
                {
                    "epoch_loss_forget-{}".format(task_i): epoch_loss_forget,
                    "epoch_loss_remain-{}".format(task_i): epoch_loss_remain,
                    "epoch_acc_forget-{}".format(task_i): epoch_acc_forget,
                    "epoch_acc_remain-{}".format(task_i): epoch_acc_remain,
                    "epoch_loss_total-{}".format(task_i): epoch_loss_total,
                    "epoch_loss_structure-{}".format(task_i): epoch_loss_structure,
                    "epoch_loss_prototype_forget-{}".format(
                        task_i
                    ): epoch_loss_prototype_forget,
                    "epoch_loss_prototype_remain-{}".format(
                        task_i
                    ): epoch_loss_prototype_remain,
                }
            )

            print(
                "Task {} Epoch {} Batch {}\t"
                "Training forget Loss {loss_forget.val:.4f} ({loss_forget.avg:.4f})\t"
                "Training remain Loss {loss_remain.val:.4f} ({loss_remain.avg:.4f})\t"
                "Training forget prototype Loss {loss_prototype_forget.val:.4f}\t"
                "Training remain prototype Loss {loss_prototype_remain.val:.4f}\t"
                "Training structure Loss {loss_structure.val:.4f} ({loss_structure.avg:.4f})\t"
                "Training total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
                "Training forget Prec@1 {top1_forget.val:.3f} ({top1_forget.avg:.3f})\t"
                "Training remain Prec@1 {top1_remain.val:.3f} ({top1_remain.avg:.3f})".format(
                    task_i,
                    epoch + 1,
                    batch + 1,
                    loss_forget=losses_forget,
                    loss_remain=losses_remain,
                    top1_forget=top1_forget,
                    top1_remain=top1_remain,
                    loss_structure=losses_structure,
                    loss_total=losses_total,
                    loss_prototype_forget=losses_prototype_forget,
                    loss_prototype_remain=losses_prototype_remain,
                )
            )

            # reset average meters
            losses_forget = util.AverageMeter()
            losses_remain = util.AverageMeter()
            top1_forget = util.AverageMeter()
            top1_remain = util.AverageMeter()
            losses_total = util.AverageMeter()
            losses_structure = util.AverageMeter()
            losses_prototype_forget = util.AverageMeter()
            losses_prototype_remain = util.AverageMeter()

        with torch.no_grad():
            if ((batch + 1) % VER_FREQ == 0) and batch != 0:
                if dataloader_open is None:
                    highest_H_mean = evaluate(
                        model,
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
                    )
                else:
                    highest_H_mean = evaluate(
                        model,
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
                        testloader_open=dataloader_open,
                    )

                model.train()

        batch += 1

        # prefetch next batch
        inputs_forget, labels_forget = prefetcher_forget.next()
        if inputs_forget is None:
            prefetcher_forget = data_prefetcher(
                dataloader_forget, device, prefetch=True
            )
            inputs_forget, labels_forget = prefetcher_forget.next()

    return (
        batch,
        highest_H_mean,
        losses_forget,
        losses_remain,
        top1_forget,
        top1_remain,
        losses_total,
        losses_structure,
        losses_prototype_forget,
        losses_prototype_remain,
    )


def evaluate(
    model: torch.nn.Module,
    testloader_forget: torch.utils.data.DataLoader,
    testloader_remain: torch.utils.data.DataLoader,
    device: torch.device,
    batch: int,
    epoch: int,
    forget_acc_before: float,
    highest_H_mean: float,
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    task_i: str,
    testloader_open: torch.utils.data.DataLoader = None,
):
    model.eval()
    for params in optimizer.param_groups:
        lr = params["lr"]
        break
    print("current learning rate:{:.7f}".format(lr))
    print("Perfom evaluation on test set and save checkpoints...")

    forget_acc = eval_data(
        model, testloader_forget, device, "forget-{}".format(task_i), batch
    )
    remain_acc = eval_data(
        model, testloader_remain, device, "remain-{}".format(task_i), batch
    )
    if testloader_open is not None:
        open_acc = eval_data(
            model, testloader_open, device, "open-{}".format(task_i), batch
        )
    forget_drop = forget_acc_before - forget_acc
    Hmean = 2 * forget_drop * remain_acc / (forget_drop + remain_acc + 1e-8)

    # save checkpoints per epoch
    if Hmean > highest_H_mean:
        highest_H_mean = Hmean
        if cfg["MULTI_GPU"]:
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    cfg["WORK_PATH"],
                    "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                        cfg["BACKBONE_NAME"], epoch + 1, batch + 1, get_time()
                    ),
                ),
            )
        else:
            torch.save(
                model.state_dict(),
                os.path.join(
                    cfg["WORK_PATH"],
                    "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                        cfg["BACKBONE_NAME"], epoch + 1, batch + 1, get_time()
                    ),
                ),
            )

        # set the number of checkpoints to be saved:2 (one additional config.txt)
        if len(os.listdir(cfg["WORK_PATH"])) >= 4:
            checkpoints = list(
                filter(lambda f: f.endswith(".pth"), os.listdir(cfg["WORK_PATH"]))
            )
            checkpoints.sort(
                key=lambda f: os.path.getmtime(os.path.join(cfg["WORK_PATH"], f))
            )
            os.remove(os.path.join(cfg["WORK_PATH"], checkpoints[0]))

    return highest_H_mean


def eval_data(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mode: str,
    batch: int = 0,
):
    """
    Evaluate the model on test set, return the accuracy (0-100)
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs, _ = model(images, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print the accuracy
    accuracy = 100 * correct / total
    print("Test {} Accuracy:{:2f}%".format(mode, accuracy))
    wandb.log({"Test {} Accuracy".format(mode): accuracy})

    return accuracy


def get_structure_loss(model: torch.nn.Module, imagenet=False):
    if isinstance(model, torch.nn.DataParallel):
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    learnable_params_name = [
        name
        for name, param in model_without_ddp.named_parameters()
        if param.requires_grad
    ]

    group_layers = []
    """
    transformer.layers.0.1.fn.fn.net.0.lora_A
    transformer.layers.0.1.fn.fn.net.0.lora_B
    transformer.layers.0.1.fn.fn.net.3.lora_A
    transformer.layers.0.1.fn.fn.net.3.lora_B
    transformer.layers.1.1.fn.fn.net.0.lora_A
    transformer.layers.1.1.fn.fn.net.0.lora_B
    transformer.layers.1.1.fn.fn.net.3.lora_A
    transformer.layers.1.1.fn.fn.net.3.lora_B
    transformer.layers.2.1.fn.fn.net.0.lora_A
    transformer.layers.2.1.fn.fn.net.0.lora_B
    transformer.layers.2.1.fn.fn.net.3.lora_A
    transformer.layers.2.1.fn.fn.net.3.lora_B
    transformer.layers.3.1.fn.fn.net.0.lora_A
    transformer.layers.3.1.fn.fn.net.0.lora_B
    transformer.layers.3.1.fn.fn.net.3.lora_A
    transformer.layers.3.1.fn.fn.net.3.lora_B
    transformer.layers.4.1.fn.fn.net.0.lora_A
    transformer.layers.4.1.fn.fn.net.0.lora_B
    transformer.layers.4.1.fn.fn.net.3.lora_A
    transformer.layers.4.1.fn.fn.net.3.lora_B
    transformer.layers.5.1.fn.fn.net.0.lora_A
    transformer.layers.5.1.fn.fn.net.0.lora_B
    transformer.layers.5.1.fn.fn.net.3.lora_A
    transformer.layers.5.1.fn.fn.net.3.lora_B
    """
    if not imagenet:
        for i in range(6):
            group_item = []
            group_item.append("transformer.layers.{}.1.fn.fn.net.0.lora_A".format(i))
            group_item.append("transformer.layers.{}.1.fn.fn.net.0.lora_B".format(i))
            group_item.append("transformer.layers.{}.1.fn.fn.net.3.lora_A".format(i))
            group_item.append("transformer.layers.{}.1.fn.fn.net.3.lora_B".format(i))
            group_layers.append(group_item)
    else:  # imagenet
        for i in range(12):
            group_item = []
            group_item.append("encoder.layers.encoder_layer_{}.mlp.0.lora_A".format(i))
            group_item.append("encoder.layers.encoder_layer_{}.mlp.0.lora_B".format(i))
            group_item.append("encoder.layers.encoder_layer_{}.mlp.3.lora_A".format(i))
            group_item.append("encoder.layers.encoder_layer_{}.mlp.3.lora_B".format(i))
            group_layers.append(group_item)

    # get the parameters
    group_params = []
    for group_item in group_layers:
        group_param = []
        for item in group_item:
            group_param.append(
                model_without_ddp.get_parameter(item)
                if item in learnable_params_name
                else None
            )
        group_params.append(group_param)

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


def get_reg_loss(
    model: torch.nn.Module,
    regularization_terms: dict,
    reg_lambda: float,
    device: torch.device,
):
    l2_loss = torch.tensor(0.0, device=device)
    if regularization_terms is None:
        return l2_loss
    if isinstance(model, torch.nn.DataParallel):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    reg_loss = torch.tensor(0.0, device=device)
    for i, reg_term in regularization_terms.items():
        task_reg_loss = torch.tensor(0.0, device=device)
        importance = reg_term["importance"]
        task_param = reg_term["task_param"]

        for n, p in model_without_ddp.named_parameters():
            if p.requires_grad:
                task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
        reg_loss += task_reg_loss
    l2_loss += reg_lambda * reg_loss
    return l2_loss


def train_one_epoch_regularzation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader_cl_forget: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch: int,
    reg_lambda: float,
    regularization_terms: dict,
    losses_CE: util.AverageMeter,
    losses_regularization: util.AverageMeter,
    losses_total: util.AverageMeter,
    task_i: str,
    testloader_forget: torch.utils.data.DataLoader,
    testloader_remain: torch.utils.data.DataLoader,
    forget_acc_before: float,
    highest_H_mean: float,
    cfg: dict,
    testloader_open: torch.utils.data.DataLoader = None,
):
    model.train()
    criterion.train()

    DISP_FREQ = 5
    VER_FREQ = 100

    for inputs_forget, labels_forget in iter(data_loader_cl_forget):
        inputs_forget = inputs_forget.to(device)
        labels_forget = labels_forget.to(device)
        outputs_forget, embeds_forget = model(inputs_forget.float(), labels_forget)

        # compute CE loss
        loss_forget = criterion(outputs_forget, labels_forget)
        losses_CE.update(loss_forget.data.item(), inputs_forget.size(0))

        # compute regularization loss
        regularization_loss = get_reg_loss(
            model, regularization_terms, reg_lambda, device
        )
        losses_regularization.update(
            regularization_loss.data.item(), inputs_forget.size(0)
        )

        losses = regularization_loss + loss_forget
        losses_total.update(losses.data.item(), inputs_forget.size(0))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # display training loss & accuracy every DISP_FREQ iterations
        if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
            epoch_loss_CE = losses_CE.avg
            epoch_loss_regularization = losses_regularization.avg
            epoch_loss_total = losses_total.avg

            wandb.log(
                {
                    "epoch_loss_CE-{}".format(task_i): epoch_loss_CE,
                    "epoch_loss_regularization-{}".format(
                        task_i
                    ): epoch_loss_regularization,
                    "epoch_loss_total-{}".format(task_i): epoch_loss_total,
                }
            )

            print(
                "Task {} Epoch {} Batch {}\t"
                "Training CE Loss {loss_CE.val:.4f} ({loss_CE.avg:.4f})\t"
                "Training regularization Loss {loss_regularization.val:.4f} ({loss_regularization.avg:.4f})\t"
                "Training total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})".format(
                    task_i,
                    epoch + 1,
                    batch + 1,
                    loss_CE=losses_CE,
                    loss_regularization=losses_regularization,
                    loss_total=losses_total,
                )
            )

            # reset average meters
            losses_CE = util.AverageMeter()
            losses_regularization = util.AverageMeter()
            losses_total = util.AverageMeter()
        with torch.no_grad():
            if ((batch + 1) % VER_FREQ == 0) and batch != 0:
                highest_H_mean = evaluate(
                    model,
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
                model.train()

        batch += 1

    return batch, highest_H_mean, losses_CE, losses_regularization, losses_total


def get_prototype_loss(output, labels, prototype_dict, distance="kl"):
    """
    Calculate the prototype loss to bring the features of each sample closer to the prototype of its corresponding category.

    parameter:
    -output (torch.Tensor): Feature tensor with shape (batch_size, d), where d is the feature dimension.
    -labels (torch.Tensor): sample labels, shape (batch_size,).
    -prototype_dict (dict): dictionary, where key is the category label and value is the prototype vector of the corresponding category.
    -distance (str): distance measurement method, optional 'euclidean' or 'kl'.

    return:
    -loss (torch.Tensor): calculated prototype loss.
    """
    loss = 0.0
    # import pdb; pdb.set_trace()
    # Take out the prototype corresponding to each label to form a tensor
    prototype_tensor = torch.stack(
        [prototype_dict[label.item()] for label in labels]
    ).to(
        output.device
    )  # (batch_size, d)

    if distance == "l2":
        loss = torch.mean((output - prototype_tensor) ** 2)
    elif distance == "kl":
        # import pdb; pdb.set_trace()
        features_log = F.log_softmax(output, dim=1)
        prototype_log = F.log_softmax(prototype_tensor, dim=1)
        loss = F.kl_div(
            features_log, prototype_log, reduction="batchmean", log_target=True
        )

    return loss
