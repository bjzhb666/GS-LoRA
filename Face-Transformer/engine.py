import torch
from util.utils import train_accuracy
import util.utils as util
from util.data_prefetcher import data_prefetcher
import wandb
from util.utils import get_time
import os

def train_one_epoch(model:torch.nn.Module,
                    dataloader_forget:torch.utils.data.DataLoader,
                    dataloader_remain:torch.utils.data.DataLoader,
                    device:torch.device,
                    criterion:torch.nn.Module,
                    optimizer:torch.optim.Optimizer,
                    epoch:int,
                    losses_forget:util.AverageMeter,
                    losses_remain:util.AverageMeter,
                    losses_total:util.AverageMeter,
                    losses_structure:util.AverageMeter,
                    top1_forget:util.AverageMeter,
                    top1_remain:util.AverageMeter,
                    beta:float,
                    alpha:float,
                    BND:float,
                    batch:int,
                    testloader_forget:torch.utils.data.DataLoader,
                    testloader_remain:torch.utils.data.DataLoader,
                    forget_acc_before:float,
                    highest_H_mean:float,
                    cfg:dict):
    """
    Train the model for one epoch and evaluate on test set and save checkpoints
    :return: batch(int), highest_H_mean(int)
    """
    model.train()
    # print('Create data prefetcher...')
    prefetcher_forget = data_prefetcher(dataloader_forget, device, prefetch=True)
    inputs_forget, labels_forget = prefetcher_forget.next() # 已经将数据移动到GPU上了

    DISP_FREQ = 3
    VER_FREQ = 3
    # import pdb; pdb.set_trace()
    for inputs_remain, labels_remain in iter(dataloader_remain):
        inputs_remain = inputs_remain.to(device)
        labels_remain = labels_remain.to(device)
        outputs_remain = model(inputs_remain.float())

        # compute remain loss
        loss_remain = criterion(outputs_remain, labels_remain)
        prec1_remain = train_accuracy(outputs_remain.data, labels_remain, topk=(1,))
        # import pdb; pdb.set_trace() 
        losses_remain.update(loss_remain.data.item(),inputs_remain.size(0))
        top1_remain.update(prec1_remain.data.item(), inputs_remain.size(0))

        outputs_forget = model(inputs_forget.float())
        # compute forget loss
        loss_forget = criterion(outputs_forget, labels_forget)
        prec1_forget = train_accuracy(outputs_forget.data, labels_forget, topk=(1,))
        
        loss_forget = -loss_forget # maximize the loss
        # loss_forget = torch.functional.F.relu(BND-loss_forget) # bounded loss
        losses_forget.update(beta*loss_forget.data.item(), inputs_forget.size(0))
        top1_forget.update(prec1_forget.data.item(), inputs_forget.size(0))

        # compute structure loss
        structure_loss = get_structure_loss(model)
        losses_structure.update(alpha*structure_loss.data.item(), inputs_remain.size(0))
        # compute regularization loss

        # compute total loss
        loss_total = loss_forget * beta + loss_remain + structure_loss * alpha
        losses_total.update(loss_total.data.item(), inputs_remain.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # display training loss & accuracy every DISP_FREQ iterations
        if ((batch+1)%DISP_FREQ ==0) and batch != 0:
            epoch_loss_forget = losses_forget.avg
            epoch_loss_remain = losses_remain.avg
            epoch_loss_total = losses_total.avg
            epoch_acc_forget = top1_forget.avg
            epoch_acc_remain = top1_remain.avg

            wandb.log({"epoch_loss_forget": epoch_loss_forget,
                          "epoch_loss_remain": epoch_loss_remain,
                          "epoch_acc_forget": epoch_acc_forget,
                          "epoch_acc_remain": epoch_acc_remain,
                          "epoch_loss_total":epoch_loss_total}, step=batch+1)

            print('Epoch {} Batch {}\t'
                      'Training forget Loss {loss_forget.val:.4f} ({loss_forget.avg:.4f})\t'
                      'Training remain Loss {loss_remain.val:.4f} ({loss_remain.avg:.4f})\t'
                      'Training structure Loss {loss_structure.val:.4f} ({loss_structure.avg:.4f})\t'
                      'Training total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t'
                      'Training forget Prec@1 {top1_forget.val:.3f} ({top1_forget.avg:.3f})\t'
                      'Training remain Prec@1 {top1_remain.val:.3f} ({top1_remain.avg:.3f})'.format(
                          epoch + 1,
                          batch + 1,
                          loss_forget=losses_forget,
                            loss_remain=losses_remain,
                          top1_forget=top1_forget,
                            top1_remain=top1_remain,
                            loss_structure=losses_structure,
                            loss_total=losses_total))
            
            # reset average meters
            losses_forget = util.AverageMeter()
            losses_remain = util.AverageMeter()
            top1_forget = util.AverageMeter()
            top1_remain = util.AverageMeter()
            losses_total = util.AverageMeter()
            losses_structure = util.AverageMeter()

        if ((batch+1)%VER_FREQ ==0) and batch != 0:
            highest_H_mean=evaluate(model, testloader_forget,testloader_remain, device, batch, epoch,
                        forget_acc_before=forget_acc_before, highest_H_mean=highest_H_mean, cfg=cfg)
            model.train()

        batch+=1

        # prefetch next batch
        inputs_forget, labels_forget = prefetcher_forget.next()
        if inputs_forget is None:
            prefetcher_forget = data_prefetcher(dataloader_forget, device, prefetch=True)
            inputs_forget, labels_forget = prefetcher_forget.next()

    return batch, highest_H_mean, losses_forget, losses_remain, top1_forget, top1_remain, losses_total, losses_structure

def train_one_epoch_regularzation(model:torch.nn.Module,):
    pass
def evaluate(model:torch.nn.Module,
             testloader_forget:torch.utils.data.DataLoader,
             testloader_remain:torch.utils.data.DataLoader,
             device:torch.device,
             batch:int,
             epoch:int,
             forget_acc_before:float,
             highest_H_mean:float,
             cfg:dict):
    model.eval()
    print("Perfom evaluation on test set and save checkpoints...")
    

    # 遍历测试集
    forget_acc = eval_data(model, testloader_forget, device, 'forget', batch)
    remain_acc = eval_data(model, testloader_remain, device, 'remain', batch)

    forget_drop = forget_acc_before - forget_acc
    Hmean = 2*forget_drop*remain_acc/(forget_drop+remain_acc)
    
    # save checkpoints per epoch
    if Hmean > highest_H_mean:
        highest_H_mean = Hmean
        if cfg['MULTI_GPU']:
            torch.save(model.module.state_dict(), 
                       os.path.join(
                                cfg['WORK_PATH'],
                                "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth"
                                .format(cfg['BACKBONE_NAME'], epoch + 1, batch + 1,
                                        get_time())))
        else:
            torch.save(model.state_dict(),
                          os.path.join(
                                  cfg['WORK_PATH'],
                                  "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth"
                                  .format(cfg['BACKBONE_NAME'], epoch + 1, batch + 1,
                                         get_time())))
        
        # set the number of checkpoints to be saved:2 (one additional config.txt)
        if len(os.listdir(cfg['WORK_PATH'])) >= 3:
            checkpoints = list(filter(lambda f:f.endswith('.pth'), os.listdir(cfg['WORK_PATH'])))
            checkpoints.sort(key=lambda f:os.path.getmtime(os.path.join(cfg['WORK_PATH'], f)))
            os.remove(os.path.join(cfg['WORK_PATH'], checkpoints[0]))

    return highest_H_mean    


def eval_data(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              device:torch.device,
              mode:str,
              batch:int=0):
    '''
    Evaluate the model on test set, return the accuracy (0-100)
    '''
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

    # 打印测试精度
    accuracy = 100 * correct / total
    print('Test {} Accuracy:{:2f}%'.format(mode, accuracy))
    wandb.log({"Test {} Accuracy".format(mode): accuracy}, step=batch+1)

    return accuracy


def get_structure_loss(model:torch.nn.Module):
    if isinstance(model, torch.nn.DataParallel):
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    learnable_params_name = [name for name, param in model_without_ddp.named_parameters() if param.requires_grad]

    group_layers = []
    '''
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
    '''
    for i in range(6):
        group_item = []
        group_item.append('transformer.layers.{}.1.fn.fn.net.0.lora_A'.format(i))
        group_item.append('transformer.layers.{}.1.fn.fn.net.0.lora_B'.format(i))
        group_item.append('transformer.layers.{}.1.fn.fn.net.3.lora_A'.format(i))
        group_item.append('transformer.layers.{}.1.fn.fn.net.3.lora_B'.format(i))
        group_layers.append(group_item)

    # get the parameters
    group_params = []
    for group_item in group_layers:
        group_param = []
        for item in group_item:
            group_param.append(model_without_ddp.get_parameter(item) if item in learnable_params_name else None)
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