import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step
    :param epoch: current epoch
    :param opt: the options (in a dict form)
    :param optimizer: the optimizer
    :return: the new learning rate"""
    steps = np.sum(epoch > np.asarray(opt["lr_decay_epochs"]))
    new_lr = opt["sgda_learning_rate"]
    if steps > 0:
        new_lr = opt["sgda_learning_rate"] * (opt["lr_decay_rate"] ** steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
    return new_lr


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


def param_dist(model, swa_model, p):
    # This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.0
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p="fro")
    return p * dist
