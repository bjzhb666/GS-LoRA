"""SCRUB / SCRUB-S (Kurmanji et al.).

Alternating max-steps (push the student *away* from the teacher on the forget
set) and min-steps (pull it towards the teacher, plus CE, on the remain set),
optionally smoothed towards a running average of the student (SGDA).

``baselines/SCRUBtrain.py`` had the *entire* min-step body written twice --
once inside the ``if i < max_steps`` branch and once in the ``else`` -- and
only the ``else`` copy ran evaluation. It also computed
``epoch = superepoch * 15 + i`` while iterating ``for i in range(10)``, so the
epoch counter that drives the LR decay skipped 5 values per super-epoch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..losses import DistillKL
from .base import BaselineTrainer, StepOutput

__all__ = ["ScrubTrainer", "sgda_adjust_learning_rate", "param_distance"]


def sgda_adjust_learning_rate(
    epoch: int, optimizer: torch.optim.Optimizer, base_lr: float, decay_epoch: int, decay_rate: float
) -> float:
    """Step decay: ``base_lr * decay_rate ** (epoch // decay_epoch)``."""
    steps = max(0, epoch // decay_epoch) if decay_epoch > 0 else 0
    lr = base_lr * (decay_rate**steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def param_distance(model: nn.Module, reference: nn.Module, weight: float) -> torch.Tensor:
    """Frobenius distance between two parameter sets, scaled by ``weight``."""
    device = next(model.parameters()).device
    if weight == 0:
        return torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    for p1, p2 in zip(model.parameters(), reference.parameters()):
        total = total + torch.norm(p1 - p2.detach())
    return weight * total


class ScrubTrainer(BaselineTrainer):
    """One ``train_one_epoch`` call is one SCRUB step within a super-epoch.

    ``phase`` selects whether the step ascends on the forget stream (``max``)
    or descends on the remain stream (``min``); :meth:`run_superepoch` drives
    the alternation.
    """

    loss_names = ("ce", "kd_forget", "kd_remain", "sgda", "total")

    def __init__(self, *args, teacher: nn.Module, swa_model: nn.Module, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.swa_model = swa_model
        self.kd = DistillKL(self.cfg.kd_T)
        self.phase: str = "max"  # 'max' (ascend on forget) or 'min' (descend on remain)

    def _teacher_logits(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.teacher(x, y)
        return logits.detach()

    def step(self, forget_batch, remain_batch) -> StepOutput:
        device = self.cfg.device
        zero = torch.zeros((), device=device)
        sgda = param_distance(self.model, self.swa_model, self.cfg.sgda_smoothing)

        if self.phase == "max":
            x, y = forget_batch
            logits, _ = self.model(x, y)
            kd_forget = self.kd(logits, self._teacher_logits(x, y))
            total = -kd_forget + sgda
            return StepOutput(ce=zero, kd_forget=kd_forget, kd_remain=zero, sgda=sgda, total=total)

        x, y = remain_batch
        logits, _ = self.model(x, y)
        kd_remain = self.kd(logits, self._teacher_logits(x, y))
        ce = self.criterion(logits, y)
        total = self.cfg.sgda_gamma * ce + self.cfg.sgda_alpha * kd_remain + sgda
        return StepOutput(ce=ce, kd_forget=zero, kd_remain=kd_remain, sgda=sgda, total=total)

    # ------------------------------------------------------------------
    def run_superepoch(self, pairs, superepoch: int) -> None:
        """``scrub_max_steps`` max+min epochs, then min-only for the rest."""
        cfg = self.cfg
        steps = cfg.scrub_steps_per_superepoch
        for i in range(steps):
            epoch = superepoch * steps + i          # contiguous, unlike `* 15 + i`
            sgda_adjust_learning_rate(
                epoch, self.optimizer, cfg.lr, cfg.scrub_decay_epoch, cfg.scrub_decay_rate
            )
            if i < cfg.scrub_max_steps:
                self.phase = "max"
                self.train_one_epoch(pairs, epoch)
            self.phase = "min"
            self.train_one_epoch(pairs, epoch)
        self.swa_model.update_parameters(self.model)
