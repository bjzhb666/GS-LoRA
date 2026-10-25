"""LIRF: Learning to Impart Representations for Forgetting.

The network is split in half. The student's bottom half is trained so that,
when fed to the frozen teacher's top half, the forget classes are destroyed
while the remain classes survive; a "deposit" bottom half simultaneously learns
to *preserve* the forgotten knowledge so it can be restored later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..engine.evaluator import ForwardFn, lirf_forward
from ..losses import attention_transfer_loss
from .base import BaselineTrainer, StepOutput

__all__ = ["LirfTrainer"]


class LirfTrainer(BaselineTrainer):
    loss_names = ("ce", "at", "kd", "deposit", "replay", "total")

    def __init__(
        self,
        *args,
        teacher_low: nn.Module,
        teacher_up: nn.Module,
        deposit_low: nn.Module,
        forget_class_ids: list[int],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_low = teacher_low.eval()
        self.teacher_up = teacher_up.eval()
        self.deposit_low = deposit_low
        for module in (self.teacher_low, self.teacher_up):
            for p in module.parameters():
                p.requires_grad_(False)

        # The class order is shuffled and labels stay global, so the forgotten
        # classes are not a contiguous logit range: select them with a mask.
        self.register_forget_mask(forget_class_ids)

    def register_forget_mask(self, forget_class_ids: list[int]) -> None:
        mask = torch.zeros(self.cfg.num_classes, dtype=torch.bool, device=self.cfg.device)
        mask[torch.as_tensor(sorted(forget_class_ids), device=self.cfg.device)] = True
        self.forget_mask = mask
        self.retain_mask = ~mask

    def eval_forward(self) -> ForwardFn:
        return lirf_forward(self.model, self.teacher_up)

    def on_epoch_start(self, epoch: int) -> None:
        self.model.train()
        self.deposit_low.train()
        self.teacher_low.eval()
        self.teacher_up.eval()

    def _kd(self, student_logits, teacher_logits, mask) -> torch.Tensor:
        t = self.cfg.lirf_T
        return F.kl_div(
            F.log_softmax(student_logits[:, mask] / t, dim=1),
            F.softmax(teacher_logits[:, mask] / t, dim=1),
            reduction="batchmean",
        ) * (self.cfg.lirf_alpha * t * t)

    def step(self, forget_batch, remain_batch) -> StepOutput:
        x_f, y_f = forget_batch
        x_r, y_r = remain_batch
        alpha = self.cfg.lirf_alpha

        student_mid = self.model(x_f)
        student_logits, _ = self.teacher_up(student_mid, y_f)

        deposit_mid = self.deposit_low(x_f)
        deposit_logits, _ = self.teacher_up(deposit_mid, y_f)

        with torch.no_grad():
            teacher_mid = self.teacher_low(x_f)
            teacher_logits, _ = self.teacher_up(teacher_mid, y_f)

        ce = self.criterion(student_logits, y_f) * (1 - alpha)
        at = attention_transfer_loss(student_mid, teacher_mid.detach())
        kd = self._kd(student_logits, teacher_logits, self.retain_mask)
        deposit = self._kd(deposit_logits, teacher_logits, self.forget_mask) + self.criterion(
            deposit_logits, y_f
        ) * (1 - alpha)

        student_remain_mid = self.model(x_r)
        remain_logits, _ = self.teacher_up(student_remain_mid, y_r)
        replay = self.criterion(remain_logits, y_r)

        total = (
            ce
            - self.cfg.lirf_w_at * at
            + self.cfg.lirf_w_kd * kd
            + self.cfg.lirf_w_deposit * deposit
            + self.cfg.lirf_w_replay * replay
        )
        return StepOutput(ce=ce, at=at, kd=kd, deposit=deposit, replay=replay, total=total)
