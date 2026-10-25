"""Teacher-student baselines: LwF, DER / DER++, FDR.

Each keeps a frozen copy of the model from before the task and constrains the
student against it on the remain stream -- LwF through logit distillation,
DER through the embeddings, FDR through the logit distance. The shared
training loop lives in :mod:`gslora.methods.base`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaselineTrainer, StepOutput

__all__ = ["LwfTrainer", "DerTrainer", "FdrTrainer", "lwf_kd_loss"]


def lwf_kd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0
) -> torch.Tensor:
    r"""Temperature-scaled KD, :math:`T^2 \cdot \mathrm{KL}(p_t \Vert p_s)`."""
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature**2)


class _TeacherStudentTrainer(BaselineTrainer):
    """Shared: a frozen teacher evaluated on the remain batch."""

    def __init__(self, *args, teacher: nn.Module, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def _teacher_on(self, x: torch.Tensor, y: torch.Tensor):
        with torch.no_grad():
            logits, emb = self.teacher(x, y)
        return logits.detach(), emb.detach()


class LwfTrainer(_TeacherStudentTrainer):
    """Learning-without-Forgetting: CE on the (relabelled) forget set,
    CE + KD on the remain set."""

    loss_names = ("ce_forget", "kd", "ce_remain", "total")

    def step(self, forget_batch, remain_batch) -> StepOutput:
        x_f, y_f = forget_batch
        x_r, y_r = remain_batch
        logits_f, _ = self.model(x_f, y_f)
        logits_r, _ = self.model(x_r, y_r)
        teacher_logits, _ = self._teacher_on(x_r, y_r)

        ce_forget = self.criterion(logits_f, y_f)
        ce_remain = self.criterion(logits_r, y_r)
        kd = lwf_kd_loss(logits_r, teacher_logits, self.cfg.lwf_T)
        total = ce_forget + self.cfg.lwf_lambda_kd * kd + self.cfg.lwf_lambda_remain * ce_remain
        return StepOutput(ce_forget=ce_forget, kd=kd, ce_remain=ce_remain, total=total)


class DerTrainer(_TeacherStudentTrainer):
    """Dark Experience Replay. ``--DER_plus`` adds a second replay CE term."""

    loss_names = ("ce_forget", "der", "ce_replay", "total")

    def step(self, forget_batch, remain_batch) -> StepOutput:
        x_f, y_f = forget_batch
        x_r, y_r = remain_batch
        logits_f, _ = self.model(x_f, y_f)
        _, emb_r = self.model(x_r, y_r)
        _, teacher_emb = self._teacher_on(x_r, y_r)

        ce_forget = self.criterion(logits_f, y_f)
        # per-sample, then averaged, so --DER_lambda is comparable across
        # --batch_size settings
        der = (emb_r - teacher_emb).pow(2).sum(dim=1).mean()

        ce_replay = torch.zeros((), device=der.device)
        if self.cfg.der_plus:
            logits_r2, _ = self.model(x_r, y_r)
            ce_replay = self.criterion(logits_r2, y_r)

        total = ce_forget + self.cfg.der_lambda * der + self.cfg.der_plus_lambda * ce_replay
        return StepOutput(ce_forget=ce_forget, der=der, ce_replay=ce_replay, total=total)


class FdrTrainer(_TeacherStudentTrainer):
    """Function-space distance regularisation (Benjamin et al., ICLR 2019)."""

    loss_names = ("ce_forget", "fdr", "total")

    def step(self, forget_batch, remain_batch) -> StepOutput:
        x_f, y_f = forget_batch
        x_r, y_r = remain_batch
        logits_f, _ = self.model(x_f, y_f)
        logits_r, _ = self.model(x_r, y_r)
        teacher_logits, _ = self._teacher_on(x_r, y_r)

        ce_forget = self.criterion(logits_f, y_f)
        fdr = torch.norm(logits_r - teacher_logits, p=2, dim=1).mean()
        total = ce_forget + self.cfg.fdr_lambda * fdr
        return StepOutput(ce_forget=ce_forget, fdr=fdr, total=total)
