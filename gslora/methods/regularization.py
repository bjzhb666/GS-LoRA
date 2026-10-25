"""Weight-anchoring baselines: L2, EWC, MAS -- plus plain retraining.

Each keeps a snapshot of the weights after a task plus a per-parameter
importance estimate, and penalises later tasks for moving important weights
away from the snapshot. ``--online`` accumulates a single importance estimate
instead of one per task.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..checkpoint import unwrap_model
from ..config import Config
from ..losses import elastic_reg_loss
from .base import BaselineTrainer, StepOutput

__all__ = [
    "RegularizationTrainer",
    "RetrainTrainer",
    "estimate_importance",
    "snapshot_params",
]

ImportanceDict = dict[str, torch.Tensor]
RegTerms = dict[int, dict[str, ImportanceDict]]


def snapshot_params(model: nn.Module) -> ImportanceDict:
    return {
        n: p.detach().clone()
        for n, p in unwrap_model(model).named_parameters()
        if p.requires_grad
    }


def _zeros_like_trainable(model: nn.Module) -> ImportanceDict:
    return {
        n: torch.zeros_like(p)
        for n, p in unwrap_model(model).named_parameters()
        if p.requires_grad
    }


def estimate_importance(
    kind: str,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    warm_start: ImportanceDict | None = None,
) -> ImportanceDict:
    """Per-parameter importance for L2 / EWC (Fisher) / MAS.

    Args:
        kind: ``"l2"``, ``"ewc"`` or ``"mas"``.
        warm_start: previous importance to accumulate onto (``--online``).
    """
    inner = unwrap_model(model)
    if kind == "l2":
        return {n: torch.ones_like(p) for n, p in inner.named_parameters() if p.requires_grad}

    if kind not in ("ewc", "mas"):
        raise ValueError(f"kind must be l2/ewc/mas, got {kind!r}")

    importance = warm_start if warm_start is not None else _zeros_like_trainable(model)
    trainable = {n: p for n, p in inner.named_parameters() if p.requires_grad}
    n_batches = len(dataloader)
    if n_batches == 0:
        raise ValueError("importance dataloader is empty")

    was_training = inner.training
    inner.eval()
    try:
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            logits, _ = model(images, labels)

            if kind == "ewc":
                loss = criterion(logits, labels)
            else:  # MAS: squared L2 norm of the output
                loss = logits.pow(2).mean()

            inner.zero_grad(set_to_none=True)
            loss.backward()
            for name, acc in importance.items():
                grad = trainable[name].grad
                if grad is None:
                    continue
                # EWC accumulates squared grads, MAS absolute grads; both are
                # averaged over batches so the scale does not depend on
                # --batch_size.
                acc += (grad.detach() ** 2 if kind == "ewc" else grad.detach().abs()) / n_batches
        inner.zero_grad(set_to_none=True)
    finally:
        inner.train(was_training)
    return importance


class RegularizationTrainer(BaselineTrainer):
    """CE on the relabelled forget set + a quadratic anchor to the old weights."""

    loss_names = ("ce", "reg", "total")

    def __init__(self, *args, regularization_terms: RegTerms | None = None, reg_lambda: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.regularization_terms = regularization_terms or {}
        self.reg_lambda = reg_lambda

    def step(self, forget_batch, remain_batch) -> StepOutput:
        # These methods only optimise on the (relabelled) forget stream; with
        # --replay the caller feeds a concatenated loader as `forget_batch`.
        x, y = forget_batch
        logits, _ = self.model(x, y)
        ce = self.criterion(logits, y)
        reg = elastic_reg_loss(
            unwrap_model(self.model), self.regularization_terms, self.reg_lambda, self.cfg.device
        )
        return StepOutput(ce=ce, reg=reg, total=ce + reg)


class RetrainTrainer(BaselineTrainer):
    """Upper bound: train a freshly initialised model on the remain set only."""

    loss_names = ("ce", "total")

    def step(self, forget_batch, remain_batch) -> StepOutput:
        x, y = remain_batch
        logits, _ = self.model(x, y)
        ce = self.criterion(logits, y)
        return StepOutput(ce=ce, total=ce)
