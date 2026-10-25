"""The one evaluation path.

``engine.py`` and ``engine_cl.py`` each had ``eval_data`` + ``evaluate``, and
``baselines/LIRFtrain.py`` had a third pair. They had diverged:

===========================  ====================  ==========================
behaviour                    engine.py             engine_cl.py
===========================  ====================  ==========================
H-mean denominator           no epsilon (ZeroDiv)  ``+ 1e-8``
model copying                ``deepcopy`` **twice**  none
restores train mode after    no                    yes (at the call site)
``open_acc``                 computed, discarded   computed, discarded
checkpoint retention         ``>= 3`` entries      ``>= 4`` entries
===========================  ====================  ==========================

The double ``deepcopy`` in ``engine.py`` is worth calling out: ``evaluate()``
copied the model, then handed the copy to ``eval_data()`` which copied it
*again*, for each of the forget/remain/open loaders -- four full model copies
per evaluation, every ``VER_FREQ`` batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..checkpoint import save_checkpoint
from ..config import Config
from ..metrics import harmonic_mean
from ..prototypes import eval_mode
from ..tracking import Tracker

__all__ = ["evaluate_accuracy", "EvalResult", "Evaluator"]

# A model may be a plain classifier taking (images, labels) -> (logits, emb),
# or a composite (LIRF's student_low + teacher_up). This is the adapter.
ForwardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def evaluate_accuracy(
    forward: ForwardFn,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Top-1 accuracy in percent. Returns 0.0 for an empty loader."""
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        logits = forward(images, labels)
        predicted = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    if total == 0:
        return 0.0
    return 100.0 * correct / total


def default_forward(model: nn.Module) -> ForwardFn:
    def _forward(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = model(images, labels)
        return out[0] if isinstance(out, tuple) else out

    return _forward


def lirf_forward(student_low: nn.Module, teacher_up: nn.Module) -> ForwardFn:
    def _forward(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits, _ = teacher_up(student_low(images), labels)
        return logits

    return _forward


@dataclass
class EvalResult:
    forget_acc: float
    remain_acc: float
    forget_drop: float
    h_mean: float
    open_acc: float | None = None
    old_acc: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def as_metrics(self, suffix: str = "") -> dict[str, float]:
        out = {
            f"acc/forget{suffix}": self.forget_acc,
            f"acc/remain{suffix}": self.remain_acc,
            f"acc/forget_drop{suffix}": self.forget_drop,
            f"acc/h_mean{suffix}": self.h_mean,
        }
        if self.open_acc is not None:
            out[f"acc/open{suffix}"] = self.open_acc
        if self.old_acc is not None:
            out[f"acc/old{suffix}"] = self.old_acc
        out.update({f"{k}{suffix}": v for k, v in self.extra.items()})
        return out


class Evaluator:
    """Runs forget/remain/open/old evaluation and keeps the best checkpoint."""

    def __init__(
        self,
        cfg: Config,
        tracker: Tracker,
        *,
        forget_loader: DataLoader,
        remain_loader: DataLoader,
        open_loader: DataLoader | None = None,
        old_loader: DataLoader | None = None,
        forget_acc_before: float = 0.0,
        task_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self.forget_loader = forget_loader
        self.remain_loader = remain_loader
        self.open_loader = open_loader
        self.old_loader = old_loader
        self.forget_acc_before = forget_acc_before
        self.task_index = task_index
        self.best_h_mean = 0.0
        self.best_checkpoint: str | None = None

    @property
    def suffix(self) -> str:
        return f"/task{self.task_index}"

    def run(
        self,
        model: nn.Module,
        *,
        forward: ForwardFn | None = None,
        epoch: int = 0,
        batch: int = 0,
        save: bool = True,
        tag: str = "",
    ) -> EvalResult:
        forward = forward or default_forward(model)
        device = self.cfg.device

        with eval_mode(model):
            forget_acc = evaluate_accuracy(forward, self.forget_loader, device)
            remain_acc = evaluate_accuracy(forward, self.remain_loader, device)
            open_acc = (
                evaluate_accuracy(forward, self.open_loader, device)
                if self.open_loader is not None
                else None
            )
            old_acc = (
                evaluate_accuracy(forward, self.old_loader, device)
                if self.old_loader is not None
                else None
            )

        drop = self.forget_acc_before - forget_acc
        result = EvalResult(
            forget_acc=forget_acc,
            remain_acc=remain_acc,
            forget_drop=drop,
            h_mean=harmonic_mean(drop, remain_acc),
            open_acc=open_acc,
            old_acc=old_acc,
        )

        suffix = f"{self.suffix}{('/' + tag) if tag else ''}"
        self.tracker.log({**result.as_metrics(suffix), "epoch": epoch, "batch": batch})

        if save and result.h_mean > self.best_h_mean:
            self.best_h_mean = result.h_mean
            path = save_checkpoint(
                model,
                self.cfg.outdir,
                backbone_name=self.cfg.net,
                epoch=epoch,
                batch=batch,
                keep=self.cfg.keep_checkpoints,
            )
            self.best_checkpoint = str(path)
        return result

    def baseline(self, model: nn.Module, forward: ForwardFn | None = None) -> EvalResult:
        """Evaluate before training and latch ``forget_acc_before``."""
        forward = forward or default_forward(model)
        with eval_mode(model):
            self.forget_acc_before = evaluate_accuracy(
                forward, self.forget_loader, self.cfg.device
            )
        return self.run(model, forward=forward, save=False, tag="before")
