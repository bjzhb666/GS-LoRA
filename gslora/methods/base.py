"""Shared scaffolding for the forgetting baselines.

Every baseline in ``baselines/*.py`` repeated the same ~70 lines of
display/evaluation boilerplate around a 10-line loss computation:

    if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
        ... build a dict of .avg values ...
        wandb.log(...)
        print("Task {} Epoch {} Batch {}\\t ...".format(...))
        ... reset every meter ...
    with torch.no_grad():
        if ((batch + 1) % VER_FREQ == 0) and batch != 0:
            highest_H_mean = evaluate(...)
            model.train()
    batch += 1

(Note ``and batch != 0`` combined with ``(batch + 1) % N`` means the very first
report is skipped only when ``N == 1``; it is vestigial.)

``BaselineTrainer`` owns that loop so each method only implements ``step()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping

import torch
import torch.nn as nn

from ..config import Config
from ..engine.evaluator import Evaluator, ForwardFn
from ..metrics import AverageMeter
from ..tracking import Tracker

__all__ = ["BaselineTrainer", "StepOutput"]


class StepOutput(dict):
    """``{name: scalar}`` losses for one step, plus the tensor to backprop.

    The ``total`` key is required and must be a tensor; everything else is
    logged.
    """

    @property
    def total(self) -> torch.Tensor:
        return self["total"]


class BaselineTrainer(ABC):
    """Template: subclasses implement :meth:`step`, the loop is shared."""

    #: names of the scalars :meth:`step` reports; one meter is made per name
    loss_names: tuple[str, ...] = ("total",)

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        tracker: Tracker,
        evaluator: Evaluator,
        *,
        task_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.tracker = tracker
        self.evaluator = evaluator
        self.task_index = task_index
        self.batch = 0
        self.meters: dict[str, AverageMeter] = {n: AverageMeter() for n in self.loss_names}

    # ------------------------------------------------------------------
    @abstractmethod
    def step(self, forget_batch, remain_batch) -> StepOutput:
        """Compute the losses for one optimisation step (no backward)."""

    def eval_forward(self) -> ForwardFn | None:
        """Override when evaluation needs a composite forward (LIRF)."""
        return None

    def on_epoch_start(self, epoch: int) -> None:
        self.model.train()

    # ------------------------------------------------------------------
    def train_one_epoch(self, pairs: Iterable, epoch: int) -> None:
        self.on_epoch_start(epoch)
        for forget_batch, remain_batch in pairs:
            out = self.step(forget_batch, remain_batch)
            loss = out.total

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            n = forget_batch[0].size(0)
            for name, meter in self.meters.items():
                value = out.get(name)
                if value is None:
                    continue
                meter.update(float(value.item() if torch.is_tensor(value) else value), n)

            self.batch += 1
            self._maybe_report(epoch)
            self._maybe_evaluate(epoch)

    def _maybe_report(self, epoch: int) -> None:
        if self.cfg.disp_freq <= 0 or self.batch % self.cfg.disp_freq:
            return
        suffix = f"/task{self.task_index}"
        self.tracker.log(
            {f"loss/{n}{suffix}": m.avg for n, m in self.meters.items()} | {"epoch": epoch},
            step=self.batch,
        )
        body = " ".join(f"{n} {m.val:.4f}({m.avg:.4f})" for n, m in self.meters.items())
        print(f"[{self.cfg.method}] task {self.task_index} epoch {epoch + 1} batch {self.batch} | {body}")
        for m in self.meters.values():
            m.reset()

    def _maybe_evaluate(self, epoch: int) -> None:
        if self.cfg.ver_freq <= 0 or self.batch % self.cfg.ver_freq:
            return
        self.evaluator.run(
            self.model, forward=self.eval_forward(), epoch=epoch, batch=self.batch
        )
        self.model.train()
