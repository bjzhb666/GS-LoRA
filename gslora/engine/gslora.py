"""The GS-LoRA training step.

One optimisation step consumes a forget batch and a remain batch and
minimises

.. math::
    \beta \cdot \mathrm{relu}(\mathrm{BND} - \mathrm{CE}_{forget})
    + \mathrm{CE}_{remain}
    + \alpha \cdot \mathcal{L}_{structure}
    + \mathcal{L}_{pro}

where the first term is the bounded selective-forgetting objective, the third
is the group-lasso over LoRA groups, and the last is the optional prototype
regulariser of GS-LoRA++.

``--warmup_alpha`` holds :math:`\alpha` at 0 for ``--alpha_epoch`` epochs and
then switches to ``--big_alpha``; ``--beta_decay`` drops :math:`\beta` to
``--small_beta`` after ``--beta_decay_epoch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn as nn

from ..config import Config
from ..lora_utils import collect_group_params, discover_lora_groups
from ..losses import bounded_forget_loss, group_sparse_loss, prototype_loss
from ..metrics import AverageMeter, topk_accuracy
from ..tracking import Tracker
from .evaluator import Evaluator
from .pairing import PairedBatches

__all__ = ["GSLoRATrainer", "EpochStats"]


@dataclass
class EpochStats:
    forget: AverageMeter = field(default_factory=AverageMeter)
    remain: AverageMeter = field(default_factory=AverageMeter)
    structure: AverageMeter = field(default_factory=AverageMeter)
    proto_forget: AverageMeter = field(default_factory=AverageMeter)
    proto_remain: AverageMeter = field(default_factory=AverageMeter)
    total: AverageMeter = field(default_factory=AverageMeter)
    acc_forget: AverageMeter = field(default_factory=AverageMeter)
    acc_remain: AverageMeter = field(default_factory=AverageMeter)

    def reset(self) -> None:
        for meter in vars(self).values():
            meter.reset()

    def as_metrics(self, suffix: str = "") -> dict[str, float]:
        return {
            f"loss/forget{suffix}": self.forget.avg,
            f"loss/remain{suffix}": self.remain.avg,
            f"loss/structure{suffix}": self.structure.avg,
            f"loss/proto_forget{suffix}": self.proto_forget.avg,
            f"loss/proto_remain{suffix}": self.proto_remain.avg,
            f"loss/total{suffix}": self.total.avg,
            f"train_acc/forget{suffix}": self.acc_forget.avg,
            f"train_acc/remain{suffix}": self.acc_remain.avg,
        }

    def format(self, task: int, epoch: int, batch: int) -> str:
        return (
            f"task {task} epoch {epoch + 1} batch {batch + 1} | "
            f"forget {self.forget.val:.4f}({self.forget.avg:.4f}) "
            f"remain {self.remain.val:.4f}({self.remain.avg:.4f}) "
            f"struct {self.structure.val:.4f}({self.structure.avg:.4f}) "
            f"proto_f {self.proto_forget.val:.4f} proto_r {self.proto_remain.val:.4f} "
            f"total {self.total.val:.4f}({self.total.avg:.4f}) | "
            f"acc_f {self.acc_forget.avg:.2f} acc_r {self.acc_remain.avg:.2f}"
        )


class GSLoRATrainer:
    """One task's worth of GS-LoRA optimisation."""

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
        beta: float | None = None,
        prototype_weight_forget: float | None = None,
        prototypes: Mapping[int, torch.Tensor] | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.tracker = tracker
        self.evaluator = evaluator
        self.task_index = task_index
        self.beta = cfg.beta if beta is None else beta
        self.pro_f_weight = (
            cfg.pro_f_weight if prototype_weight_forget is None else prototype_weight_forget
        )
        self.prototypes = prototypes
        self.stats = EpochStats()
        self.batch = 0

        # Resolved once per task rather than per step, and read off the model
        # so the group table always matches the backbone that was built.
        self.group_names = discover_lora_groups(model, cfg.grouping)
        if not self.group_names and cfg.lora_rank > 0:
            raise RuntimeError(
                f"No LoRA parameters found on {type(model).__name__} despite "
                f"--lora_rank {cfg.lora_rank}."
            )

    # ------------------------------------------------------------------
    def alpha_for(self, epoch: int) -> float:
        """Structure-loss weight for this epoch."""
        if not self.cfg.warmup_alpha:
            return self.cfg.alpha
        return 0.0 if epoch < self.cfg.alpha_epoch else self.cfg.big_alpha

    def beta_for(self, epoch: int) -> float:
        if self.cfg.beta_decay and epoch >= self.cfg.beta_decay_epoch:
            return self.cfg.small_beta
        return self.beta

    # ------------------------------------------------------------------
    def _structure_loss(self, device: torch.device) -> torch.Tensor:
        groups = collect_group_params(self.model, self.group_names, require_grad=True)
        if not any(groups):
            # LoRA frozen (e.g. --lora_rank 0): no structure term to apply.
            return torch.zeros((), device=device)
        return group_sparse_loss(groups)

    def _prototype_terms(
        self,
        embeds_forget: torch.Tensor,
        labels_forget: torch.Tensor,
        embeds_remain: torch.Tensor,
        labels_remain: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(weighted_total, forget_term, remain_term)``."""
        if not self.cfg.prototype or self.prototypes is None:
            zero = torch.zeros((), device=device)
            return zero, zero, zero

        d_forget = prototype_loss(embeds_forget, labels_forget, self.prototypes)
        d_remain = prototype_loss(embeds_remain, labels_remain, self.prototypes)
        # push the forget embeddings *away* from their prototype, up to BND_pro
        forget_term = bounded_forget_loss(d_forget, self.cfg.bnd_pro)
        total = self.pro_f_weight * forget_term + self.cfg.pro_r_weight * d_remain
        return total, forget_term, d_remain

    # ------------------------------------------------------------------
    def train_one_epoch(self, pairs: PairedBatches, epoch: int) -> None:
        cfg = self.cfg
        device = cfg.device
        alpha = self.alpha_for(epoch)
        beta = self.beta_for(epoch)

        self.model.train()
        for (x_forget, y_forget), (x_remain, y_remain) in pairs:
            logits_forget, emb_forget = self.model(x_forget, y_forget)
            logits_remain, emb_remain = self.model(x_remain, y_remain)

            loss_remain = self.criterion(logits_remain, y_remain)
            ce_forget = self.criterion(logits_forget, y_forget)
            loss_forget = bounded_forget_loss(ce_forget, cfg.bnd)
            loss_structure = self._structure_loss(device)
            proto_total, proto_f, proto_r = self._prototype_terms(
                emb_forget, y_forget, emb_remain, y_remain, device
            )

            loss = beta * loss_forget + loss_remain + alpha * loss_structure + proto_total

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            n_f, n_r = x_forget.size(0), x_remain.size(0)
            self.stats.forget.update(beta * loss_forget.item(), n_f)
            self.stats.remain.update(loss_remain.item(), n_r)
            self.stats.structure.update(alpha * loss_structure.item(), n_r)
            self.stats.proto_forget.update(self.pro_f_weight * proto_f.item(), n_r)
            self.stats.proto_remain.update(cfg.pro_r_weight * proto_r.item(), n_r)
            self.stats.total.update(loss.item(), n_r)
            self.stats.acc_forget.update(topk_accuracy(logits_forget.detach(), y_forget), n_f)
            self.stats.acc_remain.update(topk_accuracy(logits_remain.detach(), y_remain), n_r)

            self.batch += 1
            self._maybe_report(epoch)
            self._maybe_evaluate(epoch)

    def _maybe_report(self, epoch: int) -> None:
        if self.cfg.disp_freq <= 0 or self.batch % self.cfg.disp_freq:
            return
        self.tracker.log(
            {
                **self.stats.as_metrics(f"/task{self.task_index}"),
                "alpha": self.alpha_for(epoch),
                "beta": self.beta_for(epoch),
                "epoch": epoch,
            },
            step=self.batch,
        )
        print(self.stats.format(self.task_index, epoch, self.batch))
        self.stats.reset()

    def _maybe_evaluate(self, epoch: int) -> None:
        if self.cfg.ver_freq <= 0 or self.batch % self.cfg.ver_freq:
            return
        self.evaluator.run(self.model, epoch=epoch, batch=self.batch)
        self.model.train()
