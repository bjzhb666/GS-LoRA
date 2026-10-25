"""Exponential moving average of the trained weights (``--average_weight``).

Averaging is done over the ``state_dict`` *by key*, so parameters and buffers
are both covered, and a structural mismatch raises instead of being
truncated.

The snapshot is taken with the model in ``eval()`` mode, which is load-bearing:
loralib folds the LoRA deltas into the base weights on ``eval()`` and unfolds
them on ``train()``, so this averages the **merged** weights -- the quantity
that is actually deployed. The live model's mode is restored afterwards.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .checkpoint import unwrap_model

__all__ = ["ModelEma"]


class ModelEma:
    """Keeps an EMA copy of ``model``'s merged weights."""

    def __init__(self, model: nn.Module, decay: float = 0.99, device: torch.device | None = None) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.module = copy.deepcopy(unwrap_model(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)
        self._initialised = False

    @torch.no_grad()
    def _merged_state(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """``state_dict`` with LoRA deltas folded into the base weights.

        loralib folds on ``eval()`` and unfolds on ``train()``; the previous
        mode is restored so the live model is untouched.
        """
        inner = unwrap_model(model)
        was_training = inner.training
        inner.eval()
        try:
            return {k: v.detach().clone() for k, v in inner.state_dict().items()}
        finally:
            inner.train(was_training)

    @torch.no_grad()
    def set(self, model: nn.Module) -> None:
        """Hard-copy the current weights (used at ``--ema_epoch``)."""
        self.module.load_state_dict(self._merged_state(model))
        self._initialised = True

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if not self._initialised:
            self.set(model)
            return
        source = self._merged_state(model)
        target = self.module.state_dict()
        for key, value in target.items():
            new = source[key]
            if not value.is_floating_point():
                value.copy_(new)  # e.g. num_batches_tracked
            else:
                value.mul_(self.decay).add_(new.to(value.device), alpha=1.0 - self.decay)

    def step(self, model: nn.Module, epoch: int, start_epoch: int) -> bool:
        """Call once per epoch. Returns True if the EMA is live."""
        if epoch < start_epoch:
            return False
        if epoch == start_epoch:
            self.set(model)
        else:
            self.update(model)
        return True

    def eval(self) -> nn.Module:
        return self.module.eval()
