"""Iterate two dataloaders of different lengths in lockstep.

GS-LoRA consumes a forget batch and a remain batch per step. The two loaders
have very different lengths (``--data_ratio 0.1`` over 90 vs 10 classes, or
few-shot where the forget set can be *larger*), so one has to be cycled.

``PairedBatches`` always iterates the *longer* loader once per epoch and
cycles the shorter one, so neither side is truncated regardless of which is
bigger, and the same loop body serves every configuration. It is device
agnostic, so training runs on CPU as well as GPU.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import DataLoader

__all__ = ["PairedBatches", "cycle"]


def cycle(loader: DataLoader) -> Iterator:
    """Endlessly re-iterate a dataloader."""
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch
        if empty:
            raise ValueError("cycle() received an empty DataLoader")


class PairedBatches:
    """Yield ``(forget_batch, remain_batch)``, cycling the shorter loader.

    One epoch == one full pass over the **longer** of the two loaders, so no
    samples are dropped from either side regardless of which is bigger.
    """

    def __init__(
        self,
        forget_loader: DataLoader,
        remain_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.forget_loader = forget_loader
        self.remain_loader = remain_loader
        self.device = device
        self.n_forget = len(forget_loader)
        self.n_remain = len(remain_loader)
        if self.n_forget == 0 or self.n_remain == 0:
            raise ValueError(
                f"Both loaders must be non-empty "
                f"(forget={self.n_forget}, remain={self.n_remain} batches)"
            )
        self.steps = max(self.n_forget, self.n_remain)

    def __len__(self) -> int:
        return self.steps

    def _to_device(self, batch):
        images, labels = batch
        return (
            images.to(self.device, non_blocking=True).float(),
            labels.to(self.device, non_blocking=True).long(),
        )

    def __iter__(self):
        forget_it = (
            iter(self.forget_loader) if self.n_forget >= self.n_remain else cycle(self.forget_loader)
        )
        remain_it = (
            iter(self.remain_loader) if self.n_remain >= self.n_forget else cycle(self.remain_loader)
        )
        for _ in range(self.steps):
            yield self._to_device(next(forget_it)), self._to_device(next(remain_it))
