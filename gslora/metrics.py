"""Small metric helpers shared by every training loop."""

from __future__ import annotations

import torch

__all__ = ["AverageMeter", "topk_accuracy", "harmonic_mean"]


class AverageMeter:
    """Tracks the latest value and the running average of a scalar."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AverageMeter(val={self.val:.4f}, avg={self.avg:.4f}, count={self.count})"


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: int = 1) -> float:
    """Top-k accuracy in percent. Returns 0.0 for an empty batch."""
    if target.numel() == 0:
        return 0.0
    batch_size = target.size(0)
    k = min(topk, output.size(1))
    _, pred = output.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().sum().item() * 100.0 / batch_size


def harmonic_mean(forget_drop: float, remain_acc: float, eps: float = 1e-8) -> float:
    """H-mean of the accuracy drop on the forget set and the retained accuracy.

    ``engine.py`` used ``2*a*b/(a+b)`` with no epsilon, which raises
    ``ZeroDivisionError`` on the very common case where the model has not moved
    yet (``forget_drop == 0`` and ``remain_acc == 0``). ``engine_cl.py`` had the
    epsilon. Both call sites now share this one definition.

    A negative ``forget_drop`` (the forget accuracy went *up*) is clamped to 0 so
    the score stays monotone instead of producing a large positive H-mean from
    two negatives.
    """
    forget_drop = max(forget_drop, 0.0)
    remain_acc = max(remain_acc, 0.0)
    denom = forget_drop + remain_acc
    if denom <= eps:
        return 0.0
    return 2.0 * forget_drop * remain_acc / denom
