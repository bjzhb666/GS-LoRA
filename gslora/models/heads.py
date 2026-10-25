"""Margin-based classification heads for the face backbones.

``--head`` selects one of :class:`Softmax`, :class:`ArcFace`,
:class:`CosFace` (the default used throughout the paper) or
:class:`SFaceLoss`. All four take ``(embedding, label)`` and return logits,
and all four are device agnostic -- multi-GPU is handled by wrapping the
backbone in ``DataParallel``, which replicates the head, so ``device_id`` is
accepted only for signature compatibility and should be left as ``None``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

__all__ = ["Softmax", "ArcFace", "CosFace", "SFaceLoss", "build_head", "HEADS"]


class Softmax(nn.Module):
    """Plain linear classification head (kept for ``--head Softmax``)."""

    def __init__(self, in_features: int, out_features: int, device_id=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features))
        self.bias = Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class ArcFace(nn.Module):
    """ArcFace: additive angular margin (https://arxiv.org/abs/1801.07698)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device_id=None,
        s: float = 64.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # clamp: cosine can exceed 1 by ~1e-7 and make the sqrt NaN
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_(min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.to(cosine.device).view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"s={self.s}, m={self.m}"
        )


class CosFace(nn.Module):
    """CosFace: large-margin cosine loss (https://arxiv.org/abs/1801.09414)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device_id=None,
        s: float = 64.0,
        m: float = 0.35,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.to(cosine.device).view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"s={self.s}, m={self.m}"
        )


class SFaceLoss(nn.Module):
    """SFace head.

    ``forward`` returns logits, keeping the head interchangeable with the
    other three; :meth:`sface_loss` computes the intra/inter terms of the
    SFace objective.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device_id=None,
        s: float = 64.0,
        k: float = 80.0,
        a: float = 0.80,
        b: float = 1.22,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s, self.k, self.a, self.b = s, k, a, b
        self.weight = Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight, gain=2)

    def _cosine(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(input), F.normalize(self.weight))

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self._cosine(input) * self.s

    def sface_loss(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        output = self.forward(input, label)
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, label.to(output.device).view(-1, 1).long(), 1)
        zero_hot = 1.0 - one_hot

        wyi_x = torch.sum(one_hot * output, dim=1)
        with torch.no_grad():
            theta_yi = torch.acos((wyi_x / self.s).clamp(-1.0, 1.0))
            weight_yi = 1.0 / (1.0 + torch.exp(-self.k * (theta_yi - self.a)))
        intra = (-weight_yi * wyi_x).mean()

        wj = zero_hot * output
        with torch.no_grad():
            theta_j = torch.acos((wj / self.s).clamp(-1.0, 1.0))
            weight_j = 1.0 / (1.0 + torch.exp(self.k * (theta_j - self.b)))
        inter = torch.sum(weight_j * wj, dim=1).mean()
        return intra + inter


HEADS = {
    "Softmax": Softmax,
    "ArcFace": ArcFace,
    "CosFace": CosFace,
    "SFaceLoss": SFaceLoss,
    "SFace": SFaceLoss,  # the old config accepted both spellings
}


def build_head(loss_type: str, in_features: int, out_features: int, device_id=None) -> nn.Module | None:
    """Return the head named by ``loss_type``, or ``None`` for ``"None"``.

    Raises on an unrecognised name rather than leaving the backbone without a
    classifier.
    """
    if loss_type == "None":
        return None
    try:
        cls = HEADS[loss_type]
    except KeyError:
        raise ValueError(
            f"Unknown head {loss_type!r}; expected one of {sorted(set(HEADS))}"
        ) from None
    return cls(in_features=in_features, out_features=out_features, device_id=device_id)
