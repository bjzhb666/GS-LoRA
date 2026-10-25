"""Loss terms used by GS-LoRA and the baselines.

* :func:`group_sparse_loss`      -- the group-lasso structure term L_structure
* :func:`prototype_loss`         -- the prototype regulariser L_pro (GS-LoRA++)
* :func:`bounded_forget_loss`    -- the bounded selective-forgetting term
* :class:`DistillKL`,
  :func:`attention_transfer_loss`,
  :func:`elastic_reg_loss`       -- terms shared by the baselines
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "group_sparse_loss",
    "prototype_loss",
    "bounded_forget_loss",
    "DistillKL",
    "attention_transfer_loss",
    "elastic_reg_loss",
]

# Smoothing inside the sqrt of the group norm; see group_sparse_loss.
_GROUP_EPS = 1e-12


def group_sparse_loss(
    group_params: Sequence[Sequence[torch.Tensor]],
    eps: float = _GROUP_EPS,
) -> torch.Tensor:
    r"""Group-lasso penalty :math:`\sum_g \sqrt{\lVert \theta_g \rVert_2^2 + \epsilon}`.

    This is :math:`\mathcal{L}_{structure}`: summing the *un-squared* L2 norm
    of each group drives whole groups to zero, which is what selects the
    handful of LoRA groups the method actually modifies.

    Args:
        group_params: one list of tensors per group.
        eps: smoothing inside the square root. :math:`\sqrt{\cdot}` has an
            infinite derivative at the origin, and the origin is exactly where
            a successfully sparsified group ends up, so the gradient there
            would otherwise be ``NaN``. ``eps=1e-12`` shifts the value by only
            ``1e-6`` while keeping the gradient finite.
    """
    if not group_params:
        raise ValueError("group_sparse_loss received no groups")

    total: torch.Tensor | None = None
    for group in group_params:
        if not group:
            continue
        sq = sum(torch.sum(p**2) for p in group)
        norm = torch.sqrt(sq + eps)
        total = norm if total is None else total + norm

    if total is None:
        raise ValueError(
            "group_sparse_loss: every group was empty. The LoRA parameters are "
            "probably frozen or the group names do not match the backbone."
        )
    return total


def prototype_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: Mapping[int, torch.Tensor],
    distance: str = "kl",
) -> torch.Tensor:
    """Distance between each sample's embedding and its class prototype.

    Args:
        embeddings: ``(B, D)`` features produced by the current model.
        labels: ``(B,)`` class ids.
        prototypes: ``{class_id: (D,) tensor}`` computed from the frozen
            pre-trained model.
        distance: ``"kl"`` (default, as in the paper) or ``"l2"``.

    Raises:
        KeyError: if any label in the batch has no prototype.
        ValueError: if ``distance`` is not one of the supported metrics.
    """
    missing = {int(l) for l in labels.unique().tolist()} - set(prototypes)
    if missing:
        raise KeyError(
            f"No prototype for class ids {sorted(missing)}. Prototypes must be "
            f"computed over the union of the forget and remain splits."
        )

    target = torch.stack([prototypes[int(l)] for l in labels]).to(
        device=embeddings.device, dtype=embeddings.dtype
    )

    if distance == "l2":
        return torch.mean((embeddings - target) ** 2)
    if distance == "kl":
        return F.kl_div(
            F.log_softmax(embeddings, dim=1),
            F.log_softmax(target, dim=1),
            reduction="batchmean",
            log_target=True,
        )
    raise ValueError(f"distance must be 'kl' or 'l2', got {distance!r}")


def bounded_forget_loss(ce_loss: torch.Tensor, bnd: float) -> torch.Tensor:
    """``relu(BND - CE)``: push the forget CE up, but only as far as ``BND``."""
    return F.relu(bnd - ce_loss)


class DistillKL(nn.Module):
    """Temperature-scaled KL divergence, :math:`T^2 \cdot KL(p_t \Vert p_s)`."""

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        log_p_s = F.log_softmax(student_logits / t, dim=1)
        p_t = F.softmax(teacher_logits / t, dim=1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)


def attention_transfer_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """Attention-transfer loss (Zagoruyko & Komodakis, 2017), used by LIRF.

    Channel-pooled, L2-normalised spatial attention maps, with small responses
    masked out. The mask is applied out-of-place so the autograd graph stays
    valid.
    """

    def attention(x: torch.Tensor) -> torch.Tensor:
        att = F.normalize(x.pow(2).mean(dim=1).flatten(1), dim=1)
        return att * (att >= 0.005)

    return (attention(student_feat) - attention(teacher_feat)).pow(2).mean()


def elastic_reg_loss(
    model: nn.Module,
    regularization_terms: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]] | None,
    reg_lambda: float,
    device: torch.device,
) -> torch.Tensor:
    """Quadratic penalty shared by EWC / MAS / L2 (``importance * (θ - θ*)²``)."""
    zero = torch.zeros((), device=device)
    if not regularization_terms or reg_lambda == 0:
        return zero

    params = dict(model.named_parameters())
    total = zero
    for term in regularization_terms.values():
        importance = term["importance"]
        anchor = term["task_param"]
        for name, imp in importance.items():
            param = params.get(name)
            if param is None or not param.requires_grad:
                continue
            total = total + (imp * (param - anchor[name]) ** 2).sum()
    return reg_lambda * total
