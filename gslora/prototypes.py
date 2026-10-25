"""Class prototypes for the GS-LoRA++ prototype regulariser.

A prototype is the mean embedding of a class under the frozen pre-trained
model. :math:`\mathcal{L}_{pro}` then pushes forgotten classes away from their
prototype and pulls retained classes towards theirs.

``--proto_aug_num`` turns on RandAugment and averages over
``--proto_aug_repeats`` augmented copies of each sample, which helps when only
a few shots per class are available. The augmentation is applied through a
wrapper dataset, so the caller's dataset is never mutated.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Mapping

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from .data.wrappers import AugmentedDataset

__all__ = ["compute_prototypes", "eval_mode"]


@contextlib.contextmanager
def eval_mode(model: nn.Module):
    """Temporarily switch to eval, then restore the exact previous mode.

    Matters for loralib: ``train()``/``eval()`` merge and un-merge the LoRA
    deltas into the frozen weights, so leaking the mode change corrupts the
    next optimisation step.
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


@torch.no_grad()
def compute_prototypes(
    backbone: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
    aug_magnitude: int = 0,
    aug_repeats: int = 20,
    aug_ops: int = 2,
) -> dict[int, torch.Tensor]:
    """Mean embedding per class, computed from the frozen pre-trained model.

    Args:
        aug_magnitude: RandAugment magnitude. ``0`` disables augmentation.
        aug_repeats: how many augmented copies of each sample to average over.

    Returns:
        ``{class_id: (D,) cpu tensor}``.
    """
    if aug_magnitude > 0:
        # Datasets here already emit float tensors, so RandAugment has to run
        # on uint8 tensors: convert, augment, convert back.
        augment = T.Compose(
            [
                T.Lambda(lambda t: (t * 255).clamp(0, 255).to(torch.uint8)),
                T.RandAugment(num_ops=aug_ops, magnitude=aug_magnitude),
                T.Lambda(lambda t: t.float() / 255.0),
            ]
        )
        dataset = AugmentedDataset(dataset, duplication_factor=aug_repeats, transform=augment)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    sums: dict[int, torch.Tensor] = {}
    counts: defaultdict[int, int] = defaultdict(int)

    with eval_mode(backbone):
        backbone.to(device)
        for images, labels in loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            _, embeddings = backbone(images, labels)
            for emb, label in zip(embeddings, labels):
                key = int(label)
                if key in sums:
                    sums[key] += emb
                else:
                    sums[key] = emb.clone()
                counts[key] += 1

    if not sums:
        raise ValueError("compute_prototypes got an empty dataset")
    return {k: (v / counts[k]).detach().cpu() for k, v in sums.items()}


def check_prototype_coverage(
    prototypes: Mapping[int, torch.Tensor], required: set[int]
) -> None:
    missing = required - set(prototypes)
    if missing:
        raise KeyError(
            f"Prototypes missing for classes {sorted(missing)[:8]}. They must be "
            f"computed over the union of the forget and remain training splits."
        )
