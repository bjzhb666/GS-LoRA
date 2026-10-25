"""Reproducibility helpers.

Everything that consumes randomness here is driven from a single ``--seed``:
the global python/numpy/torch state, the dataloader workers, the subset
sampling, and the class visiting order that decides which classes get
forgotten.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_seed", "seed_worker", "make_generator", "shuffled_class_order"]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch.

    Args:
        seed: the seed.
        deterministic: if True, disable cudnn autotuning and request
            deterministic kernels. Slower, but makes runs bit-reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` that also seeds numpy and python's ``random``.

    Seeds numpy and python's ``random`` as well as torch, so augmentation that
    uses them differs between workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def shuffled_class_order(num_classes: int, seed: int) -> list[int]:
    """The class visiting order that defines which classes get forgotten.

    Uses a private ``random.Random`` instance, so the order depends only on
    ``seed`` and ``num_classes`` -- not on how much unrelated code has drawn
    from the global RNG beforehand.
    """
    order = list(range(num_classes))
    random.Random(seed).shuffle(order)
    return order
