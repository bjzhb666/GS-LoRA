"""Class-wise splitting and subsampling.

:class:`ClassSubset` is a ``Subset`` whose ``targets``, ``classes`` and
``class_to_idx`` describe the subset itself and stay consistent with each
other, keyed by the dataset's global label ids. Subsets are index-based, so
splitting never re-walks the dataset directory.

:class:`TaskSplit` turns a shuffled class order plus a task index into the
four class bands one forgetting task needs.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset

__all__ = ["ClassSubset", "class_subset", "few_shot_subset", "ratio_subset", "TaskSplit"]


class ClassSubset(Subset):
    """A ``Subset`` that keeps ``targets``/``classes`` *consistent with itself*.

    ``targets[i]`` is the label of ``self[i]`` (global label ids, unchanged),
    and ``classes`` lists only the class names actually present.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__(dataset, list(indices))
        parent_targets = _targets_of(dataset)
        self.targets = [parent_targets[i] for i in self.indices]

        # Class names are keyed by *global* label id, not by position, so that
        # nesting a ClassSubset inside another one still resolves correctly.
        parent_names = _idx_to_class(dataset)
        present = sorted(set(self.targets))
        self.class_ids = present
        self.idx_to_class = {t: parent_names.get(t, str(t)) for t in present}
        self.classes = [self.idx_to_class[t] for t in present]
        self.class_to_idx = {name: t for t, name in self.idx_to_class.items()}

    @property
    def transform(self):  # convenience: reach through to the backing dataset
        return getattr(self.dataset, "transform", None)

    @transform.setter
    def transform(self, value) -> None:
        self.dataset.transform = value


def _idx_to_class(dataset: Dataset) -> dict[int, str]:
    """``{global label id: class name}`` for any of the dataset shapes we use."""
    mapping = getattr(dataset, "idx_to_class", None)
    if mapping is not None:
        return dict(mapping)
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if class_to_idx is not None:
        return {int(i): name for name, i in class_to_idx.items()}
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        return {i: name for i, name in enumerate(classes)}
    return {}


def _targets_of(dataset: Dataset) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError(
            f"{type(dataset).__name__} has no `targets`; class-wise splitting "
            f"needs the label of every sample without loading the images."
        )
    if isinstance(targets, torch.Tensor):
        return targets.tolist()
    return list(targets)


def class_subset(dataset: Dataset, class_ids: Sequence[int]) -> ClassSubset:
    """All samples whose label is in ``class_ids`` (no directory re-scan)."""
    wanted = set(int(c) for c in class_ids)
    targets = _targets_of(dataset)
    indices = [i for i, t in enumerate(targets) if int(t) in wanted]
    if not indices:
        raise ValueError(
            f"No samples found for class ids {sorted(wanted)[:8]}... "
            f"(dataset has {len(set(targets))} distinct labels)"
        )
    return ClassSubset(dataset, indices)


def few_shot_subset(
    dataset: Dataset, n_shot: int, seed: int | None = None, strict: bool = False
) -> ClassSubset:
    """``n_shot`` samples per class.

    Classes with fewer than ``n_shot`` samples are taken in full and reported;
    pass ``strict=True`` to raise instead.
    """
    if n_shot <= 0:
        raise ValueError("n_shot must be positive")
    rng = random.Random(seed)
    targets = _targets_of(dataset)

    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[int(label)].append(idx)

    selected: list[int] = []
    short: list[tuple[int, int]] = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        if len(pool) < n_shot:
            short.append((cls, len(pool)))
            if strict:
                raise ValueError(f"Class {cls} has {len(pool)} samples < n_shot={n_shot}")
            selected.extend(pool)
        else:
            selected.extend(rng.sample(pool, n_shot))
    if short:
        print(
            f"[few_shot_subset] {len(short)} class(es) had fewer than {n_shot} "
            f"samples and were taken in full, e.g. {short[:3]}"
        )
    rng.shuffle(selected)
    return ClassSubset(dataset, selected)


def ratio_subset(
    dataset: Dataset, ratio: float, seed: int | None = None, min_samples: int = 1
) -> ClassSubset:
    """A random ``ratio`` fraction of the dataset.

    Drawn from a dedicated generator, so the subset depends only on ``seed``.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"data_ratio must be in (0, 1], got {ratio}")
    n = len(dataset)  # type: ignore[arg-type]
    size = max(min_samples, int(n * ratio))
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    indices = torch.randperm(n, generator=generator)[:size].tolist()
    return ClassSubset(dataset, indices)


@dataclass
class TaskSplit:
    """The class-id ranges for one continual-forgetting task.

    Layout over the (shuffled) class order, for task ``t``::

        [ 0 .. open )[ open ....... known_end )[ .. +per_forget )[ .. N )
          open        remain (replayed)          forget            already forgotten

    where ``known_end = num_of_first_cls - t * per_forget_cls``.

    ``open`` is the missing-class setting from the T-PAMI paper: classes that
    must be *retained* but whose data is unavailable for replay. It is a fixed
    prefix of the class order, so a class that is missing stays missing for
    every task. With ``open_cls_num == 0`` the band is empty and this reduces
    to the plain two-way split.

    The open band is a fixed prefix so that a class which is missing stays
    missing for every task.
    """

    remain: list[int]
    forget: list[int]
    open: list[int]
    old: list[int]

    @classmethod
    def build(
        cls,
        class_order: Sequence[int],
        task_index: int,
        num_of_first_cls: int,
        per_forget_cls: int,
        open_cls_num: int = 0,
    ) -> "TaskSplit":
        n = len(class_order)
        # boundary of "everything still known" for this task
        known_end = num_of_first_cls - task_index * per_forget_cls
        forget_end = known_end + per_forget_cls
        if known_end <= open_cls_num:
            raise ValueError(
                f"Task {task_index}: nothing left to retain "
                f"(known_end={known_end}, open_cls_num={open_cls_num})"
            )
        if forget_end > n:
            raise ValueError(
                f"Task {task_index}: forget range ends at {forget_end} but only "
                f"{n} classes exist"
            )
        return cls(
            open=list(class_order[0:open_cls_num]),
            remain=list(class_order[open_cls_num:known_end]),
            forget=list(class_order[known_end:forget_end]),
            old=list(class_order[forget_end:n]),
        )

    def __post_init__(self) -> None:
        bands = {"remain": self.remain, "forget": self.forget, "open": self.open, "old": self.old}
        seen: dict[int, str] = {}
        for name, ids in bands.items():
            for c in ids:
                if c in seen:
                    raise ValueError(
                        f"class {c} appears in both {seen[c]!r} and {name!r}"
                    )
                seen[c] = name
