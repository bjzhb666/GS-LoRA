"""Turn a :class:`~gslora.config.Config` and a task index into dataloaders.

:func:`load_image_folders` scans the dataset once per process;
:func:`build_task_data` then slices it per task with index-based subsets, so
the directory tree is never re-walked inside the task loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder

from ..config import Config
from ..seeding import make_generator, seed_worker, shuffled_class_order
from .splits import ClassSubset, TaskSplit, class_subset, few_shot_subset, ratio_subset
from .wrappers import RandomLabelWrapper

__all__ = ["build_transform", "load_image_folders", "TaskData", "build_task_data"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(cfg: Config, train: bool = False) -> T.Compose:
    """Input pipeline.

    Train and test share the same transform (the face pipeline is just
    ``ToTensor``); ``train`` is accepted so augmentation can be added later
    without touching call sites.
    """
    if cfg.normalize_input:  # ImageNet-100
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose([T.ToTensor()])


@lru_cache(maxsize=4)
def _image_folder(root: str, _transform_key: int) -> ImageFolder:
    return ImageFolder(root=root)


def load_image_folders(cfg: Config) -> tuple[ImageFolder, ImageFolder]:
    """``(train, test)`` ``ImageFolder``s, scanned once per process."""
    root = Path(cfg.data_root)
    train_dir, test_dir = root / "train", root / "test"
    for d in (train_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Expected {d} to exist. --data_mode {cfg.data_mode} resolves to "
                f"{root}; see the README for the required layout."
            )
    transform = build_transform(cfg)
    train_ds = ImageFolder(root=str(train_dir), transform=transform)
    test_ds = ImageFolder(root=str(test_dir), transform=transform)
    if len(train_ds.classes) != cfg.num_classes:
        raise ValueError(
            f"{train_dir} has {len(train_ds.classes)} classes but --data_mode "
            f"{cfg.data_mode} declares {cfg.num_classes}"
        )
    return train_ds, test_ds


@dataclass
class TaskData:
    """Everything one forgetting task needs."""

    split: TaskSplit
    # training subsets (already subsampled by --data_ratio / --few_shot)
    forget_train: Dataset
    remain_train: Dataset
    # loaders
    forget_loader: DataLoader
    remain_loader: DataLoader
    forget_train_eval_loader: DataLoader
    remain_train_eval_loader: DataLoader
    forget_test_loader: DataLoader
    remain_test_loader: DataLoader
    open_test_loader: DataLoader | None = None
    old_test_loader: DataLoader | None = None
    replay_loader: DataLoader | None = None

    def describe(self) -> str:
        parts = [
            f"remain={len(self.split.remain)}cls",
            f"forget={len(self.split.forget)}cls",
        ]
        if self.split.open:
            parts.append(f"open={len(self.split.open)}cls")
        if self.split.old:
            parts.append(f"old={len(self.split.old)}cls")
        parts.append(f"| {len(self.forget_train)} forget / {len(self.remain_train)} remain samples")
        return "  ".join(parts)


def _loader(
    dataset: Dataset,
    cfg: Config,
    *,
    shuffle: bool,
    batch_size: int | None = None,
    seed_offset: int = 0,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=cfg.device.type == "cuda",
    )
    if shuffle:
        # only meaningful when shuffling
        kwargs["worker_init_fn"] = seed_worker
        kwargs["generator"] = make_generator(cfg.seed + seed_offset)
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def _subsample(dataset: Dataset, cfg: Config, seed_offset: int) -> ClassSubset:
    if cfg.few_shot:
        return few_shot_subset(dataset, cfg.few_shot_num, seed=cfg.seed + seed_offset)
    return ratio_subset(dataset, cfg.data_ratio, seed=cfg.seed + seed_offset)


def build_task_data(
    cfg: Config,
    task_index: int,
    train_ds: ImageFolder,
    test_ds: ImageFolder,
    class_order: list[int] | None = None,
) -> TaskData:
    """Assemble the splits and loaders for one task."""
    if class_order is None:
        class_order = shuffled_class_order(cfg.num_classes, cfg.seed)

    split = TaskSplit.build(
        class_order,
        task_index,
        num_of_first_cls=cfg.num_of_first_cls,
        per_forget_cls=cfg.per_forget_cls,
        open_cls_num=cfg.open_cls_num,
    )

    forget_train_full = class_subset(train_ds, split.forget)
    remain_train_full = class_subset(train_ds, split.remain)
    forget_train = _subsample(forget_train_full, cfg, seed_offset=100 + task_index)
    remain_train = _subsample(remain_train_full, cfg, seed_offset=200 + task_index)

    forget_test = class_subset(test_ds, split.forget)
    remain_test = class_subset(test_ds, split.remain)

    # Baselines train on the forget set with scrambled labels. Applied to the
    # fitting copy only, so the reported forget-train accuracy is still
    # measured against the true labels.
    forget_train_for_fit: Dataset = forget_train
    if cfg.method in ("l2", "ewc", "mas", "lwf", "der", "fdr"):
        forget_train_for_fit = RandomLabelWrapper(
            forget_train, num_classes=cfg.num_classes, seed=cfg.seed + task_index
        )

    eval_bs = cfg.batch_size * 5
    data = TaskData(
        split=split,
        forget_train=forget_train,
        remain_train=remain_train,
        forget_loader=_loader(forget_train_for_fit, cfg, shuffle=True, seed_offset=1),
        remain_loader=_loader(remain_train, cfg, shuffle=True, seed_offset=2),
        forget_train_eval_loader=_loader(forget_train, cfg, shuffle=False, batch_size=eval_bs),
        remain_train_eval_loader=_loader(remain_train, cfg, shuffle=False, batch_size=eval_bs),
        forget_test_loader=_loader(forget_test, cfg, shuffle=False, batch_size=eval_bs),
        remain_test_loader=_loader(remain_test, cfg, shuffle=False, batch_size=eval_bs),
    )

    if split.open:
        data.open_test_loader = _loader(
            class_subset(test_ds, split.open), cfg, shuffle=False, batch_size=eval_bs
        )
    if split.old:
        data.old_test_loader = _loader(
            class_subset(test_ds, split.old), cfg, shuffle=False, batch_size=eval_bs
        )
    if cfg.replay:
        data.replay_loader = _loader(
            ConcatDataset([forget_train_for_fit, remain_train]),
            cfg,
            shuffle=True,
            seed_offset=3,
        )
    return data


def build_importance_loader(
    cfg: Config, train_ds: ImageFolder, class_order: list[int], task_index: int
) -> DataLoader:
    """Remain set of task ``task_index``, used to estimate EWC/MAS importance."""
    split = TaskSplit.build(
        class_order,
        task_index,
        num_of_first_cls=cfg.num_of_first_cls,
        per_forget_cls=cfg.per_forget_cls,
        open_cls_num=cfg.open_cls_num,
    )
    full = class_subset(train_ds, split.remain)
    subset = _subsample(full, cfg, seed_offset=300 + task_index)
    return _loader(subset, cfg, shuffle=True, seed_offset=4)
