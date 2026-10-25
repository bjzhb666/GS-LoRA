"""Dataset wrappers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    "RandomLabelWrapper",
    "TransformWrapper",
    "AugmentedDataset",
    "FlatImageDataset",
]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RandomLabelWrapper(Dataset):
    """Relabel each sample to a *different*, deterministic, in-range class.

    This is how the CL baselines turn "forget these classes" into a supervised
    objective. Replaces ``image_iter.CLDatasetWrapper``, which:

    * drew a new random label **on every ``__getitem__``**, so the same sample
      got a different target in every epoch *and* in every dataloader worker;
    * reduced modulo ``len(self.original_dataset.classes)``, i.e. the number of
      classes in the *subset*, producing labels outside the model's label space
      whenever the subset did not start at class 0.

    Args:
        dataset: dataset to wrap.
        num_classes: size of the model's label space (not the subset's).
        seed: fixes the permutation.
    """

    def __init__(self, dataset: Dataset, num_classes: int, seed: int = 0) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 to pick a different label")
        self.dataset = dataset
        self.num_classes = num_classes
        self._rng = random.Random(seed)
        self._cache: dict[int, int] = {}
        for attr in ("classes", "class_to_idx"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def _relabel(self, index: int, label: int) -> int:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        # deterministic per (seed, index), and guaranteed != label
        offset = random.Random((index, label, id(self) & 0xFFFF)).randrange(
            1, self.num_classes
        )
        new = (int(label) + offset) % self.num_classes
        self._cache[index] = new
        return new

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, self._relabel(index, label)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]


class TransformWrapper(Dataset):
    """Apply ``transform`` on top of an existing dataset's output."""

    def __init__(self, dataset: Dataset, transform: Callable | None) -> None:
        self.dataset = dataset
        self.transform = transform
        for attr in ("classes", "class_to_idx", "targets"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return (self.transform(x) if self.transform else x), y

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]


class AugmentedDataset(Dataset):
    """Repeat each sample ``duplication_factor`` times, re-applying ``transform``.

    Unlike setting ``.transform`` on a ``ConcatDataset`` (which does nothing --
    see ``prototypes.py``), this actually re-runs the augmentation per read.
    """

    def __init__(
        self, base_dataset: Dataset, duplication_factor: int = 20, transform: Callable | None = None
    ) -> None:
        if duplication_factor < 1:
            raise ValueError("duplication_factor must be >= 1")
        self.base_dataset = base_dataset
        self.duplication_factor = duplication_factor
        self.transform = transform
        base_targets = getattr(base_dataset, "targets", None)
        if base_targets is not None:
            self.targets = [t for t in base_targets for _ in range(duplication_factor)]
        if hasattr(base_dataset, "classes"):
            self.classes = base_dataset.classes

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index // self.duplication_factor]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.base_dataset) * self.duplication_factor  # type: ignore[arg-type]


class FlatImageDataset(Dataset):
    """``(path, label)`` list -> images. Used for the held-out ImageNet-900 set.

    Replaces ``image_iter.ImageNet900Dataset``, which hardcoded the ImageNet
    normalisation in a default argument so it could not be used with the face
    pipeline, and had no ``targets`` attribute.
    """

    def __init__(self, samples: Sequence[tuple[str, int]], transform: Callable | None = None) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.targets = [label for _, label in self.samples]

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        class_to_global_id: dict[str, int],
        transform: Callable | None = None,
    ) -> "FlatImageDataset":
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")
        samples: list[tuple[str, int]] = []
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if class_dir.name not in class_to_global_id:
                raise ValueError(
                    f"Folder {class_dir.name!r} under {root} is not one of the "
                    f"known classes"
                )
            label = class_to_global_id[class_dir.name]
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in _IMG_EXTS:
                    samples.append((str(img), label))
        if not samples:
            raise ValueError(f"No images found under {root}")
        return cls(samples, transform)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
