"""Dataset splitting and subsampling."""

from __future__ import annotations

import pytest
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from gslora.data.splits import (
    ClassSubset,
    TaskSplit,
    class_subset,
    few_shot_subset,
    ratio_subset,
)
from gslora.data.wrappers import AugmentedDataset, RandomLabelWrapper


@pytest.fixture
def train_folder(tiny_root) -> ImageFolder:
    return ImageFolder(root=f"{tiny_root}/train", transform=ToTensor())


# ----------------------------------------------------------------------
# ClassSubset
# ----------------------------------------------------------------------
def test_subset_targets_length_matches_subset(train_folder):
    """`targets` must describe the subset, not the dataset it came from."""
    sub = ClassSubset(train_folder, [0, 5, 11])
    assert len(sub) == 3
    assert len(sub.targets) == 3


def test_subset_targets_match_actual_items(train_folder):
    sub = ClassSubset(train_folder, [0, 7, 13, 20])
    for i in range(len(sub)):
        _, label = sub[i]
        assert sub.targets[i] == label


def test_subset_class_to_idx_is_consistent_with_targets(train_folder):
    """`classes`, `class_to_idx` and `targets` must agree with each other."""
    sub = class_subset(train_folder, [2, 5])
    assert set(sub.class_to_idx.values()) == set(sub.targets)
    for name, idx in sub.class_to_idx.items():
        assert train_folder.classes[idx] == name


# ----------------------------------------------------------------------
# class_subset
# ----------------------------------------------------------------------
def test_class_subset_selects_exactly_those_classes(train_folder):
    sub = class_subset(train_folder, [1, 4, 9])
    assert set(sub.targets) == {1, 4, 9}
    assert len(sub) == 3 * 6  # 6 images per class in the fixture


def test_class_subset_is_disjoint(train_folder):
    a = class_subset(train_folder, [0, 1, 2])
    b = class_subset(train_folder, [3, 4])
    assert not (set(a.indices) & set(b.indices))
    assert len(a) + len(b) == 5 * 6


def test_class_subset_raises_for_unknown_class(train_folder):
    with pytest.raises(ValueError, match="No samples"):
        class_subset(train_folder, [999])


# ----------------------------------------------------------------------
# few-shot / ratio
# ----------------------------------------------------------------------
def test_few_shot_takes_n_per_class(train_folder):
    sub = few_shot_subset(train_folder, n_shot=2, seed=0)
    counts: dict[int, int] = {}
    for t in sub.targets:
        counts[t] = counts.get(t, 0) + 1
    assert set(counts) == set(range(10))
    assert all(c == 2 for c in counts.values())
    assert len(sub) == 20


def test_few_shot_is_deterministic(train_folder):
    a = few_shot_subset(train_folder, 3, seed=42)
    b = few_shot_subset(train_folder, 3, seed=42)
    assert a.indices == b.indices
    c = few_shot_subset(train_folder, 3, seed=43)
    assert a.indices != c.indices


def test_few_shot_tolerates_sparse_classes(train_folder, capsys):
    """A class with fewer than n_shot samples is taken in full, not fatal."""
    sub = few_shot_subset(train_folder, n_shot=100, seed=0)
    assert len(sub) == len(train_folder)   # everything taken
    assert "fewer than" in capsys.readouterr().out
    with pytest.raises(ValueError):
        few_shot_subset(train_folder, n_shot=100, seed=0, strict=True)


def test_ratio_subset_size_and_determinism(train_folder):
    a = ratio_subset(train_folder, 0.5, seed=7)
    assert len(a) == 30
    assert a.indices == ratio_subset(train_folder, 0.5, seed=7).indices


def test_ratio_subset_rejects_bad_ratio(train_folder):
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ratio_subset(train_folder, bad)


# ----------------------------------------------------------------------
# TaskSplit
# ----------------------------------------------------------------------
def test_task_split_is_a_partition():
    order = list(range(100))
    split = TaskSplit.build(order, 0, num_of_first_cls=90, per_forget_cls=10)
    assert len(split.remain) == 90
    assert len(split.forget) == 10
    assert split.open == []
    assert split.old == []
    assert sorted(split.remain + split.forget) == order


def test_task_split_advances_across_tasks():
    order = list(range(100))
    seen_forgotten: set[int] = set()
    for task in range(9):
        s = TaskSplit.build(order, task, 90, 10)
        assert len(s.forget) == 10
        assert not (set(s.forget) & seen_forgotten), "a class is forgotten twice"
        assert set(s.old) == seen_forgotten
        seen_forgotten |= set(s.forget)
        assert not (set(s.remain) & seen_forgotten)
    assert len(seen_forgotten) == 90


def test_task_split_open_band():
    order = list(range(100))
    s = TaskSplit.build(order, 0, num_of_first_cls=90, per_forget_cls=10, open_cls_num=10)
    assert len(s.remain) == 80
    assert len(s.open) == 10
    assert len(s.forget) == 10
    assert sorted(s.remain + s.open + s.forget) == order
    # the open band is the fixed *prefix* of the class order
    assert s.open == order[:10]
    assert s.remain == order[10:90]


def test_open_classes_stay_open_across_tasks():
    """A class that is missing must stay missing for every task."""
    order = list(range(100))
    first = None
    for task in range(4):
        s = TaskSplit.build(order, task, num_of_first_cls=90,
                            per_forget_cls=10, open_cls_num=10)
        if first is None:
            first = s.open
        assert s.open == first, f"task {task}: open band moved to {s.open[:3]}..."
        assert not (set(s.open) & set(s.remain)), "an open class became replayable"
        assert not (set(s.open) & set(s.forget))


def test_task_split_respects_shuffled_order():
    order = [7, 3, 9, 1, 5, 0, 2, 8, 4, 6]
    s = TaskSplit.build(order, 0, num_of_first_cls=8, per_forget_cls=2)
    assert s.forget == [4, 6]
    assert s.remain == [7, 3, 9, 1, 5, 0, 2, 8]


def test_task_split_rejects_exhausted_schedule():
    with pytest.raises(ValueError):
        TaskSplit.build(list(range(100)), 9, num_of_first_cls=90, per_forget_cls=10)


# ----------------------------------------------------------------------
# wrappers
# ----------------------------------------------------------------------
def test_random_label_wrapper_is_stable_and_in_range(train_folder):
    """Scrambled labels must be stable across reads and inside the label space."""
    wrapped = RandomLabelWrapper(train_folder, num_classes=10, seed=0)
    first = [wrapped[i][1] for i in range(12)]
    second = [wrapped[i][1] for i in range(12)]
    assert first == second, "labels must be stable across epochs"
    assert all(0 <= y < 10 for y in first)
    originals = [train_folder.targets[i] for i in range(12)]
    assert all(a != b for a, b in zip(first, originals)), "label must differ"


def test_augmented_dataset_reapplies_transform(train_folder):
    """The transform must be re-applied per read, so duplicates differ."""
    calls = {"n": 0}

    def noisy(x):
        calls["n"] += 1
        return x + torch.randn_like(x) * 0.1

    ds = AugmentedDataset(train_folder, duplication_factor=3, transform=noisy)
    assert len(ds) == len(train_folder) * 3
    a, _ = ds[0]
    b, _ = ds[1]  # same base image, different augmentation
    assert calls["n"] == 2
    assert not torch.equal(a, b), "augmentation must actually vary"
    assert len(ds.targets) == len(ds)
