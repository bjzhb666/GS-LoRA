import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(
    image_set,
    args,
    cls_order,
    phase_idx,
    incremental,
    incremental_val,
    val_each_phase,
    balanced_ft=False,
    is_rehearsal=False,
):
    if args.dataset_file == "coco":
        return build_coco(
            image_set,
            args,
            cls_order,
            phase_idx,
            incremental,
            incremental_val,
            val_each_phase,
            balanced_ft,
            is_rehearsal,
        )
    raise ValueError(f"dataset {args.dataset_file} not supported")


import torch


class CLDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, data = self.dataset[index]
        new_data = data.copy()

        if "labels" in new_data:
            labels = new_data["labels"]
            if labels.numel() > 0:
                new_data["labels"] = torch.full_like(labels, 12)

        return image, new_data

    def __len__(self):
        return len(self.dataset)
