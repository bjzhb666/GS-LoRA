"""Evaluate a checkpoint: overall and per-class accuracy.

Replaces ``test/test_own.py``, which had a real crash in its per-class loop::

    for i in range(args.batch_size):      # <- not len(labels)
        label = labels[i]

That raises ``IndexError`` on the last batch whenever the test set size is not
a multiple of ``--batch_size``, and double-counts nothing only by luck. It also
divided by ``class_total[i]`` without checking for empty classes, hardcoded the
dataset path, and ran the whole test set twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import Config, parse_config
from ..data import load_image_folders
from ..models import build_backbone, load_backbone_weights
from ..prototypes import eval_mode

__all__ = ["main", "run"]


@torch.no_grad()
def run(cfg: Config) -> dict:
    device = cfg.device
    _, test_ds = load_image_folders(cfg)
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = build_backbone(cfg)
    if not cfg.resume:
        raise ValueError("--resume CHECKPOINT is required for evaluation")
    load_backbone_weights(model, cfg.resume, map_location="cpu")
    model = model.to(device)

    n = cfg.num_classes
    correct_per_class = torch.zeros(n, dtype=torch.long)
    total_per_class = torch.zeros(n, dtype=torch.long)

    with eval_mode(model):
        for images, labels in loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            logits, _ = model(images, labels)
            hit = (logits.argmax(1) == labels).cpu()
            labels_cpu = labels.cpu()
            # single pass, no fixed-size inner loop
            total_per_class.index_add_(0, labels_cpu, torch.ones_like(labels_cpu))
            correct_per_class.index_add_(0, labels_cpu, hit.long())

    total = int(total_per_class.sum())
    overall = 100.0 * int(correct_per_class.sum()) / total if total else 0.0
    print(f"overall accuracy: {overall:.4f}%  ({int(correct_per_class.sum())}/{total})")

    per_class = {}
    for cls in range(n):
        seen = int(total_per_class[cls])
        if seen == 0:
            continue  # class absent from this test split
        per_class[test_ds.classes[cls]] = 100.0 * int(correct_per_class[cls]) / seen

    out_path = Path(cfg.outdir) / "per_class_accuracy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"overall": overall, "per_class": per_class}, indent=2), encoding="utf-8"
    )
    print(f"per-class accuracy written to {out_path}")
    return {"overall": overall, "per_class": per_class}


def main(argv: list[str] | None = None) -> int:
    run(parse_config(argv, description="Evaluate a GS-LoRA checkpoint"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
