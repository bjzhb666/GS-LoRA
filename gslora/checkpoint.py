"""Checkpoint writing and rotation.

``--keep_checkpoints N`` keeps the N most recent checkpoints and deletes the
rest. Only files matching the checkpoint pattern are counted or removed, so
logs, configs and the ``task-level/`` directory are never at risk.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["timestamp", "unwrap_model", "save_checkpoint", "rotate_checkpoints"]

_CKPT_GLOB = "Backbone_*_checkpoint.pth"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(
        model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
    ) else model


def save_checkpoint(
    model: nn.Module,
    outdir: str | Path,
    *,
    backbone_name: str,
    epoch: int,
    batch: int,
    keep: int = 2,
) -> Path:
    """Write ``state_dict`` and prune old checkpoints. Returns the new path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (
        f"Backbone_{backbone_name}_Epoch_{epoch + 1}_Batch_{batch + 1}"
        f"_Time_{timestamp()}_checkpoint.pth"
    )
    torch.save(unwrap_model(model).state_dict(), path)
    rotate_checkpoints(outdir, keep=keep)
    return path


def rotate_checkpoints(outdir: str | Path, keep: int = 2, pattern: str = _CKPT_GLOB) -> list[Path]:
    """Keep the ``keep`` most recently modified checkpoints; delete the rest."""
    if keep < 1:
        raise ValueError("keep must be >= 1")
    outdir = Path(outdir)
    checkpoints = sorted(outdir.glob(pattern), key=lambda p: p.stat().st_mtime)
    removed: list[Path] = []
    for stale in checkpoints[:-keep]:
        stale.unlink(missing_ok=True)
        removed.append(stale)
    return removed


def save_task_checkpoint(model: nn.Module, outdir: str | Path, task_index: int) -> Path:
    path = Path(outdir) / "task-level" / f"Backbone_task_{task_index}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), path)
    return path


def load_task_checkpoint(model: nn.Module, outdir: str | Path, task_index: int, device) -> None:
    path = Path(outdir) / "task-level" / f"Backbone_task_{task_index}.pth"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint for task {task_index} at {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    unwrap_model(model).load_state_dict(state)
