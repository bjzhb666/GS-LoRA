"""Pre-train the face backbone that the forgetting experiments start from.

Replaces ``train/train_own.py`` (557 lines) and
``train/backbone_forget_main.py`` (772 lines), which differed by only ~345
lines and shared the same 180-line inline argparse block.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.utils.data import DataLoader

from ..checkpoint import save_checkpoint, unwrap_model
from ..config import Config, parse_config, timm_args
from ..data import load_image_folders
from ..lora_utils import count_trainable_parameters, mark_only_lora_as_trainable
from ..metrics import AverageMeter, topk_accuracy
from ..models import build_backbone, load_backbone_weights
from ..prototypes import eval_mode
from ..seeding import make_generator, seed_worker, set_seed
from ..tracking import build_tracker

__all__ = ["main", "run"]


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = total = 0
    with eval_mode(model):
        for images, labels in loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            logits, _ = model(images, labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total else 0.0


def run(cfg: Config) -> dict:
    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
    cfg.save()
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    tracker = build_tracker(cfg)
    device = cfg.device

    train_ds, test_ds = load_image_folders(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(cfg.seed),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_backbone(cfg)
    if cfg.resume:
        load_backbone_weights(model, cfg.resume, map_location="cpu")
    if cfg.lora_rank > 0:
        mark_only_lora_as_trainable(model)
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    if cfg.multi_gpu:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    targs = timm_args(cfg)
    optimizer = create_optimizer(targs, unwrap_model(model))
    scheduler, _ = create_scheduler(targs, optimizer)

    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    best_acc, batch = 0.0, 0

    for epoch in range(cfg.epochs):
        scheduler.step(epoch)
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            logits, _ = model(images, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(topk_accuracy(logits.detach(), labels), images.size(0))
            batch += 1

            if cfg.disp_freq > 0 and batch % cfg.disp_freq == 0:
                tracker.log(
                    {"loss/train": loss_meter.avg, "train_acc/top1": acc_meter.avg,
                     "lr": optimizer.param_groups[0]["lr"], "epoch": epoch},
                    step=batch,
                )
                print(f"epoch {epoch + 1} batch {batch} | loss {loss_meter.val:.4f}"
                      f"({loss_meter.avg:.4f}) acc {acc_meter.avg:.2f}")
                loss_meter.reset()
                acc_meter.reset()

            if cfg.ver_freq > 0 and batch % cfg.ver_freq == 0:
                acc = _accuracy(model, test_loader, device)
                tracker.log({"acc/test": acc, "epoch": epoch}, step=batch)
                print(f"  test accuracy {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    save_checkpoint(model, cfg.outdir, backbone_name=cfg.net,
                                    epoch=epoch, batch=batch, keep=cfg.keep_checkpoints)
                model.train()

    final = _accuracy(model, test_loader, device)
    # Always leave a usable checkpoint behind: with --ver_freq 0 the loop
    # above never evaluates and so never saves.
    final_path = save_checkpoint(
        model, cfg.outdir, backbone_name=cfg.net,
        epoch=cfg.epochs - 1, batch=batch, keep=cfg.keep_checkpoints,
    )
    best_acc = max(best_acc, final)
    print(f"final test accuracy {final:.2f}% (best {best_acc:.2f}%) -> {final_path}")
    tracker.set_name(cfg.run_name or f"pretrain-{cfg.net}-{cfg.data_mode}-{cfg.head}-lr{cfg.lr}")
    tracker.finish()
    return {
        "best_acc": best_acc,
        "final_acc": final,
        "outdir": cfg.outdir,
        "checkpoint": str(final_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser_defaults = parse_config(argv, description="Pre-train a face ViT backbone")
    run(parser_defaults)
    return 0


if __name__ == "__main__":
    sys.exit(main())
