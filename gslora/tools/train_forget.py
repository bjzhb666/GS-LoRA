"""Single entry point for every forgetting experiment.

Replaces all four of::

    train/train_own_forget.py          (  877 lines)
    train/train_own_forget_open.py     ( 1102 lines)
    train/train_own_forget_cl.py       ( 1807 lines)
    train/train_own_forget_cl_open.py  ( 1706 lines)
                                        -----------
                                          5492 lines

which were copy-paste variants of one another. The two axes they encoded as
separate files are now flags:

* continual vs single-step  ->  ``--num_tasks N`` (1 == single step)
* closed vs missing-class   ->  ``--open_cls_num N`` (0 == closed)

and the method is ``--method``.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from ..checkpoint import load_task_checkpoint, save_task_checkpoint, unwrap_model
from ..config import Config, parse_config, timm_args
from ..data import build_task_data, load_image_folders
from ..data.builders import build_importance_loader
from ..ema import ModelEma
from ..engine import Evaluator, GSLoRATrainer, PairedBatches
from ..lora_utils import (
    count_trainable_parameters,
    discover_lora_groups,
    group_norms,
    mark_only_lora_as_trainable,
    reinitialize_lora_parameters,
)
from ..models import build_backbone, build_lirf_models, load_backbone_weights
from ..methods import ANCHOR_METHODS, TRAINERS, needs_teacher
from ..methods.regularization import estimate_importance, snapshot_params
from ..prototypes import compute_prototypes
from ..seeding import set_seed, shuffled_class_order
from ..tracking import build_tracker

__all__ = ["main", "run"]


# ----------------------------------------------------------------------
def _freeze_for_method(model: nn.Module, cfg: Config) -> None:
    """Decide which parameters train, per method."""
    if cfg.method == "gslora":
        if cfg.lora_rank > 0:
            mark_only_lora_as_trainable(model)
        return

    inner = unwrap_model(model)
    if cfg.only_ffn:
        for name, p in inner.named_parameters():
            p.requires_grad_(("fn.fn.net" in name) or ("mlp." in name) or ("loss" in name))
        return

    # Baselines fine-tune the backbone but keep the classifier fixed unless
    # --ffn_open.
    for name, p in inner.named_parameters():
        if "loss" in name and not cfg.ffn_open:
            p.requires_grad_(False)
        if cfg.torchvision_vit and "heads" in name:
            p.requires_grad_(False)


class _NullScheduler:
    """Stand-in for ``--epochs 0`` (evaluate-only runs).

    timm's ``CosineLRScheduler`` asserts ``t_initial > 0``, so building one for
    a zero-epoch run crashes.
    """

    def step(self, epoch: int) -> None:
        return None


def _build_optimizer_and_scheduler(cfg: Config, model: nn.Module):
    targs = timm_args(cfg)
    optimizer = create_optimizer(targs, unwrap_model(model))
    if cfg.epochs <= 0:
        return optimizer, _NullScheduler()
    scheduler, _ = create_scheduler(targs, optimizer)
    return optimizer, scheduler


@dataclass
class _ImageNetContext:
    """State needed by the ImageNet-100 runs beyond the generic pipeline."""

    class_map: dict[int, int]          # ImageFolder idx -> ImageNet-1k idx
    head_cache: Path                   # where the full 1000-way head is kept
    missing_loader: object | None = None   # held-out ImageNet-900 validation set


def _setup_imagenet_head(cfg: Config, backbone, test_ds, device, tracker):
    """Map the pretrained 1000-way head onto the 100-class subset.

    Runs after ``--resume`` (which restores the full head) and before the
    parameters are frozen. Also measures the untouched model on the held-out
    ImageNet-900 split, if it is present, as the baseline for how much
    forgetting 100 classes disturbs the other 900.
    """
    from ..data.wrappers import FlatImageDataset
    from ..models.torchvision_vit import load_imagenet_class_mapping, remap_head

    root = Path(cfg.data_root)
    class_map = load_imagenet_class_mapping(root / "imagenet_folder_names.txt", test_ds)
    head_cache = Path(cfg.outdir) / "original_1k_head.pt"

    ctx = _ImageNetContext(class_map=class_map, head_cache=head_cache)

    # Accuracy of the untouched 1000-way model on the held-out 900 classes,
    # measured before we touch the head.
    missing_dir = root / "imagenet_val_missing"
    if missing_dir.is_dir():
        from torch.utils.data import DataLoader

        from ..data import build_transform

        names = [
            line.strip()
            for line in (root / "imagenet_folder_names.txt").read_text().splitlines()
            if line.strip()
        ]
        ctx.missing_loader = DataLoader(
            FlatImageDataset.from_directory(
                missing_dir,
                {name: i for i, name in enumerate(names)},
                transform=build_transform(cfg),
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        acc = _missing_class_accuracy(backbone.to(device), ctx.missing_loader, device)
        tracker.log({"acc/imagenet900_before": acc})
        print(f"held-out ImageNet-900 accuracy before editing: {acc:.2f}%")
    else:
        print(f"(no {missing_dir}; skipping the missing-class evaluation)")

    remap_head(backbone, class_map, device, cache_path=head_cache)
    print(f"remapped the ImageNet-1k head to the {len(class_map)} subset classes")
    return ctx


@torch.no_grad()
def _missing_class_accuracy(model, loader, device) -> float:
    from ..engine.evaluator import default_forward, evaluate_accuracy
    from ..prototypes import eval_mode

    with eval_mode(model):
        return evaluate_accuracy(default_forward(model), loader, device)


def _report_capacity(cfg: Config, model: nn.Module, tracker) -> None:
    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} / {total:,} ({trainable / max(total, 1):.4%})")
    tracker.log(
        {
            "params/trainable": trainable,
            "params/total": total,
            "params/ratio": trainable / max(total, 1),
            "params/lora_rank": cfg.lora_rank,
        }
    )


# ----------------------------------------------------------------------
def run(cfg: Config) -> dict:
    cfg.validate_schedule()
    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
    cfg.save()
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    tracker = build_tracker(cfg)
    device = cfg.device

    print("=" * 70)
    print(f"method={cfg.method}  net={cfg.net}  data={cfg.data_mode}  device={device}")
    print(f"tasks={cfg.num_tasks}  first_cls={cfg.num_of_first_cls}  "
          f"forget/task={cfg.per_forget_cls}  open={cfg.open_cls_num}")
    print("=" * 70)

    train_ds, test_ds = load_image_folders(cfg)
    class_order = shuffled_class_order(cfg.num_classes, cfg.seed)

    # ---- backbone -----------------------------------------------------
    backbone = build_backbone(cfg)
    if cfg.resume and cfg.method != "retrain":
        if not Path(cfg.resume).is_file():
            raise FileNotFoundError(f"--resume {cfg.resume} does not exist")
        load_backbone_weights(backbone, cfg.resume, map_location="cpu")
        print(f"loaded backbone weights from {cfg.resume}")
    elif cfg.method != "retrain":
        print("\033[33mWARNING: no --resume given; forgetting from a randomly "
              "initialised backbone is almost never what you want.\033[0m")

    # ---- ImageNet-100: slice the 1000-way head down to the 100 subset classes
    # Must happen AFTER --resume (which restores the full 1000-way head) and
    # BEFORE freezing/optimiser construction.
    imagenet = None
    if cfg.torchvision_vit:
        imagenet = _setup_imagenet_head(cfg, backbone, test_ds, device, tracker)

    _freeze_for_method(backbone, cfg)
    _report_capacity(cfg, backbone, tracker)

    if cfg.multi_gpu:
        backbone = nn.DataParallel(backbone, device_ids=cfg.gpu_ids)
    backbone = backbone.to(device)

    lirf_models = None
    if cfg.method == "lirf":
        lirf_models = build_lirf_models(cfg)
        for name, module in lirf_models.items():
            if cfg.resume:
                load_backbone_weights(module, cfg.resume, map_location="cpu", strict_report=False)
            lirf_models[name] = module.to(device)
        for p in lirf_models["teacher_low"].parameters():
            p.requires_grad_(False)
        for p in lirf_models["teacher_up"].parameters():
            p.requires_grad_(False)
        backbone = lirf_models["student_low"]

    criterion = nn.CrossEntropyLoss()
    ema = ModelEma(backbone, cfg.ema_decay, device) if cfg.average_weight else None

    regularization_terms: dict = {}
    importance_loader = None
    history: list[dict] = []

    # ---- task loop ----------------------------------------------------
    for task in range(cfg.num_tasks):
        print(f"\n\033[34m{'=' * 25} task {task} {'=' * 25}\033[0m")

        if task > 0 and cfg.method == "gslora":
            load_task_checkpoint(backbone, cfg.outdir, task - 1, device)
            reinitialize_lora_parameters(backbone)
            print(f"restored task {task - 1} weights and re-initialised LoRA")

        data = build_task_data(cfg, task, train_ds, test_ds, class_order)
        print(data.describe())

        # prototypes come from the model as it is *before* this task's edits
        prototypes = None
        if cfg.prototype:
            from torch.utils.data import ConcatDataset

            prototypes = compute_prototypes(
                backbone,
                ConcatDataset([data.forget_train, data.remain_train]),
                device=device,
                batch_size=max(1, cfg.batch_size),
                aug_magnitude=cfg.proto_aug_num,
                aug_repeats=cfg.proto_aug_repeats,
            )
            print(f"computed {len(prototypes)} class prototypes")

        if cfg.method == "retrain":
            # fresh model per task; rebuild everything that points at it
            backbone = build_backbone(cfg, pretrained=False).to(device)
            _freeze_for_method(backbone, cfg)

        optimizer, scheduler = _build_optimizer_and_scheduler(cfg, backbone)

        evaluator = Evaluator(
            cfg,
            tracker,
            forget_loader=data.forget_test_loader,
            remain_loader=data.remain_test_loader,
            open_loader=data.open_test_loader,
            old_loader=data.old_test_loader,
            task_index=task,
        )

        teacher = None
        if needs_teacher(cfg.method):
            teacher = copy.deepcopy(unwrap_model(backbone)).to(device).eval()

        trainer = _make_trainer(
            cfg, backbone, criterion, optimizer, tracker, evaluator,
            task=task, data=data, prototypes=prototypes, teacher=teacher,
            lirf_models=lirf_models, regularization_terms=regularization_terms,
        )

        before = evaluator.baseline(backbone, forward=trainer_eval_forward(trainer))
        print(f"before: forget={before.forget_acc:.2f}%  remain={before.remain_acc:.2f}%"
              + (f"  open={before.open_acc:.2f}%" if before.open_acc is not None else "")
              + (f"  old={before.old_acc:.2f}%" if before.old_acc is not None else ""))

        # ---- importance for the anchor baselines (first task only) -----
        if cfg.method in ANCHOR_METHODS and task == 0:
            importance_loader = build_importance_loader(cfg, train_ds, class_order, 0)
            regularization_terms[0] = {
                "importance": estimate_importance(
                    cfg.method, backbone, importance_loader, criterion, device
                ),
                "task_param": snapshot_params(backbone),
            }
            trainer.regularization_terms = regularization_terms

        pairs = PairedBatches(
            data.replay_loader or data.forget_loader, data.remain_loader, device
        )

        # ---- optimise --------------------------------------------------
        if cfg.method == "scrub":
            for superepoch in range(cfg.scrub_superepoch):
                trainer.run_superepoch(pairs, superepoch)
        else:
            for epoch in range(cfg.epochs):
                scheduler.step(epoch)
                trainer.train_one_epoch(pairs, epoch)
                if ema is not None and ema.step(backbone, epoch, cfg.ema_epoch):
                    evaluator.run(ema.eval(), epoch=epoch, batch=trainer.batch,
                                  save=False, tag="ema")

        after = evaluator.run(backbone, forward=trainer_eval_forward(trainer),
                              epoch=cfg.epochs, batch=trainer.batch, save=False, tag="after")
        print(f"after:  forget={after.forget_acc:.2f}%  remain={after.remain_acc:.2f}%  "
              f"H={after.h_mean:.2f}  (best H during task: {evaluator.best_h_mean:.2f})")

        # ---- post-task bookkeeping -------------------------------------
        if cfg.method == "gslora":
            norms = group_norms(backbone, discover_lora_groups(backbone, cfg.grouping))
            tracker.log({f"lora_norm/task{task}": norms})
            print("LoRA group norms: " + ", ".join(f"{n:.4f}" for n in norms))

        if cfg.method in ANCHOR_METHODS and task < cfg.num_tasks - 1:
            next_loader = build_importance_loader(cfg, train_ds, class_order, task + 1)
            importance = estimate_importance(
                cfg.method, backbone, next_loader, criterion, device,
                warm_start=regularization_terms[0]["importance"] if cfg.online else None,
            )
            key = 0 if cfg.online else task + 1
            regularization_terms[key] = {
                "importance": importance,
                "task_param": snapshot_params(backbone),
            }

        # ImageNet-100: how much did editing the 100 classes damage the other
        # 900? Temporarily put the full 1000-way head back to find out.
        if imagenet is not None and imagenet.missing_loader is not None:
            from ..models.torchvision_vit import restore_head

            restored = restore_head(
                unwrap_model(backbone), device, cache_path=imagenet.head_cache
            )
            acc900 = _missing_class_accuracy(restored, imagenet.missing_loader, device)
            tracker.log({f"acc/imagenet900/task{task}": acc900})
            print(f"held-out ImageNet-900 accuracy after task {task}: {acc900:.2f}%")
            del restored

        save_task_checkpoint(backbone, cfg.outdir, task)
        history.append({"task": task, **after.as_metrics(), "best_h_mean": evaluator.best_h_mean})

    tracker.set_name(cfg.run_name or _auto_run_name(cfg))
    tracker.finish()
    return {"history": history, "outdir": cfg.outdir}


def trainer_eval_forward(trainer):
    getter = getattr(trainer, "eval_forward", None)
    return getter() if callable(getter) else None


def _make_trainer(cfg, backbone, criterion, optimizer, tracker, evaluator, *,
                  task, data, prototypes, teacher, lirf_models, regularization_terms):
    common = dict(
        cfg=cfg, model=backbone, criterion=criterion, optimizer=optimizer,
        tracker=tracker, evaluator=evaluator, task_index=task,
    )
    if cfg.method == "gslora":
        return GSLoRATrainer(
            **common,
            beta=cfg.cl_beta_list[task],
            prototype_weight_forget=(cfg.cl_prof_list[task] if cfg.cl_prof_list else None),
            prototypes=prototypes,
        )

    cls = TRAINERS[cfg.method]
    if cfg.method in ANCHOR_METHODS:
        lam = {"l2": cfg.l2_lambda, "ewc": cfg.ewc_lambda, "mas": cfg.mas_lambda}[cfg.method]
        return cls(**common, regularization_terms=regularization_terms, reg_lambda=lam)
    if cfg.method == "lirf":
        return cls(**common, teacher_low=lirf_models["teacher_low"],
                   teacher_up=lirf_models["teacher_up"],
                   deposit_low=lirf_models["deposit_low"],
                   forget_class_ids=data.split.forget)
    if cfg.method == "scrub":
        swa = torch.optim.swa_utils.AveragedModel(
            backbone, avg_fn=lambda avg, new, n: (1 - 0.1) * avg + 0.1 * new
        ).to(cfg.device)
        return cls(**common, teacher=teacher, swa_model=swa)
    if teacher is not None:
        return cls(**common, teacher=teacher)
    return cls(**common)


def _auto_run_name(cfg: Config) -> str:
    bits = [cfg.method, cfg.data_mode, f"r{cfg.lora_rank}",
            f"first{cfg.num_of_first_cls}", f"fgt{cfg.per_forget_cls}",
            f"t{cfg.num_tasks}", f"lr{cfg.lr}", f"beta{cfg.beta}", f"alpha{cfg.alpha}"]
    if cfg.few_shot:
        bits.insert(0, f"shot{cfg.few_shot_num}")
    if cfg.open_set:
        bits.insert(0, f"open{cfg.open_cls_num}")
    return "-".join(bits)


def main(argv: list[str] | None = None) -> int:
    cfg = parse_config(argv, description="GS-LoRA continual / single-step forgetting")
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
