"""Backbone construction.

:func:`build_backbone` instantiates exactly the backbone ``--net`` names, so a
face run never touches torchvision's ImageNet weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .vit_face import ViT_face, ViT_face_low, ViT_face_up, ViTs_face

__all__ = ["build_backbone", "build_lirf_models", "load_backbone_weights"]

def _face_kwargs(cfg: Config) -> dict:
    """Geometry of the face backbones, read from the config instead of being
    repeated as literals at ~10 construction sites."""
    return dict(
        loss_type=cfg.head,
        GPU_ID=None,
        num_class=cfg.num_classes,
        image_size=cfg.input_size[0],
        patch_size=cfg.vit_patch_size,
        dim=cfg.vit_dim,
        depth=cfg.vit_depth,
        heads=cfg.vit_heads,
        dim_head=cfg.vit_dim_head,
        mlp_dim=cfg.vit_mlp_dim,
        dropout=cfg.vit_dropout,
        emb_dropout=cfg.vit_dropout,
        lora_rank=cfg.lora_rank,
        lora_pos=cfg.lora_pos,
    )


def _vits_extra(cfg: Config) -> dict:
    return dict(ac_patch_size=cfg.vits_ac_patch_size, pad=cfg.vits_pad)


def build_backbone(cfg: Config, pretrained: bool = True) -> nn.Module:
    """Build exactly the one backbone ``cfg.net`` names."""
    if cfg.net == "VIT":
        return ViT_face(**_face_kwargs(cfg))
    if cfg.net == "VITs":
        return ViTs_face(**_face_kwargs(cfg), **_vits_extra(cfg))
    if cfg.net == "VIT_B16":
        from .torchvision_vit import build_vit_b16

        return build_vit_b16(lora_rank=cfg.lora_rank, pretrained=pretrained)
    raise ValueError(f"Unknown --net {cfg.net!r}")


def build_lirf_models(cfg: Config) -> dict[str, nn.Module]:
    """The four half-networks LIRF needs (teacher low/up, student low, deposit low)."""
    if cfg.net != "VIT":
        raise ValueError("The LIRF baseline is only implemented for --net VIT")
    kwargs = _face_kwargs(cfg)
    return {
        "teacher_low": ViT_face_low(**kwargs),
        "teacher_up": ViT_face_up(**kwargs),
        "student_low": ViT_face_low(**kwargs),
        "deposit_low": ViT_face_low(**kwargs),
    }


def load_backbone_weights(
    model: nn.Module,
    path: str,
    *,
    map_location: str | torch.device = "cpu",
    allow_missing_substrings: tuple[str, ...] = ("lora_",),
    strict_report: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a checkpoint, tolerating only the keys we expect to be absent.

    Loads with an explicit ``map_location`` and ``weights_only=True``, unwraps
    a ``module.`` prefix or a nested ``state_dict``/``model`` key, and raises
    if anything other than the expected LoRA parameters is missing -- which
    almost always means the wrong checkpoint for this ``--net``.
    """
    state = torch.load(path, map_location=map_location, weights_only=True)
    # Unwrap common nesting and a DataParallel 'module.' prefix.
    for key in ("state_dict", "model", "backbone"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_real = [
        k for k in missing if not any(s in k for s in allow_missing_substrings)
    ]
    if strict_report and unexpected_real:
        raise RuntimeError(
            f"Checkpoint {path!r} is missing {len(unexpected_real)} non-LoRA "
            f"keys, e.g. {unexpected_real[:5]}. This is almost certainly the "
            f"wrong checkpoint for --net {type(model).__name__}."
        )
    return list(missing), list(unexpected)
