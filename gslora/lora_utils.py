"""LoRA group bookkeeping.

The single source of truth for *which parameters form a group*, which is what
:math:`\mathcal{L}_{structure}` is computed over and what the sparsity report
at the end of a task measures.

Three granularities are supported (``--grouping``):

* ``block``  -- all four LoRA matrices of a transformer block form one group
                (the paper's setting)
* ``lora``   -- the A/B pair of a single Linear is a group
* ``matrix`` -- every matrix on its own

Prefer :func:`discover_lora_groups`, which reads the groups off the model, over
the static :func:`lora_group_names` table: it cannot disagree with the backbone
that was actually built.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Literal, Sequence

import torch
import torch.nn as nn

__all__ = [
    "GroupType",
    "GroupPos",
    "lora_group_names",
    "discover_lora_groups",
    "count_lora_blocks",
    "collect_group_params",
    "group_norms",
    "reinitialize_lora_parameters",
    "mark_only_lora_as_trainable",
    "count_trainable_parameters",
]

GroupType = Literal["block", "lora", "matrix"]
GroupPos = Literal["FFN", "Attention"]

# Parameter-name templates. ``{i}`` is the transformer block index.
#
# ViT_face FFN:  transformer.layers.{i}.1.fn.fn.net.{0,3}.lora_{A,B}
# ViT_face attn: transformer.layers.{i}.0.fn.fn.to_qkv.lora_{A,B}
# torchvision ViT FFN: encoder.layers.encoder_layer_{i}.mlp.{0,3}.lora_{A,B}
_FACE_FFN = "transformer.layers.{i}.1.fn.fn.net.{j}.lora_{ab}"
_FACE_ATTN = "transformer.layers.{i}.0.fn.fn.to_qkv.lora_{ab}"
_TV_FFN = "encoder.layers.encoder_layer_{i}.mlp.{j}.lora_{ab}"


def lora_group_names(
    num_layers: int,
    group_type: GroupType = "block",
    group_pos: GroupPos = "FFN",
    torchvision_vit: bool = False,
) -> list[list[str]]:
    """Enumerate parameter names, grouped.

    Args:
        num_layers: number of transformer blocks that carry LoRA.
        group_type: ``block`` (all 4 matrices of a block form one group),
            ``lora`` (the A/B pair of one Linear is a group) or ``matrix``
            (every matrix on its own).
        group_pos: whether LoRA sits on the FFN or on the qkv projection.
        torchvision_vit: use torchvision ``vit_b_16`` naming instead of
            ``ViT_face`` naming.

    Returns:
        A list of groups, each a list of parameter names.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if group_pos not in ("FFN", "Attention"):
        raise ValueError(f"group_pos must be 'FFN' or 'Attention', got {group_pos!r}")

    if group_pos == "Attention":
        if torchvision_vit:
            raise ValueError(
                "Attention-position LoRA is only implemented for the ViT_face "
                "backbones, not for the torchvision ViT."
            )
        # MergedLinear keeps a single A/B pair for the fused qkv projection, so
        # 'block', 'lora' and 'matrix' would all collapse to the same thing.
        return [
            [_FACE_ATTN.format(i=i, ab="A"), _FACE_ATTN.format(i=i, ab="B")]
            for i in range(num_layers)
        ]

    tpl = _TV_FFN if torchvision_vit else _FACE_FFN
    # j=0 is the first Linear of the MLP, j=3 the second one.
    def name(i: int, j: int, ab: str) -> str:
        return tpl.format(i=i, j=j, ab=ab)

    if group_type == "block":
        return [
            [name(i, 0, "A"), name(i, 0, "B"), name(i, 3, "A"), name(i, 3, "B")]
            for i in range(num_layers)
        ]
    if group_type == "lora":
        return [
            [name(i, j, "A"), name(i, j, "B")]
            for j in (0, 3)
            for i in range(num_layers)
        ]
    if group_type == "matrix":
        return [
            [name(i, j, ab)]
            for j in (0, 3)
            for ab in ("A", "B")
            for i in range(num_layers)
        ]
    raise ValueError(f"group_type must be block/lora/matrix, got {group_type!r}")


# Matches `...layers.<i>.` (ViT_face) and `...layers.encoder_layer_<i>.`
# (torchvision), i.e. the transformer-block index in a parameter name.
_BLOCK_RE = re.compile(r"(?:^|\.)layers\.(?:encoder_layer_)?(\d+)\.")


def discover_lora_groups(
    model: nn.Module, group_type: GroupType = "block"
) -> list[list[str]]:
    """Derive the groups from the model's own parameters.

    Reading the block count off the model keeps the group table consistent
    with whatever backbone was built, including depths the config never
    mentions.

    Ordering matches :func:`lora_group_names` so the reported norm lists stay
    comparable.
    """
    model = unwrap(model)
    entries: list[tuple[str, int, str, str]] = []  # (slot, block, ab, name)
    for name, _ in model.named_parameters():
        if ".lora_A" not in name and ".lora_B" not in name:
            continue
        match = _BLOCK_RE.search(name)
        if match is None:
            raise ValueError(
                f"Cannot work out the transformer-block index of LoRA parameter "
                f"{name!r}. Extend _BLOCK_RE if this backbone names blocks "
                f"differently."
            )
        block = int(match.group(1))
        head, ab = name.rsplit(".lora_", 1)
        slot = head[match.end():]  # e.g. '1.fn.fn.net.0' or 'mlp.0' or '0.fn.fn.to_qkv'
        entries.append((slot, block, ab, name))

    if not entries:
        return []

    if group_type == "block":
        by_block: dict[int, list[tuple]] = {}
        for entry in entries:
            by_block.setdefault(entry[1], []).append(entry)
        return [
            [e[3] for e in sorted(by_block[b], key=lambda e: (e[0], e[2]))]
            for b in sorted(by_block)
        ]
    if group_type == "lora":
        by_pair: dict[tuple[str, int], list[tuple]] = {}
        for entry in entries:
            by_pair.setdefault((entry[0], entry[1]), []).append(entry)
        return [
            [e[3] for e in sorted(by_pair[k], key=lambda e: e[2])]
            for k in sorted(by_pair)
        ]
    if group_type == "matrix":
        return [[e[3]] for e in sorted(entries, key=lambda e: (e[0], e[2], e[1]))]
    raise ValueError(f"group_type must be block/lora/matrix, got {group_type!r}")


def count_lora_blocks(model: nn.Module) -> int:
    """How many transformer blocks actually carry LoRA parameters."""
    return len(discover_lora_groups(model, "block"))


def unwrap(model: nn.Module) -> nn.Module:
    """Strip a ``DataParallel`` / ``DistributedDataParallel`` wrapper."""
    return model.module if hasattr(model, "module") and isinstance(
        model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
    ) else model


def collect_group_params(
    model: nn.Module,
    groups: Sequence[Sequence[str]],
    require_grad: bool = True,
    strict: bool = True,
) -> list[list[torch.nn.Parameter]]:
    """Resolve parameter names to tensors, dropping names that are absent.

    A name that is absent from the model is either an error (``strict=True``,
    the default) or skipped, rather than being carried through as ``None``.
    """
    model = unwrap(model)
    available = dict(model.named_parameters())
    out: list[list[torch.nn.Parameter]] = []
    for group in groups:
        resolved: list[torch.nn.Parameter] = []
        for pname in group:
            param = available.get(pname)
            if param is None:
                if strict:
                    raise KeyError(
                        f"LoRA parameter {pname!r} not found on "
                        f"{type(model).__name__}. Check --vit_depth / --lora_pos "
                        f"against the backbone you built."
                    )
                continue
            if require_grad and not param.requires_grad:
                continue
            resolved.append(param)
        out.append(resolved)
    return out


@torch.no_grad()
def group_norms(
    model: nn.Module,
    groups: Sequence[Sequence[str]],
    ord: Literal["L1", "L2"] = "L2",
) -> list[float]:
    """Sum of per-tensor norms within each group (used for reporting sparsity)."""
    if ord not in ("L1", "L2"):
        raise ValueError("ord must be 'L1' or 'L2'")
    p = 1 if ord == "L1" else 2
    params = collect_group_params(model, groups, require_grad=False, strict=False)
    return [
        float(sum(torch.norm(t.detach(), p=p) for t in group)) if group else 0.0
        for group in params
    ]


@torch.no_grad()
def reinitialize_lora_parameters(model: nn.Module, a: float = math.sqrt(5)) -> None:
    """Reset LoRA A to Kaiming-uniform and B to zero (loralib's own init).

    Called between continual tasks so each task starts from a fresh adapter.
    The default ``a`` matches ``loralib.Linear.reset_parameters``; it is exposed
    so the initialisation scale can be varied.
    """
    model = unwrap(model)
    for name, param in model.named_parameters():
        if "lora_A" in name:
            nn.init.kaiming_uniform_(param, a=a)
        elif "lora_B" in name:
            nn.init.zeros_(param)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Thin wrapper so callers do not need to import loralib directly."""
    import loralib

    loralib.mark_only_lora_as_trainable(unwrap(model), bias=bias)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for every loralib layer in the model."""
    import loralib

    for name, module in unwrap(model).named_modules():
        if isinstance(module, loralib.LoRALayer):
            yield name, module
