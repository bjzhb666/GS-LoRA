"""Transformer building blocks shared by ``ViT_face`` and ``ViTs_face``.

Pre-norm residual blocks. LoRA goes either on the FFN Linears or on the fused
qkv projection, selected by ``--lora_pos``.

Parameter names follow ``transformer.layers.{i}.{0,1}.fn.fn.net.{0,3}``
``.lora_{A,B}``, which is what :mod:`gslora.lora_utils` groups over.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import loralib as lora

__all__ = ["Residual", "PreNorm", "FeedForward", "Attention", "Transformer"]


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """MLP block. ``net.0`` and ``net.3`` are the LoRA-carrying Linears."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, lora_rank: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            lora.Linear(dim, hidden_dim, r=lora_rank),
            nn.GELU(),
            nn.Dropout(dropout),
            lora.Linear(hidden_dim, dim, r=lora_rank),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        lora_rank: int = 0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim**-0.5
        self.to_qkv = lora.MergedLinear(
            in_features=dim,
            out_features=inner_dim * 3,
            r=lora_rank,
            enable_lora=[True, True, True],
            bias=False,
        )
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            # out-of-place: masked_fill_ on `dots` breaks autograd under
            # torch.compile and is unnecessary here
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Stack of pre-norm (attention, FFN) residual pairs.

    Args:
        start_layer: index of the first block to actually run. The LIRF
            baseline splits a ViT in half and runs the top half on features
            produced elsewhere, so it builds the full stack (keeping block
            indices, and therefore checkpoint keys, aligned) but starts at
            ``depth // 2``.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        lora_rank: int,
        lora_pos: str = "FFN",
        start_layer: int = 0,
    ) -> None:
        super().__init__()
        if lora_pos not in ("FFN", "Attention"):
            raise ValueError(f"lora_pos must be 'FFN' or 'Attention', got {lora_pos!r}")
        self.depth = depth
        self.start_layer = start_layer
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    Residual(
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                lora_rank=lora_rank if lora_pos == "Attention" else 0,
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout=dropout,
                                lora_rank=lora_rank if lora_pos == "FFN" else 0,
                            ),
                        )
                    ),
                ]
            )
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for attn, ff in list(self.layers)[self.start_layer :]:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
