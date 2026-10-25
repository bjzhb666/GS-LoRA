"""Face ViT backbones.

``ViT_face``, ``ViT_face_low`` and ``ViT_face_up`` were three near-identical
~110-line classes that differed only in which part of the network they run.
They now share :class:`_ViTFaceBase`; the public names, the module hierarchy
and therefore the ``state_dict`` keys are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .heads import build_head
from .transformer import Transformer

__all__ = ["ViT_face", "ViT_face_low", "ViT_face_up", "ViTs_face", "MIN_NUM_PATCHES"]

MIN_NUM_PATCHES = 16


class _ViTFaceBase(nn.Module):
    """Shared patch-embedding / head plumbing."""

    def __init__(
        self,
        *,
        loss_type: str,
        GPU_ID=None,
        num_class: int,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        lora_rank: int = 8,
        lora_pos: str = "FFN",
        patch_dim: int | None = None,
        transformer_depth: int | None = None,
        start_layer: int = 0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")
        num_patches = (image_size // patch_size) ** 2
        if num_patches <= MIN_NUM_PATCHES:
            raise ValueError(
                f"{num_patches} patches is too few for attention to be effective "
                f"(need > {MIN_NUM_PATCHES}); decrease patch_size."
            )
        if pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")

        self.patch_size = patch_size
        self.pool = pool
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(
            patch_dim if patch_dim is not None else channels * patch_size**2, dim
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,
            transformer_depth if transformer_depth is not None else depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            lora_rank,
            lora_pos=lora_pos,
            start_layer=start_layer,
        )

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim))
        self.loss = build_head(loss_type, in_features=dim, out_features=num_class, device_id=GPU_ID)

    # -- pieces subclasses reuse -------------------------------------------
    def embed_patches(self, img: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # out-of-place: `x += ...` mutates the patch-embedding output in place
        x = x + self.pos_embedding[:, : (n + 1)]
        return self.dropout(x)

    def head_from_tokens(self, x: torch.Tensor, label: torch.Tensor | None):
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        emb = self.mlp_head(self.to_latent(x))
        if label is None:
            return emb
        if self.loss is None:
            raise RuntimeError(
                "This backbone was built with loss_type='None' but was called "
                "with a label; it has no classification head."
            )
        return self.loss(emb, label), emb


class ViT_face(_ViTFaceBase):
    """Full face ViT.

    Returns ``emb`` when called without a label, else ``(logits, emb)``.
    """

    def forward(self, img: torch.Tensor, label: torch.Tensor | None = None, mask=None):
        x = self.transformer(self.embed_patches(img), mask)
        return self.head_from_tokens(x, label)


class ViT_face_low(_ViTFaceBase):
    """Bottom half of a face ViT; returns token features (LIRF student/teacher)."""

    def __init__(self, *, depth: int, **kwargs) -> None:
        super().__init__(depth=depth, transformer_depth=depth // 2, **kwargs)

    def forward(self, img: torch.Tensor, label: torch.Tensor | None = None, mask=None) -> torch.Tensor:
        return self.transformer(self.embed_patches(img), mask)


class ViT_face_up(_ViTFaceBase):
    """Top half of a face ViT; consumes token features from :class:`ViT_face_low`.

    Keeps the full ``depth`` blocks (so block indices and therefore checkpoint
    keys line up with :class:`ViT_face`) but only executes ``depth//2`` onward.
    """

    def __init__(self, *, depth: int, **kwargs) -> None:
        super().__init__(depth=depth, transformer_depth=depth, start_layer=depth // 2, **kwargs)

    def forward(self, x: torch.Tensor, label: torch.Tensor | None = None, mask=None):
        return self.head_from_tokens(self.transformer(x, mask), label)


class ViTs_face(_ViTFaceBase):
    """Face ViT with overlapping ("soft split") patches.

    Shares the LoRA-aware ``Attention``/``Transformer`` with :class:`ViT_face`,
    so ``--lora_pos Attention`` applies here too.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        ac_patch_size: int,
        pad: int,
        channels: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            patch_dim=channels * ac_patch_size**2,
            **kwargs,
        )
        self.soft_split = nn.Unfold(
            kernel_size=(ac_patch_size, ac_patch_size),
            stride=(patch_size, patch_size),
            padding=(pad, pad),
        )

    def embed_patches(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_to_embedding(self.soft_split(img).transpose(1, 2))
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (n + 1)]
        return self.dropout(x)

    def forward(self, img: torch.Tensor, label: torch.Tensor | None = None, mask=None):
        x = self.transformer(self.embed_patches(img), mask)
        return self.head_from_tokens(x, label)
