"""Synthetic fixtures so the whole suite runs without CASIA-100 / ImageNet-100."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from gslora.config import Config

# These tests use deliberately tiny tensors. On a many-core host torch's
# default thread pool (one thread per core) spends all its time on barriers:
# measured 13 ms for a single 32x64 `linear` on a 64-core box, ~18 s for one
# 12-step training run. Pinned to 1 thread it is ~0.6 s.
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory) -> str:
    """A 10-class ImageFolder tree of 32x32 images, laid out as train/ + test/.

    Class ``k`` is a solid colour that depends on ``k``, so a model can in
    principle fit it -- useful for the end-to-end smoke test.
    """
    root = tmp_path_factory.mktemp("tiny_faces")
    for split, per_class in (("train", 6), ("test", 3)):
        for cls in range(10):
            d = root / split / f"id{cls:03d}"
            d.mkdir(parents=True)
            for j in range(per_class):
                colour = (cls * 25 % 256, (cls * 70 + j) % 256, (255 - cls * 25) % 256)
                Image.new("RGB", (32, 32), colour).save(d / f"{j}.png")
    return str(root)


@pytest.fixture
def tiny_cfg(tiny_root, tmp_path) -> Config:
    """A Config pointing at ``tiny_root`` with a tiny, CPU-sized model."""
    return Config(
        method="gslora",
        outdir=str(tmp_path / "out"),
        data_mode="tsne",          # 10 classes, 112x112 spec
        data_root=tiny_root,
        batch_size=4,
        num_workers=0,
        data_ratio=1.0,
        num_tasks=1,
        num_of_first_cls=8,
        per_forget_cls=2,
        net="VIT",
        head="CosFace",
        image_size=32,
        vit_depth=2,
        vit_dim=32,
        vit_heads=2,
        vit_dim_head=16,
        vit_mlp_dim=64,
        vit_patch_size=4,
        lora_rank=4,
        epochs=1,
        workers_id="cpu",
        tracker="none",
        disp_freq=0,
        ver_freq=0,
    )


@pytest.fixture
def tiny_model():
    """A 2-block ViT_face on 32x32 inputs (>16 patches, as the assert requires)."""
    from gslora.models import ViT_face

    torch.manual_seed(0)
    return ViT_face(
        loss_type="CosFace",
        GPU_ID=None,
        num_class=10,
        image_size=32,
        patch_size=4,
        dim=32,
        depth=2,
        heads=2,
        mlp_dim=64,
        dim_head=16,
        lora_rank=4,
    )
