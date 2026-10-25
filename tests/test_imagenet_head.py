"""ImageNet-100: load the 1000-way pretrained head, keep only the 100 classes.

The backbone is torchvision's ImageNet-1k ViT-B/16, but the data has 100
classes whose ``ImageFolder`` indices (0..99, lexicographic) have nothing to
do with the ImageNet-1k class ids, so the head has to be remapped before
training and restored for the held-out evaluation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.models import VisionTransformer
from torchvision.transforms import ToTensor

from gslora.models.torchvision_vit import (
    ModifiedViT,
    inject_lora_into_mlp,
    load_imagenet_class_mapping,
    remap_head,
    restore_head,
)

N_GLOBAL = 20   # stand-in for ImageNet-1k
N_SUBSET = 5    # stand-in for the 100-class subset


@pytest.fixture
def global_names() -> list[str]:
    return [f"n{i:04d}" for i in range(N_GLOBAL)]


@pytest.fixture
def subset_folder(tmp_path, global_names) -> ImageFolder:
    """An ImageFolder over 5 of the 20 'global' classes, deliberately chosen so
    that lexicographic ImageFolder order != global order."""
    chosen = [global_names[7], global_names[2], global_names[19], global_names[0], global_names[11]]
    for name in chosen:
        d = tmp_path / "test" / name
        d.mkdir(parents=True)
        Image.new("RGB", (32, 32), (1, 2, 3)).save(d / "a.png")
    return ImageFolder(root=str(tmp_path / "test"), transform=ToTensor())


@pytest.fixture
def model() -> ModifiedViT:
    vit = VisionTransformer(
        image_size=32, patch_size=8, num_layers=2, num_heads=2,
        hidden_dim=16, mlp_dim=32, num_classes=N_GLOBAL,
    )
    # give the head recognisable per-class values
    with torch.no_grad():
        for cls in range(N_GLOBAL):
            vit.heads.head.weight[cls].fill_(float(cls))
            vit.heads.head.bias[cls] = float(100 + cls)
    return ModifiedViT(vit)


# ----------------------------------------------------------------------
def test_class_mapping_is_folder_index_to_global_index(subset_folder, global_names, tmp_path):
    names_file = tmp_path / "imagenet_folder_names.txt"
    names_file.write_text("\n".join(global_names))

    mapping = load_imagenet_class_mapping(names_file, subset_folder)

    assert len(mapping) == N_SUBSET
    # ImageFolder sorts lexicographically: n0000, n0002, n0007, n0011, n0019
    assert subset_folder.classes == ["n0000", "n0002", "n0007", "n0011", "n0019"]
    assert mapping == {0: 0, 1: 2, 2: 7, 3: 11, 4: 19}


def test_mapping_rejects_a_class_outside_the_1k_list(subset_folder, tmp_path):
    names_file = tmp_path / "short.txt"
    names_file.write_text("n0000\nn0002")       # missing n0007 etc.
    with pytest.raises(ValueError, match="absent from the 1k list"):
        load_imagenet_class_mapping(names_file, subset_folder)


def test_mapping_reports_a_missing_names_file(subset_folder, tmp_path):
    with pytest.raises(FileNotFoundError, match="imagenet_folder_names"):
        load_imagenet_class_mapping(tmp_path / "nope.txt", subset_folder)


# ----------------------------------------------------------------------
def test_remap_head_selects_the_right_rows(model, tmp_path):
    mapping = {0: 0, 1: 2, 2: 7, 3: 11, 4: 19}
    cache = tmp_path / "head.pt"

    remap_head(model, mapping, torch.device("cpu"), cache_path=cache)

    head = model.heads.head
    assert head.out_features == N_SUBSET, "head must be sliced to the subset size"
    for folder_idx, global_idx in mapping.items():
        assert torch.all(head.weight[folder_idx] == float(global_idx)), (
            f"row {folder_idx} should carry ImageNet class {global_idx}"
        )
        assert head.bias[folder_idx] == float(100 + global_idx)


def test_remap_head_preserves_the_pretrained_predictions(model, tmp_path):
    """The subset head must score the kept classes exactly as the 1k head did."""
    mapping = {0: 0, 1: 2, 2: 7, 3: 11, 4: 19}
    x = torch.randn(3, 3, 32, 32)
    full_logits, _ = model(x)

    remap_head(model, mapping, torch.device("cpu"), cache_path=tmp_path / "h.pt")
    subset_logits, _ = model(x)

    expected = full_logits[:, [mapping[i] for i in range(N_SUBSET)]]
    torch.testing.assert_close(subset_logits, expected)


def test_restore_head_round_trips(model, tmp_path):
    cache = tmp_path / "head.pt"
    mapping = {0: 0, 1: 2, 2: 7, 3: 11, 4: 19}
    x = torch.randn(2, 3, 32, 32)
    before, _ = model(x)

    remap_head(model, mapping, torch.device("cpu"), cache_path=cache)
    restored = restore_head(model, torch.device("cpu"), cache_path=cache)

    assert restored.heads.head.out_features == N_GLOBAL
    after, _ = restored(x)
    torch.testing.assert_close(after, before)
    # and the caller's model must keep its subset head
    assert model.heads.head.out_features == N_SUBSET


def test_restore_head_without_a_cache_is_an_error(model, tmp_path):
    with pytest.raises(FileNotFoundError, match="remap_head"):
        restore_head(model, torch.device("cpu"), cache_path=tmp_path / "absent.pt")


# ----------------------------------------------------------------------
def test_lora_injection_then_head_remap_keeps_everything_pretrained(model, tmp_path):
    """Injecting LoRA and remapping the head must leave the MLPs pretrained."""
    mlp = model.encoder.layers.encoder_layer_0.mlp
    with torch.no_grad():
        nn.init.constant_(mlp[0].weight, 0.25)
        nn.init.constant_(mlp[3].bias, 0.75)

    inject_lora_into_mlp(model, rank=4)
    remap_head(model, {0: 0, 1: 2, 2: 7, 3: 11, 4: 19}, torch.device("cpu"),
               cache_path=tmp_path / "h.pt")

    import loralib

    out_mlp = model.encoder.layers.encoder_layer_0.mlp
    assert isinstance(out_mlp[0], loralib.Linear)
    assert torch.all(out_mlp[0].weight == 0.25), "FFN weights were discarded"
    assert torch.all(out_mlp[3].bias == 0.75), "FFN bias was discarded"
    assert model.heads.head.out_features == N_SUBSET


# ----------------------------------------------------------------------
# End-to-end: the wiring that was actually missing
# ----------------------------------------------------------------------
@pytest.fixture
def fake_imagenet(tmp_path, monkeypatch):
    """A 100-class ImageNet-100 lookalike plus a stubbed-out ViT-B/16.

    The real backbone is an 86M-parameter 224x224 model whose weights are a
    330 MB download, so `vit_b_16` is replaced by a tiny VisionTransformer
    with the same module names. `--data_mode imagenet100` still asserts 100
    classes, so the tree really does have 100 of them.
    """
    import torchvision.models as tv

    root = tmp_path / "imagenet100"
    names = [f"n{i:08d}" for i in range(1000)]
    (root / "imagenet_folder_names.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "imagenet_folder_names.txt").write_text("\n".join(names))

    # every 10th global class is in the subset, so folder idx != global idx
    chosen = [names[i * 10] for i in range(100)]
    for split, n in (("train", 2), ("test", 1)):
        for name in chosen:
            d = root / split / name
            d.mkdir(parents=True)
            for j in range(n):
                Image.new("RGB", (32, 32), (j * 7 % 256, 40, 90)).save(d / f"{j}.png")

    def fake_vit_b_16(*args, **kwargs):
        return VisionTransformer(
            image_size=224, patch_size=32, num_layers=1, num_heads=2,
            hidden_dim=16, mlp_dim=32, num_classes=1000,
        )

    monkeypatch.setattr(tv, "vit_b_16", fake_vit_b_16)
    return str(root)


def test_imagenet_pipeline_slices_the_head(fake_imagenet, tmp_path):
    """An ImageNet-100 run must end up with a 100-way head, not the 1000-way one."""
    from gslora.config import Config
    from gslora.tools import train_forget

    cfg = Config(
        method="gslora", outdir=str(tmp_path / "out"),
        data_mode="imagenet100", data_root=fake_imagenet,
        net="VIT_B16", head="CosFace",
        batch_size=8, num_workers=0, data_ratio=1.0,
        num_tasks=1, num_of_first_cls=80, per_forget_cls=20,
        lora_rank=4, epochs=1, lr=1e-3, warmup_epochs=0,
        workers_id="cpu", tracker="jsonl", disp_freq=0, ver_freq=0,
    )
    out = train_forget.run(cfg)

    # the head must have been sliced 1000 -> 100
    state = torch.load(
        tmp_path / "out" / "task-level" / "Backbone_task_0.pth",
        map_location="cpu", weights_only=True,
    )
    assert state["heads.head.weight"].shape[0] == 100, (
        f"head still has {state['heads.head.weight'].shape[0]} outputs; "
        f"remap_head was not applied"
    )
    # and the full 1000-way head must be cached for the missing-class eval
    assert (tmp_path / "out" / "original_1k_head.pt").exists()
    assert len(out["history"]) == 1
