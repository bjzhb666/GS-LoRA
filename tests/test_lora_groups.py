"""LoRA group enumeration: the parameter tables L_structure is computed over."""

from __future__ import annotations

import pytest
import torch

from gslora.lora_utils import (
    collect_group_params,
    group_norms,
    lora_group_names,
    mark_only_lora_as_trainable,
    reinitialize_lora_parameters,
)


def test_block_grouping_shape():
    groups = lora_group_names(6, "block", "FFN")
    assert len(groups) == 6
    assert all(len(g) == 4 for g in groups)
    assert groups[0] == [
        "transformer.layers.0.1.fn.fn.net.0.lora_A",
        "transformer.layers.0.1.fn.fn.net.0.lora_B",
        "transformer.layers.0.1.fn.fn.net.3.lora_A",
        "transformer.layers.0.1.fn.fn.net.3.lora_B",
    ]


def test_grouping_variants_partition_the_same_names():
    block = lora_group_names(4, "block", "FFN")
    lora = lora_group_names(4, "lora", "FFN")
    matrix = lora_group_names(4, "matrix", "FFN")
    assert (len(block), len(lora), len(matrix)) == (4, 8, 16)
    flat = lambda gs: sorted(n for g in gs for n in g)  # noqa: E731
    assert flat(block) == flat(lora) == flat(matrix)


@pytest.mark.parametrize("depth", [2, 6, 12, 18])
def test_depth_is_honoured(depth):
    """The group table must follow the backbone depth, not a fixed constant."""
    assert len(lora_group_names(depth, "block", "FFN")) == depth


def test_attention_position():
    groups = lora_group_names(3, "block", "Attention")
    assert len(groups) == 3
    assert groups[1] == [
        "transformer.layers.1.0.fn.fn.to_qkv.lora_A",
        "transformer.layers.1.0.fn.fn.to_qkv.lora_B",
    ]


def test_torchvision_naming():
    groups = lora_group_names(12, "block", "FFN", torchvision_vit=True)
    assert len(groups) == 12
    assert groups[0][0] == "encoder.layers.encoder_layer_0.mlp.0.lora_A"


def test_invalid_arguments():
    with pytest.raises(ValueError):
        lora_group_names(0, "block", "FFN")
    with pytest.raises(ValueError):
        lora_group_names(4, "nonsense", "FFN")
    with pytest.raises(ValueError):
        lora_group_names(4, "block", "elsewhere")
    with pytest.raises(ValueError):
        lora_group_names(4, "block", "Attention", torchvision_vit=True)


# ----------------------------------------------------------------------
def test_names_resolve_against_a_real_model(tiny_model):
    groups = lora_group_names(2, "block", "FFN")
    params = collect_group_params(tiny_model, groups, require_grad=False)
    assert len(params) == 2
    assert all(len(g) == 4 for g in params), "every LoRA name must exist on the model"


def test_collect_raises_on_wrong_depth(tiny_model):
    """A group table that does not match the backbone must fail loudly."""
    groups = lora_group_names(6, "block", "FFN")  # model only has 2 blocks
    with pytest.raises(KeyError, match="not found"):
        collect_group_params(tiny_model, groups, require_grad=False)


def test_mark_only_lora_as_trainable(tiny_model):
    mark_only_lora_as_trainable(tiny_model)
    trainable = {n for n, p in tiny_model.named_parameters() if p.requires_grad}
    assert trainable, "something must stay trainable"
    assert all("lora_" in n for n in trainable)


def test_reinitialize_lora(tiny_model):
    for n, p in tiny_model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.constant_(p, 3.0)
    reinitialize_lora_parameters(tiny_model)
    for n, p in tiny_model.named_parameters():
        if "lora_B" in n:
            assert torch.all(p == 0), "lora_B must be re-zeroed"
        if "lora_A" in n:
            assert not torch.all(p == 0), "lora_A must be re-randomised"


def test_group_norms_are_zero_for_fresh_b(tiny_model):
    """lora_B starts at zero, lora_A does not, so block norms are > 0."""
    norms = group_norms(tiny_model, lora_group_names(2, "block", "FFN"))
    assert len(norms) == 2
    assert all(n > 0 for n in norms)


# ----------------------------------------------------------------------
# discovery from the model (preferred over the static name table)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("grouping", ["block", "lora", "matrix"])
def test_discovery_matches_the_static_table(tiny_model, grouping):
    from gslora.lora_utils import discover_lora_groups

    assert discover_lora_groups(tiny_model, grouping) == lora_group_names(
        2, grouping, "FFN"
    )


def test_discovery_tracks_the_real_depth():
    """Discovery reads the block count off the model, at any depth."""
    from gslora.lora_utils import count_lora_blocks, discover_lora_groups
    from gslora.models import ViT_face

    for depth in (1, 3, 7):
        model = ViT_face(
            loss_type="CosFace", GPU_ID=None, num_class=4, image_size=32,
            patch_size=4, dim=16, depth=depth, heads=2, mlp_dim=32,
            dim_head=8, lora_rank=2,
        )
        assert count_lora_blocks(model) == depth
        assert len(discover_lora_groups(model, "block")) == depth


def test_discovery_handles_torchvision_naming():
    from torchvision.models import VisionTransformer

    from gslora.lora_utils import count_lora_blocks, discover_lora_groups
    from gslora.models.torchvision_vit import ModifiedViT, inject_lora_into_mlp

    model = inject_lora_into_mlp(
        ModifiedViT(
            VisionTransformer(image_size=32, patch_size=16, num_layers=3,
                              num_heads=2, hidden_dim=16, mlp_dim=32, num_classes=5)
        ),
        rank=2,
    )
    assert count_lora_blocks(model) == 3
    groups = discover_lora_groups(model, "block")
    assert groups[0] == [
        "encoder.layers.encoder_layer_0.mlp.0.lora_A",
        "encoder.layers.encoder_layer_0.mlp.0.lora_B",
        "encoder.layers.encoder_layer_0.mlp.3.lora_A",
        "encoder.layers.encoder_layer_0.mlp.3.lora_B",
    ]


def test_discovery_finds_attention_lora():
    from gslora.lora_utils import discover_lora_groups
    from gslora.models import ViT_face

    model = ViT_face(
        loss_type="CosFace", GPU_ID=None, num_class=4, image_size=32,
        patch_size=4, dim=16, depth=2, heads=2, mlp_dim=32, dim_head=8,
        lora_rank=2, lora_pos="Attention",
    )
    assert discover_lora_groups(model, "block") == lora_group_names(2, "block", "Attention")


def test_discovery_returns_nothing_without_lora():
    from gslora.lora_utils import discover_lora_groups
    from gslora.models import ViT_face

    model = ViT_face(
        loss_type="CosFace", GPU_ID=None, num_class=4, image_size=32,
        patch_size=4, dim=16, depth=2, heads=2, mlp_dim=32, dim_head=8, lora_rank=0,
    )
    assert discover_lora_groups(model, "block") == []
