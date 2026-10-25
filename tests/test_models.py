"""Model forward passes, heads, and checkpoint-key compatibility."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from gslora.models import ViT_face, ViT_face_low, ViT_face_up, ViTs_face
from gslora.models.heads import ArcFace, CosFace, SFaceLoss, Softmax, build_head


FACE_KWARGS = dict(
    loss_type="CosFace",
    GPU_ID=None,
    num_class=10,
    image_size=32,
    patch_size=4,
    dim=32,
    depth=4,
    heads=2,
    mlp_dim=64,
    dim_head=16,
    lora_rank=4,
)


# ----------------------------------------------------------------------
# heads
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", [Softmax, ArcFace, CosFace, SFaceLoss])
def test_heads_produce_logits(cls):
    head = cls(in_features=8, out_features=5, device_id=None)
    emb = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 4, 2])
    out = head(emb, labels)
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


def test_softmax_head_runs_on_cpu():
    """Every head must work with device_id=None, i.e. on CPU."""
    head = Softmax(in_features=4, out_features=3, device_id=None)
    out = head(torch.randn(2, 4), torch.tensor([0, 1]))
    assert out.shape == (2, 3)


def test_arcface_no_nan_for_collinear_embedding():
    """cos can round marginally above 1; sqrt(1 - cos^2) must stay finite."""
    head = ArcFace(in_features=4, out_features=3, device_id=None)
    with torch.no_grad():
        head.weight.copy_(torch.eye(3, 4))
    emb = head.weight[:2].clone()          # exactly collinear -> cosine == 1
    out = head(emb, torch.tensor([0, 1]))
    assert torch.isfinite(out).all()


def test_build_head_rejects_unknown_name():
    """An unknown head name must raise at construction."""
    with pytest.raises(ValueError, match="Unknown head"):
        build_head("NotAHead", 4, 3)
    assert build_head("None", 4, 3) is None


# ----------------------------------------------------------------------
# backbones
# ----------------------------------------------------------------------
def test_vit_face_shapes():
    model = ViT_face(**FACE_KWARGS)
    x = torch.randn(3, 3, 32, 32)
    y = torch.tensor([0, 1, 2])
    logits, emb = model(x, y)
    assert logits.shape == (3, 10)
    assert emb.shape == (3, 32)
    assert model(x).shape == (3, 32), "no label -> embedding only"


def test_vits_face_shapes():
    model = ViTs_face(**FACE_KWARGS, ac_patch_size=6, pad=1)
    logits, emb = model(torch.randn(2, 3, 32, 32), torch.tensor([0, 3]))
    assert logits.shape == (2, 10)
    assert emb.shape == (2, 32)


def test_vit_face_backward():
    model = ViT_face(**FACE_KWARGS)
    logits, _ = model(torch.randn(2, 3, 32, 32), torch.tensor([1, 2]))
    nn.CrossEntropyLoss()(logits, torch.tensor([1, 2])).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_lora_rank_zero_has_no_lora_params():
    model = ViT_face(**{**FACE_KWARGS, "lora_rank": 0})
    assert not [n for n, _ in model.named_parameters() if "lora_" in n]


def test_lora_pos_attention_puts_lora_on_qkv():
    """--lora_pos Attention must put LoRA on qkv for both face backbones."""
    for cls, extra in ((ViT_face, {}), (ViTs_face, {"ac_patch_size": 6, "pad": 1})):
        model = cls(**{**FACE_KWARGS, "lora_pos": "Attention"}, **extra)
        names = [n for n, _ in model.named_parameters() if "lora_" in n]
        assert names, f"{cls.__name__} has no LoRA params"
        assert all("to_qkv" in n for n in names), f"{cls.__name__}: LoRA not on qkv"


def test_lora_parameter_names_are_checkpoint_stable():
    """Checkpoint keys are part of the public interface; pin them."""
    model = ViT_face(**FACE_KWARGS)
    keys = set(model.state_dict())
    for i in range(4):
        for j in (0, 3):
            for ab in ("A", "B"):
                assert f"transformer.layers.{i}.1.fn.fn.net.{j}.lora_{ab}" in keys
    for prefix in ("pos_embedding", "cls_token", "patch_to_embedding.weight",
                   "mlp_head.0.weight", "loss.weight"):
        assert prefix in keys


# ----------------------------------------------------------------------
# LIRF halves
# ----------------------------------------------------------------------
def test_lirf_halves_compose_to_the_full_model():
    low = ViT_face_low(**FACE_KWARGS)
    up = ViT_face_up(**FACE_KWARGS)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    mid = low(x)
    assert mid.dim() == 3, "low half returns tokens"
    logits, emb = up(mid, y)
    assert logits.shape == (2, 10)
    assert emb.shape == (2, 32)


def test_lirf_up_keeps_full_block_indices():
    """ViT_face_up must keep `depth` blocks so its keys line up with ViT_face,
    while only executing the top half."""
    full_keys = set(ViT_face(**FACE_KWARGS).state_dict())
    up = ViT_face_up(**FACE_KWARGS)
    assert set(up.state_dict()).issubset(full_keys)
    assert up.transformer.start_layer == 2
    assert len(up.transformer.layers) == 4


def test_lirf_low_has_half_the_blocks():
    low = ViT_face_low(**FACE_KWARGS)
    assert len(low.transformer.layers) == 2


# ----------------------------------------------------------------------
# torchvision ViT
# ----------------------------------------------------------------------
def test_inject_lora_preserves_pretrained_weights():
    """Injecting LoRA must carry the replaced layer's weights across."""
    from torchvision.models import vit_b_16

    from gslora.models.torchvision_vit import ModifiedViT, inject_lora_into_mlp

    vit = vit_b_16(weights=None)
    block = vit.encoder.layers.encoder_layer_0
    with torch.no_grad():
        nn.init.constant_(block.mlp[0].weight, 0.1234)
        nn.init.constant_(block.mlp[0].bias, 0.5678)
    before_w = block.mlp[0].weight.detach().clone()
    before_b = block.mlp[0].bias.detach().clone()

    model = inject_lora_into_mlp(ModifiedViT(vit), rank=4)
    after = model.encoder.layers.encoder_layer_0.mlp[0]
    import loralib

    assert isinstance(after, loralib.Linear), "LoRA was not injected"
    torch.testing.assert_close(after.weight, before_w)
    torch.testing.assert_close(after.bias, before_b)


def test_modified_vit_forward_and_deepcopy():
    """The wrapper must own its submodules, so a deepcopy is independent."""
    from torchvision.models import VisionTransformer

    from gslora.models.torchvision_vit import ModifiedViT

    vit = VisionTransformer(
        image_size=32, patch_size=8, num_layers=2, num_heads=2,
        hidden_dim=16, mlp_dim=32, num_classes=5,
    )
    model = ModifiedViT(vit)
    x = torch.randn(2, 3, 32, 32)
    logits, emb = model(x, torch.tensor([0, 1]))
    assert logits.shape == (2, 5)
    assert emb.shape == (2, 16)

    # torchvision zero-initialises heads.head, so compare embeddings, not logits
    clone = copy.deepcopy(model)
    with torch.no_grad():
        clone.conv_proj.weight.fill_(0.0)
    # mutating the clone must not affect the original...
    torch.testing.assert_close(model(x)[1], emb)
    # ...and the clone's patch embedding must actually use the clone's conv_proj,
    # which it would not if _process_input were still bound to the original ViT
    assert not torch.allclose(clone(x)[1], emb)
    assert not any("_process_input" in k for k in model.state_dict())

    # and the original ViT object must be independent of the clone too
    with torch.no_grad():
        vit.conv_proj.weight.fill_(7.0)
    assert not torch.allclose(model(x)[1], clone(x)[1])
