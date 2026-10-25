"""torchvision ViT-B/16 adapted to the ``(logits, embeddings)`` interface.

Used for the ImageNet-100 experiments:

* :class:`ModifiedViT` wraps torchvision's ``VisionTransformer`` so it returns
  both the logits and the ``[CLS]`` embedding the prototype loss needs.
* :func:`inject_lora_into_mlp` swaps each encoder block's MLP Linears for
  ``loralib`` equivalents, carrying the pretrained weights across.
* :func:`remap_head` / :func:`restore_head` slice the pretrained 1000-way head
  down to the 100-class subset and put it back for the held-out evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

import loralib as lora

__all__ = ["ModifiedViT", "inject_lora_into_mlp", "build_vit_b16", "remap_head", "restore_head"]


class ModifiedViT(nn.Module):
    """Wrap a torchvision ``VisionTransformer`` to return ``(logits, embedding)``.

    Output: ``logits`` ``(B, num_classes)``, ``embedding`` ``(B, hidden_dim)``
    taken from the ``[CLS]`` token.
    """

    def __init__(self, vit_model: nn.Module) -> None:
        super().__init__()
        self.patch_size = vit_model.patch_size
        self.image_size = vit_model.image_size
        self.hidden_dim = vit_model.hidden_dim
        self.conv_proj = vit_model.conv_proj
        self.class_token = vit_model.class_token
        self.encoder = vit_model.encoder
        self.heads = vit_model.heads

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} input, got {h}x{w}"
            )
        x = self.conv_proj(x)                      # (n, hidden, h//p, w//p)
        x = x.reshape(n, self.hidden_dim, (h // p) * (w // p))
        return x.permute(0, 2, 1)                  # (n, seq, hidden)

    def forward(self, x: torch.Tensor, label: torch.Tensor | None = None):
        x = self._process_input(x)
        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        embeddings = x[:, 0]
        return self.heads(embeddings), embeddings


def inject_lora_into_mlp(model: nn.Module, rank: int = 8) -> nn.Module:
    """Replace every ``nn.Linear`` inside an encoder block's ``mlp`` with
    ``lora.Linear``, **copying the pretrained weights across**."""
    if rank <= 0:
        return model
    for module in model.modules():
        mlp = getattr(module, "mlp", None)
        if mlp is None or not isinstance(mlp, nn.Module):
            continue
        for child_name, child in list(mlp.named_children()):
            if not isinstance(child, nn.Linear) or isinstance(child, lora.Linear):
                continue
            new = lora.Linear(
                child.in_features,
                child.out_features,
                r=rank,
                bias=child.bias is not None,
            )
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            new.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(mlp, child_name, new)
    return model


def build_vit_b16(lora_rank: int = 8, pretrained: bool = True) -> ModifiedViT:
    """Build the ImageNet backbone.

    ``pretrained=False`` skips the torchvision weight download; the old code
    constructed ``vit_b_16(weights=IMAGENET1K_V1)`` *eagerly inside a dict
    literal* alongside the two face backbones, so every face run also tried to
    download a 330 MB ImageNet checkpoint.
    """
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    return inject_lora_into_mlp(ModifiedViT(vit_b_16(weights=weights)), rank=lora_rank)


# ----------------------------------------------------------------------
# Head remapping for the ImageNet-100 subset
# ----------------------------------------------------------------------
def remap_head(
    model: ModifiedViT,
    current_id_to_original_id: dict[int, int],
    device: torch.device,
    cache_path: str | Path = "results/original_VIT_head/classifier.pt",
) -> ModifiedViT:
    """Slice the 1000-way head down to the classes of the current subset.

    Mutates ``model`` in place and caches the full 1000-way head at
    ``cache_path`` so :func:`restore_head` can put it back later.
    """
    old = model.heads.head
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"weight": old.weight.detach().cpu(), "bias": old.bias.detach().cpu()},
            cache_path,
        )

    original_ids = list(current_id_to_original_id.values())
    new = nn.Linear(old.in_features, len(original_ids))
    with torch.no_grad():
        index = torch.as_tensor(original_ids, device=old.weight.device)
        new.weight.copy_(old.weight.index_select(0, index))
        new.bias.copy_(old.bias.index_select(0, index))
    model.heads.head = new
    return model.to(device)


def restore_head(
    model: ModifiedViT,
    device: torch.device,
    cache_path: str | Path = "results/original_VIT_head/classifier.pt",
) -> ModifiedViT:
    """Put the cached 1000-way ImageNet head back (for missing-class eval).

    Returns a *copy*; the caller's model keeps its subset head.
    """
    import copy

    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached ImageNet head at {cache_path}. remap_head() must run first."
        )
    state = torch.load(cache_path, map_location="cpu", weights_only=True)
    model = copy.deepcopy(model)
    head = nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
    with torch.no_grad():
        head.weight.copy_(state["weight"])
        head.bias.copy_(state["bias"])
    model.heads.head = head
    return model.to(device)


def load_imagenet_class_mapping(
    folder_names_path: str | Path, test_dataset
) -> dict[int, int]:
    """Map ``ImageFolder`` indices of the 100-class subset to ImageNet-1k ids."""
    folder_names_path = Path(folder_names_path)
    if not folder_names_path.exists():
        raise FileNotFoundError(
            f"Missing {folder_names_path}. Download imagenet_folder_names.txt "
            f"(see README) -- it defines the canonical 1000-class folder order."
        )
    names = [line.strip() for line in folder_names_path.read_text().splitlines() if line.strip()]
    unknown = set(test_dataset.classes) - set(names)
    if unknown:
        raise ValueError(f"Test classes absent from the 1k list: {sorted(unknown)[:5]}")
    global_index = {name: i for i, name in enumerate(names)}
    return {test_dataset.class_to_idx[c]: global_index[c] for c in test_dataset.classes}
