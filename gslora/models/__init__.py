from .factory import build_backbone, build_lirf_models, load_backbone_weights
from .heads import ArcFace, CosFace, SFaceLoss, Softmax, build_head
from .transformer import Attention, FeedForward, PreNorm, Residual, Transformer
from .vit_face import ViT_face, ViT_face_low, ViT_face_up, ViTs_face

__all__ = [
    "ArcFace",
    "Attention",
    "CosFace",
    "FeedForward",
    "PreNorm",
    "Residual",
    "SFaceLoss",
    "Softmax",
    "Transformer",
    "ViT_face",
    "ViT_face_low",
    "ViT_face_up",
    "ViTs_face",
    "build_backbone",
    "build_head",
    "build_lirf_models",
    "load_backbone_weights",
]


def __getattr__(name: str):
    # torchvision_vit pulls in torchvision.models; keep it lazy so face-only
    # runs never touch it.
    if name in ("ModifiedViT", "build_vit_b16", "inject_lora_into_mlp", "remap_head", "restore_head"):
        from . import torchvision_vit

        return getattr(torchvision_vit, name)
    raise AttributeError(name)
