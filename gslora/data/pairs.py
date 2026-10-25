"""LFW-style ``.bin`` verification pairs.

A ``.bin`` file is a pickled ``(encoded_jpegs, issame_list)``. Decoding is
done with Pillow, so no extra image backend is needed.

``load_bin`` returns ``([unflipped, flipped], issame_list)``; index 1 is the
horizontally mirrored copy, as the verification protocol expects.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

__all__ = ["load_bin", "get_val_pair", "get_val_data"]


def load_bin(path: str | Path, image_size: tuple[int, int] = (112, 112)) -> tuple[list[torch.Tensor], list[bool]]:
    """Decode a ``.bin`` pair file.

    Returns:
        ``([unflipped, flipped], issame_list)`` where each tensor is
        ``(2 * len(issame_list), 3, H, W)``.
    """
    path = Path(path)
    with path.open("rb") as fh:
        try:
            bins, issame_list = pickle.load(fh, encoding="bytes")
        except UnicodeDecodeError:  # pragma: no cover - very old pickles
            fh.seek(0)
            bins, issame_list = pickle.load(fh, encoding="latin1")

    n = len(issame_list) * 2
    h, w = image_size
    data_list = [torch.zeros((n, 3, h, w)) for _ in range(2)]

    for i in range(n):
        with Image.open(io.BytesIO(bins[i])) as handle:
            img = handle.convert("RGB")
        if img.size != (w, h):
            img = _resize_short(img, min(h, w))
        arr = np.asarray(img, dtype=np.uint8)                 # (H, W, 3)
        chw = torch.from_numpy(arr).permute(2, 0, 1).float()  # (3, H, W)
        data_list[0][i] = chw
        data_list[1][i] = torch.flip(chw, dims=[2])           # horizontal mirror
    return data_list, list(issame_list)


def _resize_short(img: Image.Image, size: int) -> Image.Image:
    """Scale so the shorter side equals ``size`` (``mx.image.resize_short``)."""
    w, h = img.size
    scale = size / min(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)


def get_val_pair(root: str | Path, name: str) -> tuple[list[torch.Tensor], list[bool]]:
    path = Path(root) / f"{name}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Verification set {name!r} not found at {path}")
    return load_bin(path)


def get_val_data(data_path: str | Path, targets: list[str]) -> list[tuple[str, list[torch.Tensor], list[bool]]]:
    if not targets:
        raise ValueError("targets must be non-empty")
    return [(t, *get_val_pair(data_path, t)) for t in targets]
