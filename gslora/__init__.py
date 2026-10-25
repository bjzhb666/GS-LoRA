"""GS-LoRA / GS-LoRA++: practical continual forgetting for pre-trained vision models.

Reference implementation for

* *Continual Forgetting for Pre-trained Vision Models* (CVPR 2024)
* *Practical Continual Forgetting for Pre-trained Vision Models* (T-PAMI 2026)

Project page: https://bjzhb666.github.io/GS-LoRA/
"""

from __future__ import annotations

__version__ = "0.2.0"

from .config import Config, build_parser, config_from_args, parse_config
from .metrics import AverageMeter, harmonic_mean, topk_accuracy

__all__ = [
    "AverageMeter",
    "Config",
    "__version__",
    "build_parser",
    "config_from_args",
    "harmonic_mean",
    "parse_config",
    "topk_accuracy",
]
