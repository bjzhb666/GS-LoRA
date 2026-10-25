"""Registry of forgetting methods.

``train_own_forget_cl.py`` dispatched with a 300-line
``if args.one_stage: ... elif args.retrain: ... elif args.LIRF: ...`` chain
over nine mutually-exclusive boolean flags, each of which also had to be
checked again when building meters, optimisers, teachers and checkpoints.
``--method`` plus this table replaces all of it.
"""

from __future__ import annotations

from .base import BaselineTrainer, StepOutput
from .distillation import DerTrainer, FdrTrainer, LwfTrainer, lwf_kd_loss
from .lirf import LirfTrainer
from .regularization import (
    RegularizationTrainer,
    RetrainTrainer,
    estimate_importance,
    snapshot_params,
)
from .scrub import ScrubTrainer, param_distance, sgda_adjust_learning_rate

__all__ = [
    "BaselineTrainer",
    "DerTrainer",
    "FdrTrainer",
    "LirfTrainer",
    "LwfTrainer",
    "RegularizationTrainer",
    "RetrainTrainer",
    "ScrubTrainer",
    "StepOutput",
    "TRAINERS",
    "estimate_importance",
    "lwf_kd_loss",
    "needs_teacher",
    "param_distance",
    "sgda_adjust_learning_rate",
    "snapshot_params",
]

#: ``--method`` -> trainer class. ``gslora`` is handled by
#: :class:`gslora.engine.gslora.GSLoRATrainer`, which has a different
#: signature (prototypes, group-lasso) and so is not in this table.
TRAINERS: dict[str, type[BaselineTrainer]] = {
    "retrain": RetrainTrainer,
    "l2": RegularizationTrainer,
    "ewc": RegularizationTrainer,
    "mas": RegularizationTrainer,
    "lirf": LirfTrainer,
    "scrub": ScrubTrainer,
    "lwf": LwfTrainer,
    "der": DerTrainer,
    "fdr": FdrTrainer,
}

#: methods that need a frozen copy of the pre-task model
TEACHER_METHODS = frozenset({"lwf", "der", "fdr", "scrub"})

#: methods whose objective is the quadratic weight anchor
ANCHOR_METHODS = frozenset({"l2", "ewc", "mas"})


def needs_teacher(method: str) -> bool:
    return method in TEACHER_METHODS
