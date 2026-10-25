"""One typed configuration object shared by every entry point.

:class:`Config` is built from the command line by :func:`parse_config`,
resolves the dataset and device, validates itself, and is then the only thing
passed around. ``--help`` on any of the three tools prints the full flag list.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

__all__ = ["Config", "DATASETS", "build_parser", "config_from_args", "parse_config"]


@dataclass(frozen=True)
class DatasetSpec:
    root: str
    num_classes: int
    input_size: tuple[int, int]
    normalize: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "casia100": DatasetSpec("./data/faces_webface_112x112_sub100_train_test/", 100, (112, 112)),
    "casia1000": DatasetSpec("./data/faces_webface_112x112_sub1000/", 1000, (112, 112)),
    "casia": DatasetSpec("./data/faces_webface_112x112/", 10572, (112, 112)),
    "tsne": DatasetSpec("./data/faces_Tsne_sub/", 10, (112, 112)),
    "imagenet100": DatasetSpec("./data/imagenet100/", 100, (224, 224), normalize=True),
}

# Methods that are mutually exclusive forgetting strategies.
METHODS = (
    "gslora",   # ours (one_stage)
    "retrain",
    "l2",
    "ewc",
    "mas",
    "lirf",
    "scrub",
    "lwf",
    "der",
    "fdr",
)


@dataclass
class Config:
    """Everything a run needs. Immutable in spirit; mutated only by the CLI."""

    # ---------------- experiment ----------------
    method: str = "gslora"
    outdir: str = "./exps/debug"
    resume: str = ""
    seed: int = 1337
    deterministic: bool = False
    run_name: str = ""

    # ---------------- data ----------------
    data_mode: str = "casia100"
    data_root: str = ""           # resolved from data_mode unless overridden
    batch_size: int = 256
    num_workers: int = 4
    data_ratio: float = 0.1
    few_shot: bool = False
    few_shot_num: int = 4

    # ---------------- task schedule ----------------
    num_tasks: int = 1
    num_of_first_cls: int = 90
    per_forget_cls: int = 10
    open_cls_num: int = 0          # 0 disables the missing-class (open) setting

    # ---------------- model ----------------
    net: Literal["VIT", "VITs", "VIT_B16"] = "VIT"
    head: Literal["Softmax", "ArcFace", "CosFace", "SFaceLoss"] = "CosFace"
    vit_depth: int = 6
    # Backbone geometry (defaults are the paper's Face Transformer).
    vit_dim: int = 512
    vit_heads: int = 8
    vit_dim_head: int = 64
    vit_mlp_dim: int = 2048
    vit_patch_size: int = 8
    vit_dropout: float = 0.1
    vits_ac_patch_size: int = 12
    vits_pad: int = 4
    embedding_size: int = 512
    lora_rank: int = 8
    lora_pos: Literal["FFN", "Attention"] = "FFN"
    grouping: Literal["block", "lora", "matrix"] = "block"

    # ---------------- optimisation ----------------
    epochs: int = 100
    opt: str = "adamw"
    opt_eps: float = 1e-8
    opt_betas: list[float] | None = None
    momentum: float = 0.9
    weight_decay: float = 0.05
    sched: str = "cosine"
    lr: float = 5e-4
    warmup_lr: float = 1e-6
    min_lr: float = 1e-5
    decay_epochs: int = 30
    warmup_epochs: int = 3
    cooldown_epochs: int = 10
    patience_epochs: int = 10
    decay_rate: float = 0.1

    # ---------------- GS-LoRA objective ----------------
    bnd: float = 10.0
    beta: float = 0.03
    alpha: float = 0.1
    cl_beta_list: list[float] = field(default_factory=list)
    warmup_alpha: bool = False
    big_alpha: float = 1e-4
    alpha_epoch: int = 0
    beta_decay: bool = False
    small_beta: float = 1e-4
    beta_decay_epoch: int = 50

    # ---------------- prototype regularisation (GS-LoRA++) ----------------
    prototype: bool = False
    pro_f_weight: float = 0.0
    pro_r_weight: float = 0.0
    cl_prof_list: list[float] = field(default_factory=list)
    bnd_pro: float = 18.0
    proto_aug_num: int = 0
    proto_aug_repeats: int = 20

    # ---------------- weight averaging ----------------
    average_weight: bool = False
    ema_decay: float = 0.99
    ema_epoch: int = 50

    # ---------------- baselines ----------------
    replay: bool = False
    online: bool = False
    ffn_open: bool = False
    only_ffn: bool = False
    l2_lambda: float = 0.1
    ewc_lambda: float = 0.1
    mas_lambda: float = 0.1
    n_fisher_sample: int | None = None
    lirf_T: float = 10.0
    lirf_alpha: float = 0.1
    lirf_w_at: float = 300.0
    lirf_w_kd: float = 10.0
    lirf_w_deposit: float = 0.05
    lirf_w_replay: float = 5.0
    scrub_superepoch: int = 10
    scrub_max_steps: int = 5
    scrub_steps_per_superepoch: int = 10
    sgda_smoothing: float = 0.0
    sgda_gamma: float = 0.99
    sgda_alpha: float = 0.001
    sgda_momentum: float = 0.9
    sgda_weight_decay: float = 5e-4
    scrub_decay_epoch: int = 100
    scrub_decay_rate: float = 0.1
    kd_T: float = 2.0
    lwf_T: float = 2.0
    lwf_lambda_kd: float = 0.5
    lwf_lambda_remain: float = 1.0
    der_lambda: float = 0.1
    der_plus: bool = False
    der_plus_lambda: float = 0.1
    fdr_lambda: float = 0.1

    # ---------------- logging / runtime ----------------
    workers_id: str = "cpu"
    disp_freq: int = 5
    ver_freq: int = 100
    keep_checkpoints: int = 2
    tracker: Literal["none", "jsonl", "wandb"] = "jsonl"
    wandb_project: str = "gslora"
    wandb_group: str | None = None
    wandb_offline: bool = False

    # ---------------- derived (filled by __post_init__) ----------------
    gpu_ids: list[int] = field(default_factory=list)
    device: torch.device = field(default=torch.device("cpu"))
    multi_gpu: bool = False
    num_classes: int = 100
    #: (0, 0) means "take it from the dataset spec"; --image_size overrides it
    input_size: tuple[int, int] = (0, 0)
    image_size: int = 0
    normalize_input: bool = False

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.data_mode not in DATASETS:
            raise ValueError(
                f"Unknown --data_mode {self.data_mode!r}. Known: {sorted(DATASETS)}"
            )
        spec = DATASETS[self.data_mode]
        if not self.data_root:
            self.data_root = spec.root
        self.num_classes = spec.num_classes
        self.normalize_input = spec.normalize
        if self.image_size:
            self.input_size = (self.image_size, self.image_size)
        elif self.input_size == (0, 0):
            self.input_size = spec.input_size
        self.input_size = tuple(self.input_size)

        # device
        if self.workers_id in ("cpu", "") or not torch.cuda.is_available():
            self.gpu_ids = []
            self.device = torch.device("cpu")
            self.multi_gpu = False
        else:
            self.gpu_ids = [int(i) for i in str(self.workers_id).split(",") if i != ""]
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            self.multi_gpu = len(self.gpu_ids) > 1

        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Checks that apply to every entry point.

        The forgetting *schedule* is checked separately by
        :meth:`validate_schedule`, because ``pretrain`` and ``evaluate`` do not
        use ``--num_of_first_cls`` / ``--per_forget_cls`` at all.
        """
        if self.method not in METHODS:
            raise ValueError(f"--method must be one of {METHODS}, got {self.method!r}")

        if self.num_tasks < 1:
            raise ValueError("--num_tasks must be >= 1")

        if self.net == "VIT_B16" and self.data_mode != "imagenet100":
            raise ValueError("--net VIT_B16 is only wired up for --data_mode imagenet100")
        if self.lora_pos == "Attention" and self.net == "VIT_B16":
            raise ValueError("--lora_pos Attention is not supported for VIT_B16")

        if self.prototype and self.pro_f_weight == 0 and self.pro_r_weight == 0 and not self.cl_prof_list:
            raise ValueError(
                "--prototype was set but both prototype weights are 0; either "
                "set --pro_f_weight/--pro_r_weight or drop --prototype"
            )

    def validate_schedule(self) -> None:
        """Check the continual-forgetting class schedule.

        Called by the forgetting entry point only; ``pretrain`` and
        ``evaluate`` do not use the task schedule. Fails fast so a bad
        schedule is reported before any data is loaded.
        """
        # default the per-task beta schedule to a constant
        if self.method == "gslora":
            if not self.cl_beta_list:
                self.cl_beta_list = [self.beta] * self.num_tasks
            elif len(self.cl_beta_list) < self.num_tasks:
                raise ValueError(
                    f"--cl_beta_list has {len(self.cl_beta_list)} entries but "
                    f"--num_tasks is {self.num_tasks}"
                )
        if self.cl_prof_list and len(self.cl_prof_list) < self.num_tasks:
            raise ValueError(
                f"--cl_prof_list has {len(self.cl_prof_list)} entries but "
                f"--num_tasks is {self.num_tasks}"
            )

        last_known = self.num_of_first_cls - (self.num_tasks - 1) * self.per_forget_cls
        if last_known - self.open_cls_num <= 0:
            raise ValueError(
                f"Class schedule exhausts the dataset: num_of_first_cls="
                f"{self.num_of_first_cls} - ({self.num_tasks}-1)*{self.per_forget_cls} "
                f"- open_cls_num={self.open_cls_num} = "
                f"{last_known - self.open_cls_num} <= 0"
            )
        if self.num_of_first_cls + self.per_forget_cls > self.num_classes:
            raise ValueError(
                f"num_of_first_cls + per_forget_cls = "
                f"{self.num_of_first_cls + self.per_forget_cls} exceeds the "
                f"{self.num_classes} classes in {self.data_mode}"
            )

    # ------------------------------------------------------------------
    @property
    def num_lora_layers(self) -> int:
        """How many transformer blocks actually carry LoRA.

        For ``VIT_B16`` this is the torchvision ViT-B/16 depth (12), *not*
        ``--vit_depth``. The old code passed ``--vit_depth 6`` for ImageNet runs
        and got away with it only because ``cal_norm`` and ``engine_cl`` each
        hardcoded 12 in their ImageNet branch.
        """
        return 12 if self.net == "VIT_B16" else self.vit_depth

    @property
    def torchvision_vit(self) -> bool:
        return self.net == "VIT_B16"

    @property
    def open_set(self) -> bool:
        return self.open_cls_num > 0

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, torch.device) else value
        return out

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else Path(self.outdir) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path


# ----------------------------------------------------------------------
def build_parser(description: str = "GS-LoRA continual forgetting") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)

    g = p.add_argument_group("experiment")
    g.add_argument("--method", default="gslora", choices=METHODS)
    g.add_argument("--outdir", default="./exps/debug", help="output directory")
    g.add_argument("-r", "--resume", default="", help="checkpoint to start from")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--deterministic", action="store_true")
    g.add_argument("--run_name", default="")

    g = p.add_argument_group("data")
    g.add_argument("-d", "--data_mode", default="casia100", choices=sorted(DATASETS))
    g.add_argument("--data_root", default="", help="override the dataset root")
    g.add_argument("-b", "--batch_size", type=int, default=256)
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--data_ratio", type=float, default=0.1)
    g.add_argument("--few_shot", action="store_true")
    g.add_argument("--few_shot_num", type=int, default=4)

    g = p.add_argument_group("task schedule")
    g.add_argument("--num_tasks", type=int, default=1)
    g.add_argument("--num_of_first_cls", type=int, default=90)
    g.add_argument("--per_forget_cls", type=int, default=10)
    g.add_argument("--open_cls_num", type=int, default=0,
                   help="missing-class setting: N classes are held out of replay (0 = off)")

    g = p.add_argument_group("model")
    g.add_argument("-n", "--net", default="VIT", choices=["VIT", "VITs", "VIT_B16"])
    g.add_argument("-head", "--head", default="CosFace",
                   choices=["Softmax", "ArcFace", "CosFace", "SFaceLoss"])
    g.add_argument("--vit_depth", type=int, default=6)
    g.add_argument("--image_size", type=int, default=0,
                   help="override the dataset's input resolution (0 = dataset default)")
    g.add_argument("--vit_dim", type=int, default=512)
    g.add_argument("--vit_heads", type=int, default=8)
    g.add_argument("--vit_dim_head", type=int, default=64)
    g.add_argument("--vit_mlp_dim", type=int, default=2048)
    g.add_argument("--vit_patch_size", type=int, default=8)
    g.add_argument("--vit_dropout", type=float, default=0.1)
    g.add_argument("--vits_ac_patch_size", type=int, default=12)
    g.add_argument("--vits_pad", type=int, default=4)
    g.add_argument("--embedding_size", type=int, default=512)
    g.add_argument("--lora_rank", type=int, default=8)
    g.add_argument("--lora_pos", default="FFN", choices=["FFN", "Attention"])
    g.add_argument("--grouping", default="block", choices=["block", "lora", "matrix"])

    g = p.add_argument_group("optimisation")
    g.add_argument("-e", "--epochs", type=int, default=100)
    g.add_argument("--opt", default="adamw")
    g.add_argument("--opt-eps", dest="opt_eps", type=float, default=1e-8)
    g.add_argument("--opt-betas", dest="opt_betas", type=float, nargs="+", default=None)
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.05)
    g.add_argument("--sched", default="cosine")
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--warmup-lr", dest="warmup_lr", type=float, default=1e-6)
    g.add_argument("--min-lr", dest="min_lr", type=float, default=1e-5)
    g.add_argument("--decay-epochs", dest="decay_epochs", type=int, default=30)
    g.add_argument("--warmup-epochs", dest="warmup_epochs", type=int, default=3)
    g.add_argument("--cooldown-epochs", dest="cooldown_epochs", type=int, default=10)
    g.add_argument("--patience-epochs", dest="patience_epochs", type=int, default=10)
    g.add_argument("--decay-rate", "--dr", dest="decay_rate", type=float, default=0.1)

    g = p.add_argument_group("GS-LoRA objective")
    g.add_argument("--BND", dest="bnd", type=float, default=10.0)
    g.add_argument("--beta", type=float, default=0.03)
    g.add_argument("--alpha", type=float, default=0.1)
    g.add_argument("--cl_beta_list", nargs="*", type=float, default=[])
    g.add_argument("--warmup_alpha", action="store_true")
    g.add_argument("--big_alpha", type=float, default=1e-4)
    g.add_argument("--alpha_epoch", type=int, default=0,
                   help="epochs before the structure loss is switched on (0 = from the start)")
    g.add_argument("--beta_decay", action="store_true")
    g.add_argument("--small_beta", type=float, default=1e-4)
    g.add_argument("--beta_decay_epoch", type=int, default=50)

    g = p.add_argument_group("prototype regularisation")
    g.add_argument("--prototype", action="store_true")
    g.add_argument("--pro_f_weight", type=float, default=0.0)
    g.add_argument("--pro_r_weight", type=float, default=0.0)
    g.add_argument("--cl_prof_list", nargs="*", type=float, default=[])
    g.add_argument("--BND_pro", dest="bnd_pro", type=float, default=18.0)
    g.add_argument("--proto_aug_num", "--aug_num", dest="proto_aug_num", type=int, default=0)
    g.add_argument("--proto_aug_repeats", type=int, default=20)

    g = p.add_argument_group("weight averaging")
    g.add_argument("--average_weight", action="store_true")
    g.add_argument("--ema_decay", type=float, default=0.99)
    g.add_argument("--ema_epoch", type=int, default=50)

    g = p.add_argument_group("baselines")
    g.add_argument("--replay", action="store_true")
    g.add_argument("--online", action="store_true")
    g.add_argument("--ffn_open", action="store_true")
    g.add_argument("--only_ffn", action="store_true")
    g.add_argument("--l2_lambda", type=float, default=0.1)
    g.add_argument("--ewc_lambda", type=float, default=0.1)
    g.add_argument("--mas_lambda", type=float, default=0.1)
    g.add_argument("--n_fisher_sample", type=int, default=None)
    g.add_argument("--LIRF_T", dest="lirf_T", type=float, default=10.0)
    g.add_argument("--LIRF_alpha", dest="lirf_alpha", type=float, default=0.1)
    g.add_argument("--lirf_w_at", type=float, default=300.0)
    g.add_argument("--lirf_w_kd", type=float, default=10.0)
    g.add_argument("--lirf_w_deposit", type=float, default=0.05)
    g.add_argument("--lirf_w_replay", type=float, default=5.0)
    g.add_argument("--SCRUB_superepoch", dest="scrub_superepoch", type=int, default=10)
    g.add_argument("--scrub_max_steps", type=int, default=5)
    g.add_argument("--scrub_steps_per_superepoch", type=int, default=10)
    g.add_argument("--sgda_smoothing", type=float, default=0.0)
    g.add_argument("--sgda_gamma", type=float, default=0.99)
    g.add_argument("--sgda_alpha", type=float, default=0.001)
    g.add_argument("--sgda_momentum", type=float, default=0.9)
    g.add_argument("--sgda_weight_decay", type=float, default=5e-4)
    g.add_argument("--scrub_decay_epoch", type=int, default=100)
    g.add_argument("--scrub_decay_rate", type=float, default=0.1)
    g.add_argument("--kd_T", type=float, default=2.0)
    g.add_argument("--Lwf_T", dest="lwf_T", type=float, default=2.0)
    g.add_argument("--Lwf_lambda_kd", dest="lwf_lambda_kd", type=float, default=0.5)
    g.add_argument("--Lwf_lambda_remain", dest="lwf_lambda_remain", type=float, default=1.0)
    g.add_argument("--DER_lambda", dest="der_lambda", type=float, default=0.1)
    g.add_argument("--DER_plus", dest="der_plus", action="store_true")
    g.add_argument("--DER_plus_lambda", dest="der_plus_lambda", type=float, default=0.1)
    g.add_argument("--FDR_lambda", dest="fdr_lambda", type=float, default=0.1)

    g = p.add_argument_group("runtime / logging")
    g.add_argument("-w", "--workers_id", default="cpu", help="'cpu' or comma-separated GPU ids")
    g.add_argument("--disp_freq", type=int, default=5)
    g.add_argument("--ver_freq", type=int, default=100)
    g.add_argument("--keep_checkpoints", type=int, default=2)
    g.add_argument("--tracker", default="jsonl", choices=["none", "jsonl", "wandb"])
    g.add_argument("--wandb_project", default="gslora")
    g.add_argument("--wandb_group", default=None)
    g.add_argument("--wandb_offline", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    known = {f.name for f in dataclasses.fields(Config)}
    kwargs = {k: v for k, v in vars(args).items() if k in known}
    return Config(**kwargs)


def parse_config(argv: list[str] | None = None, **parser_kwargs) -> Config:
    return config_from_args(build_parser(**parser_kwargs).parse_args(argv))


# ----------------------------------------------------------------------
def timm_args(cfg: Config) -> argparse.Namespace:
    """Adapter for timm's args-protocol optimiser/scheduler factories.

    timm's ``create_optimizer(args, model)`` and ``create_scheduler(args, opt)``
    read a fixed set of attribute names off a namespace. Keeping that coupling
    in exactly one place means the rest of the codebase talks to :class:`Config`
    and a timm upgrade only touches this function.
    """
    return argparse.Namespace(
        opt=cfg.opt,
        opt_eps=cfg.opt_eps,
        opt_betas=cfg.opt_betas,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        lr=cfg.lr,
        sched=cfg.sched,
        epochs=cfg.epochs,
        warmup_lr=cfg.warmup_lr,
        min_lr=cfg.min_lr,
        decay_epochs=cfg.decay_epochs,
        warmup_epochs=cfg.warmup_epochs,
        cooldown_epochs=cfg.cooldown_epochs,
        patience_epochs=cfg.patience_epochs,
        decay_rate=cfg.decay_rate,
        # timm reads these for the noisy-LR option; we never enable it.
        lr_noise=None,
        lr_noise_pct=0.67,
        lr_noise_std=1.0,
        seed=cfg.seed,
        lr_cycle_mul=1.0,
        lr_cycle_decay=0.5,
        lr_cycle_limit=1,
        lr_k_decay=1.0,
    )
