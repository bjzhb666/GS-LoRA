"""Config validation -- failures that used to surface mid-training."""

from __future__ import annotations

import json

import pytest

from gslora.config import Config, parse_config, timm_args


def base(**over) -> dict:
    d = dict(
        data_mode="casia100", num_tasks=1, num_of_first_cls=90,
        per_forget_cls=10, net="VIT", workers_id="cpu",
    )
    d.update(over)
    return d


def test_dataset_resolution():
    cfg = Config(**base())
    assert cfg.num_classes == 100
    assert cfg.input_size == (112, 112)
    assert cfg.data_root.endswith("faces_webface_112x112_sub100_train_test/")
    assert cfg.normalize_input is False


def test_imagenet_normalises():
    cfg = Config(**base(data_mode="imagenet100", net="VIT_B16"))
    assert cfg.normalize_input is True
    assert cfg.input_size == (224, 224)


def test_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown --data_mode"):
        Config(**base(data_mode="nope"))


def test_cl_beta_list_is_defaulted_not_indexerror():
    """An unset per-task beta schedule defaults to a constant."""
    cfg = Config(**base(num_tasks=5, beta=0.07))
    cfg.validate_schedule()
    assert cfg.cl_beta_list == [0.07] * 5


def test_cl_beta_list_length_is_checked():
    with pytest.raises(ValueError, match="cl_beta_list"):
        Config(**base(num_tasks=5, cl_beta_list=[0.1, 0.2])).validate_schedule()


def test_schedule_must_fit_the_dataset():
    """A schedule that does not fit the dataset must be rejected up front."""
    with pytest.raises(ValueError, match="exhausts the dataset"):
        Config(**base(num_tasks=12, num_of_first_cls=90, per_forget_cls=10)).validate_schedule()
    with pytest.raises(ValueError, match="exceeds the"):
        Config(**base(num_of_first_cls=95, per_forget_cls=10)).validate_schedule()


def test_nine_task_schedule_is_valid():
    cfg = Config(**base(num_tasks=9, num_of_first_cls=90, per_forget_cls=10))
    cfg.validate_schedule()
    assert cfg.num_tasks == 9


def test_pretrain_does_not_need_a_forgetting_schedule():
    """pretrain/evaluate must not be blocked by --num_of_first_cls defaults
    that make no sense for a 10-class dataset."""
    cfg = Config(**base(data_mode="tsne"))      # 10 classes, defaults say 90+10
    assert cfg.num_classes == 10                # construction succeeds
    with pytest.raises(ValueError):
        cfg.validate_schedule()                 # only the forgetting path objects


def test_image_size_override():
    cfg = Config(**base(image_size=32))
    assert cfg.input_size == (32, 32)
    assert Config(**base()).input_size == (112, 112)


def test_num_lora_layers_tracks_the_backbone():
    """The LoRA layer count follows the backbone, not --vit_depth."""
    assert Config(**base(net="VIT", vit_depth=6)).num_lora_layers == 6
    assert Config(**base(net="VIT", vit_depth=12)).num_lora_layers == 12
    cfg = Config(**base(data_mode="imagenet100", net="VIT_B16", vit_depth=6))
    assert cfg.num_lora_layers == 12
    assert cfg.torchvision_vit is True


def test_vit_b16_requires_imagenet():
    with pytest.raises(ValueError, match="VIT_B16"):
        Config(**base(net="VIT_B16", data_mode="casia100"))


def test_attention_lora_rejected_for_vit_b16():
    with pytest.raises(ValueError, match="Attention"):
        Config(**base(net="VIT_B16", data_mode="imagenet100", lora_pos="Attention"))


def test_prototype_without_weights_is_rejected():
    with pytest.raises(ValueError, match="prototype"):
        Config(**base(prototype=True))
    Config(**base(prototype=True, pro_f_weight=0.5))  # fine


def test_open_set_flag():
    assert Config(**base()).open_set is False
    assert Config(**base(open_cls_num=10)).open_set is True


def test_roundtrip_to_disk(tmp_path):
    cfg = Config(**base(outdir=str(tmp_path)))
    path = cfg.save()
    loaded = json.loads(path.read_text())
    assert loaded["num_of_first_cls"] == 90
    assert loaded["device"] == "cpu"


def test_cli_maps_legacy_flag_names():
    """The old scripts' flag spellings must keep working."""
    cfg = parse_config(
        [
            "-d", "casia100", "-n", "VIT", "-head", "CosFace", "-b", "48",
            "-e", "100", "--lora_rank", "8", "--BND", "110", "--beta", "0.15",
            "--alpha", "0.01", "--BND_pro", "50", "--aug_num", "5",
            "--weight-decay", "0.01", "--decay-epochs", "100",
            "--min-lr", "1e-5", "--warmup-epochs", "0",
            "--num_of_first_cls", "90", "--per_forget_cls", "10",
        ]
    )
    assert (cfg.bnd, cfg.beta, cfg.alpha, cfg.bnd_pro) == (110.0, 0.15, 0.01, 50.0)
    assert cfg.proto_aug_num == 5
    assert cfg.weight_decay == 0.01
    assert cfg.decay_epochs == 100
    assert cfg.warmup_epochs == 0


def test_timm_adapter_has_every_attribute_timm_reads():
    from timm.optim import create_optimizer
    from timm.scheduler import create_scheduler
    import torch.nn as nn

    cfg = Config(**base(opt="adamw", lr=1e-3, sched="cosine", epochs=5))
    targs = timm_args(cfg)
    opt = create_optimizer(targs, nn.Linear(4, 4))
    sched, _ = create_scheduler(targs, opt)
    sched.step(0)
    assert opt.param_groups[0]["lr"] > 0
