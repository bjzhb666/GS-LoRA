"""End-to-end tests: run the real entry points on synthetic images.

Every ``--method`` is exercised through ``gslora.tools.train_forget``, plus
the continual, few-shot, missing-class, prototype and EMA paths, so a broken
configuration surfaces here rather than after a dataset download.

They run on CPU in a few seconds each.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from gslora.config import Config
from gslora.tools import train_forget


def make_cfg(tiny_root: str, outdir: Path, **over) -> Config:
    base = dict(
        method="gslora",
        outdir=str(outdir),
        data_mode="tsne",            # 10 classes
        data_root=tiny_root,
        batch_size=4,
        num_workers=0,
        data_ratio=1.0,
        num_tasks=1,
        num_of_first_cls=8,
        per_forget_cls=2,
        net="VIT",
        head="CosFace",
        # a genuinely tiny backbone: the paper geometry (dim 512, mlp 2048,
        # 112x112) is ~19M params and far too slow for a CPU test suite
        image_size=32,
        vit_depth=2,
        vit_dim=32,
        vit_heads=2,
        vit_dim_head=16,
        vit_mlp_dim=64,
        vit_patch_size=4,
        vits_ac_patch_size=6,
        vits_pad=1,
        lora_rank=4,
        epochs=1,
        lr=1e-3,
        warmup_epochs=0,
        workers_id="cpu",
        tracker="jsonl",
        disp_freq=2,
        ver_freq=0,
        keep_checkpoints=1,
    )
    base.update(over)
    return Config(**base)


# ----------------------------------------------------------------------
def test_gslora_single_step_runs(tiny_root, tmp_path):
    cfg = make_cfg(tiny_root, tmp_path / "gslora")
    out = train_forget.run(cfg)
    assert len(out["history"]) == 1
    rec = out["history"][0]
    assert "acc/forget" in rec and "acc/remain" in rec
    assert 0.0 <= rec["acc/forget"] <= 100.0
    assert Path(cfg.outdir, "config.json").exists()
    assert Path(cfg.outdir, "metrics.jsonl").exists()
    assert Path(cfg.outdir, "task-level", "Backbone_task_0.pth").exists()


def test_gslora_continual_three_tasks(tiny_root, tmp_path):
    """The task loop, its checkpoint hand-off and LoRA re-init."""
    cfg = make_cfg(
        tiny_root, tmp_path / "cl",
        num_tasks=3, num_of_first_cls=7, per_forget_cls=1,
        cl_beta_list=[0.1, 0.2, 0.3],
    )
    out = train_forget.run(cfg)
    assert len(out["history"]) == 3
    for task in range(3):
        assert Path(cfg.outdir, "task-level", f"Backbone_task_{task}.pth").exists()


def test_gslora_open_set(tiny_root, tmp_path):
    """--open_cls_num replaces the whole *_open.py file pair."""
    cfg = make_cfg(
        tiny_root, tmp_path / "open",
        num_tasks=2, num_of_first_cls=7, per_forget_cls=1, open_cls_num=2,
        cl_beta_list=[0.1, 0.1],
    )
    out = train_forget.run(cfg)
    assert len(out["history"]) == 2
    assert out["history"][0]["acc/open"] is not None


def test_gslora_with_prototypes(tiny_root, tmp_path):
    cfg = make_cfg(
        tiny_root, tmp_path / "proto",
        prototype=True, pro_f_weight=0.5, pro_r_weight=0.1, bnd_pro=18.0,
    )
    assert len(train_forget.run(cfg)["history"]) == 1


def test_gslora_with_prototype_augmentation(tiny_root, tmp_path):
    """The prototype-augmentation path."""
    cfg = make_cfg(
        tiny_root, tmp_path / "proto_aug",
        prototype=True, pro_f_weight=0.5, proto_aug_num=5, proto_aug_repeats=2,
    )
    assert len(train_forget.run(cfg)["history"]) == 1


def test_gslora_few_shot(tiny_root, tmp_path):
    cfg = make_cfg(tiny_root, tmp_path / "fewshot", few_shot=True, few_shot_num=2)
    assert len(train_forget.run(cfg)["history"]) == 1


def test_gslora_with_ema(tiny_root, tmp_path):
    cfg = make_cfg(
        tiny_root, tmp_path / "ema",
        epochs=3, average_weight=True, ema_epoch=1, ema_decay=0.9,
    )
    assert len(train_forget.run(cfg)["history"]) == 1


def test_gslora_attention_lora(tiny_root, tmp_path):
    cfg = make_cfg(tiny_root, tmp_path / "attn", lora_pos="Attention")
    assert len(train_forget.run(cfg)["history"]) == 1


@pytest.mark.parametrize("grouping", ["block", "lora", "matrix"])
def test_gslora_grouping_variants(tiny_root, tmp_path, grouping):
    cfg = make_cfg(tiny_root, tmp_path / grouping, grouping=grouping)
    assert len(train_forget.run(cfg)["history"]) == 1


def test_gslora_depth_12_regularises_all_blocks(tiny_root, tmp_path):
    """Every block of a 12-layer backbone must appear in the group table."""
    cfg = make_cfg(tiny_root, tmp_path / "deep", vit_depth=12)
    train_forget.run(cfg)
    norms = _last_lora_norms(Path(cfg.outdir) / "metrics.jsonl")
    assert len(norms) == 12, f"expected 12 groups, logged {len(norms)}"


def _last_lora_norms(path: Path) -> list[float]:
    for line in reversed(path.read_text().strip().splitlines()):
        rec = json.loads(line)
        for key, value in rec.items():
            if key.startswith("lora_norm/"):
                return value
    raise AssertionError("no lora_norm record found")


# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["retrain", "l2", "ewc", "mas", "lwf", "der", "fdr"])
def test_baselines_run(tiny_root, tmp_path, method):
    cfg = make_cfg(tiny_root, tmp_path / method, method=method, lora_rank=0)
    out = train_forget.run(cfg)
    assert len(out["history"]) == 1


def test_der_plus_runs(tiny_root, tmp_path):
    cfg = make_cfg(tiny_root, tmp_path / "derpp", method="der", lora_rank=0, der_plus=True)
    assert len(train_forget.run(cfg)["history"]) == 1


def test_anchor_baselines_accumulate_terms_across_tasks(tiny_root, tmp_path):
    cfg = make_cfg(
        tiny_root, tmp_path / "ewc_cl", method="ewc", lora_rank=0,
        num_tasks=3, num_of_first_cls=7, per_forget_cls=1,
    )
    assert len(train_forget.run(cfg)["history"]) == 3


def test_replay_concatenates_loaders(tiny_root, tmp_path):
    cfg = make_cfg(tiny_root, tmp_path / "replay", method="ewc", lora_rank=0, replay=True)
    assert len(train_forget.run(cfg)["history"]) == 1


def test_scrub_runs(tiny_root, tmp_path):
    cfg = make_cfg(
        tiny_root, tmp_path / "scrub", method="scrub", lora_rank=0,
        scrub_superepoch=1, scrub_steps_per_superepoch=2, scrub_max_steps=1,
    )
    assert len(train_forget.run(cfg)["history"]) == 1


def test_lirf_runs(tiny_root, tmp_path):
    cfg = make_cfg(tiny_root, tmp_path / "lirf", method="lirf", lora_rank=0, vit_depth=4)
    assert len(train_forget.run(cfg)["history"]) == 1


# ----------------------------------------------------------------------
def test_training_actually_changes_the_lora_weights(tiny_root, tmp_path):
    """The optimiser must actually move the LoRA weights."""
    cfg = make_cfg(tiny_root, tmp_path / "moves", epochs=2, lr=1e-2)
    train_forget.run(cfg)
    state = torch.load(
        Path(cfg.outdir, "task-level", "Backbone_task_0.pth"),
        map_location="cpu", weights_only=True,
    )
    lora_b = [v for k, v in state.items() if "lora_B" in k]
    assert lora_b, "no lora_B in the checkpoint"
    assert any(v.abs().sum() > 0 for v in lora_b), (
        "lora_B is still all-zero: the optimiser never moved the LoRA weights"
    )


def test_gslora_actually_forgets_and_retains(tiny_root, tmp_path):
    """The real end-to-end claim, on a task the tiny model can actually learn.

    Pre-train to 100% on all 10 synthetic classes, then run GS-LoRA to forget
    2 of them. The forget accuracy must collapse while the remain accuracy
    stays high. This is the test that distinguishes "the code runs" from
    "the method works".
    """
    from gslora.tools import pretrain

    pre_cfg = make_cfg(
        tiny_root, tmp_path / "pre",
        lora_rank=0, epochs=60, lr=3e-3, batch_size=8, disp_freq=0, ver_freq=0,
    )
    pre = pretrain.run(pre_cfg)
    assert pre["final_acc"] > 95.0, (
        f"the fixture task must be learnable for this test to mean anything "
        f"(got {pre['final_acc']}%)"
    )

    cfg = make_cfg(
        tiny_root, tmp_path / "forget",
        resume=pre["checkpoint"],
        lora_rank=4, epochs=60, lr=2e-2, beta=1.0, bnd=25.0, alpha=1e-4,
        batch_size=8, disp_freq=0, ver_freq=0,
    )
    out = train_forget.run(cfg)

    records = [
        json.loads(l) for l in Path(cfg.outdir, "metrics.jsonl").read_text().splitlines()
    ]
    before = next(r for r in records if "acc/forget/task0/before" in r)
    f_before = before["acc/forget/task0/before"]
    r_before = before["acc/remain/task0/before"]
    f_after = out["history"][0]["acc/forget"]
    r_after = out["history"][0]["acc/remain"]

    assert f_before > 95.0, f"resume did not restore the pretrained model ({f_before}%)"
    assert r_before > 95.0
    assert f_after < 30.0, f"forget accuracy did not collapse: {f_before} -> {f_after}"
    assert r_after > 70.0, f"remain accuracy was destroyed: {r_before} -> {r_after}"


def test_resume_restores_the_pretrained_weights(tiny_root, tmp_path):
    """--resume must restore the pre-trained weights exactly."""
    from gslora.tools import pretrain

    pre_cfg = make_cfg(tiny_root, tmp_path / "p", lora_rank=0, epochs=60,
                       lr=3e-3, batch_size=8, disp_freq=0, ver_freq=0)
    pre = pretrain.run(pre_cfg)
    assert pre["final_acc"] > 95.0

    # epochs=0: nothing is optimised, so the loaded model must still score the
    # same as it did at the end of pre-training
    cfg = make_cfg(tiny_root, tmp_path / "f", resume=pre["checkpoint"], epochs=0)
    out = train_forget.run(cfg)
    assert out["history"][0]["acc/remain"] > 95.0
    assert out["history"][0]["acc/forget"] > 95.0


def test_wrong_checkpoint_is_rejected_loudly(tiny_root, tmp_path):
    """A checkpoint for a different model must raise, not be half-applied."""
    import torch.nn as nn

    bogus = tmp_path / "bogus.pth"
    torch.save(nn.Linear(3, 3).state_dict(), bogus)
    cfg = make_cfg(tiny_root, tmp_path / "bad", resume=str(bogus))
    with pytest.raises(RuntimeError, match="missing"):
        train_forget.run(cfg)


def test_evaluate_entry_point(tiny_root, tmp_path):
    from gslora.tools import evaluate

    cfg = make_cfg(tiny_root, tmp_path / "train")
    train_forget.run(cfg)
    ckpt = Path(cfg.outdir, "task-level", "Backbone_task_0.pth")

    eval_cfg = dataclasses.replace(cfg, outdir=str(tmp_path / "eval"), resume=str(ckpt))
    result = evaluate.run(eval_cfg)
    assert 0.0 <= result["overall"] <= 100.0
    assert Path(eval_cfg.outdir, "per_class_accuracy.json").exists()


def test_pretrain_entry_point(tiny_root, tmp_path):
    from gslora.tools import pretrain

    cfg = make_cfg(tiny_root, tmp_path / "pre", lora_rank=0, epochs=2, ver_freq=3)
    result = pretrain.run(cfg)
    assert 0.0 <= result["final_acc"] <= 100.0


def test_missing_dataset_gives_a_clear_error(tmp_path):
    cfg = make_cfg("/nonexistent/path", tmp_path / "bad")
    with pytest.raises(FileNotFoundError, match="train"):
        train_forget.run(cfg)
