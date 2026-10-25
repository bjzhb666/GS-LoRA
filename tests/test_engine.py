"""Engine pieces: paired iteration, evaluation, checkpoint rotation, EMA."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gslora.checkpoint import rotate_checkpoints, save_checkpoint
from gslora.config import Config
from gslora.ema import ModelEma
from gslora.engine.evaluator import Evaluator, default_forward, evaluate_accuracy
from gslora.engine.pairing import PairedBatches, cycle
from gslora.prototypes import eval_mode
from gslora.tracking import JsonlTracker, NullTracker


def _loader(n: int, batch_size: int, label: int = 0) -> DataLoader:
    x = torch.randn(n, 3, 8, 8)
    y = torch.full((n,), label, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


# ----------------------------------------------------------------------
# PairedBatches
# ----------------------------------------------------------------------
def test_paired_length_is_the_longer_loader():
    """An epoch is one pass over the longer loader, so neither side is cut."""
    pairs = PairedBatches(_loader(10, 2), _loader(40, 2), torch.device("cpu"))
    assert len(pairs) == 20
    assert sum(1 for _ in pairs) == 20


def test_paired_cycles_the_short_side_without_dropping_the_long_one():
    forget, remain = _loader(4, 2), _loader(20, 2)   # 2 vs 10 batches
    pairs = PairedBatches(forget, remain, torch.device("cpu"))
    seen_remain = 0
    for (xf, _), (xr, _) in pairs:
        assert xf.size(0) > 0 and xr.size(0) > 0
        seen_remain += xr.size(0)
    assert seen_remain == 20, "every remain sample must be visited once"


def test_paired_works_when_forget_is_longer():
    pairs = PairedBatches(_loader(40, 2), _loader(4, 2), torch.device("cpu"))
    seen_forget = sum(xf.size(0) for (xf, _), _ in pairs)
    assert seen_forget == 40


def test_paired_rejects_empty_loader():
    with pytest.raises(ValueError, match="non-empty"):
        PairedBatches(_loader(0, 2), _loader(4, 2), torch.device("cpu"))


def test_paired_yields_float_and_long():
    (xf, yf), _ = next(iter(PairedBatches(_loader(4, 2), _loader(4, 2), torch.device("cpu"))))
    assert xf.dtype == torch.float32
    assert yf.dtype == torch.int64


def test_cycle_raises_on_empty():
    with pytest.raises(ValueError):
        next(cycle(_loader(0, 2)))


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
class _Const(nn.Module):
    """Always predicts `target`."""

    def __init__(self, n_classes: int, target: int) -> None:
        super().__init__()
        self.n_classes, self.target = n_classes, target
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, label=None):
        logits = torch.full((x.size(0), self.n_classes), -10.0)
        logits[:, self.target] = 10.0
        return logits, torch.zeros(x.size(0), 4)


def test_evaluate_accuracy():
    model = _Const(5, target=3)
    assert evaluate_accuracy(default_forward(model), _loader(8, 4, label=3), torch.device("cpu")) == 100.0
    assert evaluate_accuracy(default_forward(model), _loader(8, 4, label=1), torch.device("cpu")) == 0.0


def test_evaluate_accuracy_empty_loader_is_zero_not_zerodivision():
    model = _Const(5, 0)
    assert evaluate_accuracy(default_forward(model), _loader(0, 4), torch.device("cpu")) == 0.0


def test_evaluator_computes_hmean_and_keeps_best(tmp_path):
    cfg = Config(
        data_mode="casia100", outdir=str(tmp_path), workers_id="cpu",
        net="VIT", keep_checkpoints=2,
    )
    model = _Const(5, target=3)
    ev = Evaluator(
        cfg, NullTracker(),
        forget_loader=_loader(8, 4, label=1),   # model gets these wrong -> 0%
        remain_loader=_loader(8, 4, label=3),   # and these right     -> 100%
        forget_acc_before=100.0,
    )
    res = ev.run(model, epoch=0, batch=0)
    assert res.forget_acc == 0.0
    assert res.remain_acc == 100.0
    assert res.forget_drop == 100.0
    assert res.h_mean == pytest.approx(100.0)
    assert ev.best_h_mean == pytest.approx(100.0)
    assert ev.best_checkpoint is not None


def test_eval_mode_restores_training_flag():
    """Evaluating must not leave the model in eval(): for loralib that would
    keep the LoRA deltas merged into the frozen weights."""
    model = nn.Linear(2, 2).train()
    with eval_mode(model):
        assert not model.training
    assert model.training
    model.eval()
    with eval_mode(model):
        pass
    assert not model.training


# ----------------------------------------------------------------------
# checkpoint rotation
# ----------------------------------------------------------------------
def test_rotation_counts_checkpoints_not_directory_entries(tmp_path):
    """Retention must count checkpoints, not unrelated files in the directory."""
    (tmp_path / "config.txt").write_text("x")
    (tmp_path / "metrics.jsonl").write_text("x")
    (tmp_path / "task-level").mkdir()

    model = nn.Linear(2, 2)
    for i in range(5):
        save_checkpoint(model, tmp_path, backbone_name="VIT", epoch=i, batch=i, keep=2)
        import time

        time.sleep(0.01)  # distinct mtimes

    kept = sorted(tmp_path.glob("Backbone_*_checkpoint.pth"))
    assert len(kept) == 2, f"expected 2 checkpoints, found {[p.name for p in kept]}"
    assert (tmp_path / "config.txt").exists(), "unrelated files must survive"
    assert (tmp_path / "task-level").is_dir()


def test_rotation_keeps_the_newest(tmp_path):
    import time

    model = nn.Linear(2, 2)
    paths = []
    for i in range(4):
        paths.append(save_checkpoint(model, tmp_path, backbone_name="VIT", epoch=i, batch=i, keep=1))
        time.sleep(0.01)
    assert paths[-1].exists()
    assert not paths[0].exists()


def test_rotation_rejects_keep_zero(tmp_path):
    with pytest.raises(ValueError):
        rotate_checkpoints(tmp_path, keep=0)


# ----------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------
def test_ema_tracks_then_lags_the_model():
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ModelEma(model, decay=0.5)

    ema.step(model, epoch=5, start_epoch=5)              # hard copy
    assert torch.all(ema.module.weight == 0.0)

    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.step(model, epoch=6, start_epoch=5)              # 0.5*0 + 0.5*1
    assert torch.allclose(ema.module.weight, torch.full((2, 3), 0.5))

    ema.step(model, epoch=7, start_epoch=5)              # 0.5*0.5 + 0.5*1
    assert torch.allclose(ema.module.weight, torch.full((2, 3), 0.75))


def test_ema_is_inactive_before_start_epoch():
    model = nn.Linear(2, 2)
    ema = ModelEma(model, decay=0.9)
    assert ema.step(model, epoch=0, start_epoch=10) is False


def test_ema_does_not_disturb_the_live_model_mode(tiny_model):
    tiny_model.train()
    ema = ModelEma(tiny_model, decay=0.9)
    ema.step(tiny_model, epoch=0, start_epoch=0)
    ema.step(tiny_model, epoch=1, start_epoch=0)
    assert tiny_model.training, "the live model must still be in train mode"


def test_ema_rejects_bad_decay():
    with pytest.raises(ValueError):
        ModelEma(nn.Linear(2, 2), decay=1.0)


# ----------------------------------------------------------------------
# tracking
# ----------------------------------------------------------------------
def test_jsonl_tracker_writes_records(tmp_path):
    t = JsonlTracker(tmp_path)
    t.log({"loss": torch.tensor(1.5), "acc": 90.0}, step=3)
    t.log({"norms": [torch.tensor(1.0), torch.tensor(2.0)]})
    t.finish()
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    import json

    first = json.loads(lines[0])
    assert first["loss"] == 1.5 and first["_step"] == 3
    assert json.loads(lines[1])["norms"] == [1.0, 2.0]
