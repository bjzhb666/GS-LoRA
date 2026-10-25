"""Loss terms: values, gradients, and the edge cases they have to survive."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from gslora.losses import (
    DistillKL,
    attention_transfer_loss,
    bounded_forget_loss,
    elastic_reg_loss,
    group_sparse_loss,
    prototype_loss,
)
from gslora.methods.distillation import lwf_kd_loss
from gslora.metrics import harmonic_mean, topk_accuracy


# ----------------------------------------------------------------------
# group lasso
# ----------------------------------------------------------------------
def test_group_sparse_value_matches_sum_of_l2_norms():
    a = torch.tensor([[3.0, 4.0]])          # norm 5
    b = torch.tensor([[6.0, 8.0]])          # norm 10
    got = group_sparse_loss([[a], [b]]).item()
    assert math.isclose(got, 15.0, rel_tol=1e-5)


def test_group_sparse_pools_within_a_group():
    a = torch.tensor([3.0])
    b = torch.tensor([4.0])
    # one group of two tensors -> sqrt(9 + 16) = 5, not 3 + 4 = 7
    assert math.isclose(group_sparse_loss([[a, b]]).item(), 5.0, rel_tol=1e-5)


def test_group_sparse_gradient_is_finite_at_zero():
    """The gradient must stay finite at the origin.

    A group driven to exactly zero is the success case for group sparsity, and
    it is also where an unsmoothed sqrt has an infinite derivative.
    """
    a = torch.zeros(4, 8, requires_grad=True)
    b = torch.zeros(8, 4, requires_grad=True)
    loss = group_sparse_loss([[a, b]])
    loss.backward()
    assert torch.isfinite(loss), "loss must be finite"
    assert torch.isfinite(a.grad).all(), "lora_A gradient must not be NaN"
    assert torch.isfinite(b.grad).all(), "lora_B gradient must not be NaN"
    assert loss.item() < 1e-5, "value must still be ~0 at the origin"


def test_group_sparse_gradient_matches_analytic_away_from_zero():
    a = torch.tensor([3.0, 4.0], requires_grad=True)
    group_sparse_loss([[a]]).backward()
    # d||a||/da = a / ||a||
    torch.testing.assert_close(a.grad, torch.tensor([0.6, 0.8]), rtol=1e-4, atol=1e-6)


def test_group_sparse_rejects_empty_input():
    with pytest.raises(ValueError):
        group_sparse_loss([])
    with pytest.raises(ValueError):
        group_sparse_loss([[], []])


# ----------------------------------------------------------------------
# LwF knowledge distillation
# ----------------------------------------------------------------------
def test_lwf_kd_is_not_identically_zero():
    """The KD term must contribute both a value and a gradient."""
    torch.manual_seed(0)
    student = torch.randn(8, 20, requires_grad=True)
    teacher = torch.randn(8, 20)
    loss = lwf_kd_loss(student, teacher, temperature=2.0)
    loss.backward()
    assert loss.item() > 0.0
    assert student.grad.norm().item() > 0.0


def test_lwf_kd_is_zero_when_student_equals_teacher():
    logits = torch.randn(4, 7)
    assert lwf_kd_loss(logits, logits.clone(), 2.0).item() == pytest.approx(0.0, abs=1e-6)


def test_lwf_kd_is_nonnegative():
    torch.manual_seed(1)
    for _ in range(20):
        s, t = torch.randn(6, 11), torch.randn(6, 11)
        assert lwf_kd_loss(s, t, 2.0).item() >= -1e-6


# ----------------------------------------------------------------------
# prototype loss
# ----------------------------------------------------------------------
def test_prototype_loss_zero_at_the_prototype():
    protos = {0: torch.tensor([1.0, 2.0, 3.0]), 1: torch.tensor([0.0, 1.0, 0.0])}
    emb = torch.stack([protos[0], protos[1], protos[0]])
    labels = torch.tensor([0, 1, 0])
    assert prototype_loss(emb, labels, protos, "kl").item() == pytest.approx(0.0, abs=1e-6)
    assert prototype_loss(emb, labels, protos, "l2").item() == pytest.approx(0.0, abs=1e-6)


def test_prototype_loss_grows_with_distance():
    protos = {0: torch.zeros(5)}
    labels = torch.tensor([0, 0])
    near = prototype_loss(torch.zeros(2, 5) + 0.1, labels, protos, "l2")
    far = prototype_loss(torch.zeros(2, 5) + 5.0, labels, protos, "l2")
    assert far > near


def test_prototype_loss_reports_missing_class():
    with pytest.raises(KeyError, match="No prototype"):
        prototype_loss(torch.randn(2, 4), torch.tensor([0, 7]), {0: torch.zeros(4)})


def test_prototype_loss_rejects_unknown_distance():
    """An unsupported metric must raise, not return a non-tensor."""
    with pytest.raises(ValueError):
        prototype_loss(torch.randn(1, 3), torch.tensor([0]), {0: torch.zeros(3)}, "cosine")


# ----------------------------------------------------------------------
# misc
# ----------------------------------------------------------------------
def test_bounded_forget_loss_saturates():
    assert bounded_forget_loss(torch.tensor(2.0), bnd=10.0).item() == pytest.approx(8.0)
    assert bounded_forget_loss(torch.tensor(50.0), bnd=10.0).item() == 0.0


def test_distill_kl_zero_for_identical_logits():
    logits = torch.randn(4, 9)
    assert DistillKL(2.0)(logits, logits.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_attention_transfer_loss_zero_for_identical_features():
    feat = torch.randn(3, 8, 5)
    assert attention_transfer_loss(feat, feat.clone()).item() == pytest.approx(0.0, abs=1e-9)


def test_attention_transfer_keeps_the_graph_intact():
    """Masking small responses must not break the autograd graph."""
    student = torch.randn(3, 8, 5, requires_grad=True)
    attention_transfer_loss(student, torch.randn(3, 8, 5)).backward()
    assert student.grad is not None


def test_elastic_reg_loss_zero_at_the_anchor():
    model = torch.nn.Linear(3, 2)
    terms = {
        0: {
            "importance": {n: torch.ones_like(p) for n, p in model.named_parameters()},
            "task_param": {n: p.detach().clone() for n, p in model.named_parameters()},
        }
    }
    loss = elastic_reg_loss(model, terms, 1.0, torch.device("cpu"))
    assert loss.item() == pytest.approx(0.0, abs=1e-9)
    with torch.no_grad():
        model.weight += 1.0
    moved = elastic_reg_loss(model, terms, 1.0, torch.device("cpu"))
    assert moved.item() == pytest.approx(model.weight.numel(), rel=1e-5)


def test_elastic_reg_loss_handles_none():
    assert elastic_reg_loss(torch.nn.Linear(2, 2), None, 1.0, torch.device("cpu")).item() == 0.0


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def test_harmonic_mean_no_zero_division():
    """engine.py computed 2ab/(a+b) with no epsilon; a=b=0 is the state every
    run starts in."""
    assert harmonic_mean(0.0, 0.0) == 0.0


def test_harmonic_mean_clamps_negative_drop():
    # forget accuracy went up -> drop < 0; must not yield a high score
    assert harmonic_mean(-50.0, 90.0) == 0.0


def test_harmonic_mean_value():
    assert harmonic_mean(50.0, 50.0) == pytest.approx(50.0)
    assert harmonic_mean(100.0, 50.0) == pytest.approx(2 * 100 * 50 / 150)


def test_topk_accuracy():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([1, 0, 0])
    assert topk_accuracy(logits, labels, topk=1) == pytest.approx(200.0 / 3)
    assert topk_accuracy(logits, labels, topk=2) == pytest.approx(100.0)


def test_topk_accuracy_empty_batch():
    assert topk_accuracy(torch.zeros(0, 5), torch.zeros(0, dtype=torch.long)) == 0.0
