import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.train_pinn import (  # noqa: E402
    _compute_model_selection_score,
    compute_batch_loss,
    evaluate_anchor_metrics,
)


def test_compute_batch_loss_gate_term_tracks_gate_logits():
    total_costs = torch.tensor([[5.0], [5.0]], dtype=torch.float32)
    aux_pred = torch.zeros(2, 6, dtype=torch.float32)
    residuals = torch.zeros(2, 6, dtype=torch.float32)
    gt_labels = torch.tensor([1, 0], dtype=torch.int64)
    group_slices = [(0, 2)]

    good_gate_logits = torch.tensor([[6.0], [-6.0]], dtype=torch.float32)
    bad_gate_logits = -good_gate_logits

    _, good_stats = compute_batch_loss(
        total_costs=total_costs,
        aux_pred=aux_pred,
        residuals=residuals,
        gt_labels=gt_labels,
        group_slices=group_slices,
        gate_logits=good_gate_logits,
        alpha=0.0,
        beta=0.0,
        gamma=1.0,
        margin=1.0,
        temperature=1.0,
    )
    _, bad_stats = compute_batch_loss(
        total_costs=total_costs,
        aux_pred=aux_pred,
        residuals=residuals,
        gt_labels=gt_labels,
        group_slices=group_slices,
        gate_logits=bad_gate_logits,
        alpha=0.0,
        beta=0.0,
        gamma=1.0,
        margin=1.0,
        temperature=1.0,
    )

    assert good_stats.gate_term < bad_stats.gate_term


class _FakePINN:
    def eval(self):
        return self

    def __call__(self, pa, ma, pb, mb, ep, residuals, return_gate_logits=False):
        assert return_gate_logits is True
        batch_size = pa.shape[0]
        residual_dim = residuals.shape[1]
        total_cost = torch.full((batch_size, 1), 10.0, dtype=torch.float32, device=pa.device)
        aux = torch.zeros(batch_size, residual_dim, dtype=torch.float32, device=pa.device)
        weights = torch.zeros(batch_size, residual_dim, dtype=torch.float32, device=pa.device)
        gate_logits = torch.tensor([[8.0], [-8.0]], dtype=torch.float32, device=pa.device)
        return total_cost, aux, weights, gate_logits


def test_evaluate_anchor_metrics_uses_gate_logits_for_gate_accuracy():
    pa = torch.zeros(2, 3, 8, dtype=torch.float32)
    pb = torch.zeros(2, 3, 8, dtype=torch.float32)
    ma = torch.zeros(2, 3, dtype=torch.bool)
    mb = torch.zeros(2, 3, dtype=torch.bool)
    ep = torch.zeros(2, 4, dtype=torch.float32)
    residuals = torch.zeros(2, 6, dtype=torch.float32)
    gt = torch.tensor([1, 0], dtype=torch.int64)
    bhat = torch.tensor([0.1, 0.2], dtype=torch.float32)
    group_slices = [(0, 2)]
    anchor_ids = ["anchor_0"]

    loader = [(pa, ma, pb, mb, ep, residuals, gt, bhat, group_slices, anchor_ids)]
    metrics = evaluate_anchor_metrics(_FakePINN(), loader, torch.device("cpu"))

    assert metrics["gate_acc_known"] == pytest.approx(1.0)


def test_model_selection_score_composite_changes_with_ranking_metrics():
    base_metrics = {
        "val_anchor_top1_acc": 0.95,
        "val_pairwise_acc_gt_crossclass": 0.80,
        "val_mean_anchor_spearman": 0.20,
        "val_gate_acc_known": 0.60,
    }
    improved_rank_metrics = {
        **base_metrics,
        "val_pairwise_acc_gt_crossclass": 0.90,
        "val_mean_anchor_spearman": 0.40,
    }

    base_score = _compute_model_selection_score(base_metrics, strategy="composite")
    improved_score = _compute_model_selection_score(improved_rank_metrics, strategy="composite")
    top1_score = _compute_model_selection_score(base_metrics, strategy="top1")

    assert improved_score > base_score
    assert top1_score == pytest.approx(base_metrics["val_anchor_top1_acc"])
