import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.evaluate_transformer import (  # noqa: E402
    _build_anchor_outcomes,
    _compute_anchor_metrics,
    _get_pair_gt_label,
    _make_complementarity_decision,
)


def test_build_anchor_outcomes_filters_by_min_candidates_and_label_diversity():
    rows = [
        # Eligible anchor A: has both classes and >=2 candidates
        {"anchor_id": "A", "candidate_id": "A1", "bhat_cost": 2.0, "tx_cost": 1.0, "gap": 1.0, "gt_label": 1},
        {"anchor_id": "A", "candidate_id": "A2", "bhat_cost": 1.0, "tx_cost": 2.0, "gap": 1.2, "gt_label": 0},
        # Anchor B: only one candidate -> ineligible
        {"anchor_id": "B", "candidate_id": "B1", "bhat_cost": 1.0, "tx_cost": 1.0, "gap": 1.0, "gt_label": 1},
        # Anchor C: all positives -> ineligible
        {"anchor_id": "C", "candidate_id": "C1", "bhat_cost": 1.0, "tx_cost": 1.0, "gap": 1.0, "gt_label": 1},
        {"anchor_id": "C", "candidate_id": "C2", "bhat_cost": 2.0, "tx_cost": 2.0, "gap": 1.0, "gt_label": 1},
        # Anchor D: unknown GT -> removed before grouping
        {"anchor_id": "D", "candidate_id": "D1", "bhat_cost": 1.0, "tx_cost": 1.0, "gap": 1.0, "gt_label": None},
    ]

    outcomes, by_anchor = _build_anchor_outcomes(rows, min_candidates=2)

    assert len(by_anchor) == 3
    assert sorted(by_anchor.keys()) == ["A", "B", "C"]
    assert len(outcomes) == 1
    assert outcomes[0]["anchor_id"] == "A"
    assert outcomes[0]["bhat_correct"] is False
    assert outcomes[0]["tx_correct"] is True


def test_compute_anchor_metrics_reports_error_lift_recovery_and_regression():
    outcomes = [
        {"anchor_id": "A", "bhat_correct": False, "tx_correct": True},
        {"anchor_id": "B", "bhat_correct": False, "tx_correct": True},
        {"anchor_id": "C", "bhat_correct": True, "tx_correct": False},
        {"anchor_id": "D", "bhat_correct": True, "tx_correct": True},
    ]

    metrics = _compute_anchor_metrics(outcomes, include_anchor_ids=True)

    assert metrics["num_anchors"] == 4
    assert metrics["bhat_top1_error_rate"] == pytest.approx(0.5)
    assert metrics["tx_top1_error_rate"] == pytest.approx(0.25)
    assert metrics["edge_error_lift"] == pytest.approx(0.5)
    assert metrics["bhat_failure_recovery_rate"] == pytest.approx(1.0)
    assert metrics["tx_regression_rate_on_bhat_correct"] == pytest.approx(0.5)
    assert sorted(metrics["bhat_wrong_tx_right_anchor_ids"]) == ["A", "B"]
    assert metrics["bhat_right_tx_wrong_anchor_ids"] == ["C"]


def test_make_complementarity_decision_continue_and_pivot_paths():
    overall_continue = {
        "num_anchors": 100,
        "bhat_failure_recovery_rate": 0.20,
        "tx_regression_rate_on_bhat_correct": 0.05,
        "tx_top1_error_rate": 0.20,
        "bhat_top1_error_rate": 0.25,
    }
    decision_continue = _make_complementarity_decision(overall_continue, go_lift_threshold=0.15)
    assert decision_continue["recommendation"] == "continue"
    assert all(check["pass"] for check in decision_continue["checks"])

    overall_pivot = {
        "num_anchors": 100,
        "bhat_failure_recovery_rate": 0.10,  # below threshold
        "tx_regression_rate_on_bhat_correct": 0.06,  # above threshold
        "tx_top1_error_rate": 0.30,  # worse than Bhattacharyya
        "bhat_top1_error_rate": 0.25,
    }
    decision_pivot = _make_complementarity_decision(overall_pivot, go_lift_threshold=0.15)
    assert decision_pivot["recommendation"] == "pivot"
    assert not any(check["pass"] for check in decision_pivot["checks"])


def test_make_complementarity_decision_handles_insufficient_anchor_data():
    overall_empty = {
        "num_anchors": 0,
        "bhat_failure_recovery_rate": 0.0,
        "tx_regression_rate_on_bhat_correct": 0.0,
        "tx_top1_error_rate": 0.0,
        "bhat_top1_error_rate": 0.0,
    }
    decision = _make_complementarity_decision(overall_empty, go_lift_threshold=0.15)
    assert decision["recommendation"] == "pivot"
    assert decision["checks"] == []
    assert "insufficient_data" in decision["reasons"][0]


def test_get_pair_gt_label_handles_missing_and_present_gt():
    track_missing = {"timestamp": [0.0]}
    track_with_gt_a = {"gt_ids": [[{"$oid": "veh_a"}]], "timestamp": [0.0]}
    track_with_gt_b = {"gt_ids": [[{"$oid": "veh_b"}]], "timestamp": [0.0]}

    assert _get_pair_gt_label(track_missing, track_with_gt_a) is None
    assert _get_pair_gt_label(track_with_gt_a, track_with_gt_a) == 1
    assert _get_pair_gt_label(track_with_gt_a, track_with_gt_b) == 0
