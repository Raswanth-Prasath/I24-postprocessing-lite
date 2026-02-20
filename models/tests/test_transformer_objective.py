import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.evaluate_transformer import (  # noqa: E402
    _resolve_score_mapping,
    _should_add_explicit_time_penalty,
    model_output_to_raw_base_cost,
)
from models.transformer_model import SiameseTransformerNetwork  # noqa: E402


def test_training_objective_controls_output_head():
    cls_model = SiameseTransformerNetwork(training_objective="classification")
    rank_model = SiameseTransformerNetwork(training_objective="ranking")

    assert cls_model.training_objective == "classification"
    assert rank_model.training_objective == "ranking"
    assert isinstance(cls_model.similarity_head[-1], torch.nn.Sigmoid)
    assert not isinstance(rank_model.similarity_head[-1], torch.nn.Sigmoid)


def test_invalid_training_objective_raises():
    with pytest.raises(ValueError):
        SiameseTransformerNetwork(training_objective="unknown")


def test_direct_cost_mapping_is_softplus_non_negative_and_monotonic():
    raw_vals = [-2.0, 0.0, 2.0]
    mapped = [
        model_output_to_raw_base_cost(
            x,
            training_objective="ranking",
            score_mapping="direct_cost",
            scale_factor=5.0,
            similarity_mapping="linear",
            similarity_power=1.0,
            similarity_clip_eps=1e-2,
        )
        for x in raw_vals
    ]
    assert mapped[0] >= 0.0
    assert mapped[0] < mapped[1] < mapped[2]


def test_legacy_similarity_mapping_uses_scale_factor():
    val = model_output_to_raw_base_cost(
        0.8,
        training_objective="classification",
        score_mapping="legacy_similarity",
        scale_factor=5.0,
        similarity_mapping="linear",
        similarity_power=1.0,
        similarity_clip_eps=1e-2,
    )
    assert val == pytest.approx(1.0)


def test_auto_score_mapping_by_objective():
    assert _resolve_score_mapping("classification", "auto") == "legacy_similarity"
    assert _resolve_score_mapping("ranking", "auto") == "direct_cost"


def test_time_penalty_policy_avoids_double_count_for_ranking_direct_cost():
    assert _should_add_explicit_time_penalty("ranking", "direct_cost") is False
    assert _should_add_explicit_time_penalty("ranking", "auto") is False
    assert _should_add_explicit_time_penalty("ranking", "legacy_similarity") is True
    assert _should_add_explicit_time_penalty("classification", "auto") is True
    assert _should_add_explicit_time_penalty("classification", "legacy_similarity") is True
