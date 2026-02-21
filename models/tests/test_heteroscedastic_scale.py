import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / "utils" / "stitch_cost_interface.py"
MODULE_SPEC = importlib.util.spec_from_file_location("stitch_cost_interface_for_hetero_tests", MODULE_PATH)
STITCH_COST_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(STITCH_COST_MODULE)

HeteroscedasticCostFunction = STITCH_COST_MODULE.HeteroscedasticCostFunction


class _DummyHeteroModel:
    def __init__(self, score: float):
        self._score = float(score)

    def predict_from_embedding(self, emb, query_dt):
        return None, None

    def compute_mahalanobis(self, mu, chol_params, targets, query_mask):
        return torch.tensor([self._score], dtype=torch.float32)


def _build_stubbed_cost_fn(score: float):
    import numpy as np

    cost_fn = object.__new__(HeteroscedasticCostFunction)
    tracked = []

    cost_fn.np = np
    cost_fn.torch = torch
    cost_fn.model = _DummyHeteroModel(score=score)
    cost_fn.cost_reduction = "mean"
    cost_fn.n_query_points = 1
    cost_fn.eval_count = 0
    cost_fn.invalid_cost_count = 0

    cost_fn._gap = lambda track1, track2: 1.0
    cost_fn._build_query_tensors = lambda track1, track2, TIME_WIN: (
        torch.zeros((1, 1), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.bool),
        torch.zeros((1, 1, 2), dtype=torch.float32),
    )
    cost_fn._cache_get_or_encode_embedding = lambda track: torch.zeros((1, 8), dtype=torch.float32)
    cost_fn._track_cost = lambda cost: tracked.append(cost)

    return cost_fn, tracked


def test_heteroscedastic_cost_returns_sqrt_scaled_mahalanobis():
    cost_fn, tracked = _build_stubbed_cost_fn(score=9.0)

    track = {"timestamp": [0.0], "x_position": [0.0], "y_position": [0.0]}
    cost = cost_fn.compute_cost(track, track, TIME_WIN=15.0, param={})

    assert cost == pytest.approx(3.0)
    assert tracked == [pytest.approx(3.0)]
    assert cost_fn.eval_count == 1
    assert cost_fn.invalid_cost_count == 0
