import math
import sys
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / "utils" / "stitch_cost_interface.py"
MODULE_SPEC = importlib.util.spec_from_file_location("stitch_cost_interface_for_tests", MODULE_PATH)
STITCH_COST_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(STITCH_COST_MODULE)

CostFunctionFactory = STITCH_COST_MODULE.CostFunctionFactory
GateRankCostFunction = STITCH_COST_MODULE.GateRankCostFunction
StitchCostFunction = STITCH_COST_MODULE.StitchCostFunction


class MockCostFunction(StitchCostFunction):
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = 0

    def compute_cost(self, track1: dict, track2: dict, TIME_WIN: float, param: dict) -> float:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.value


def test_reject_path():
    gate_fn = MockCostFunction(value=4.2)
    rank_fn = MockCostFunction(value=0.9)
    cost_fn = GateRankCostFunction(gate_fn=gate_fn, rank_fn=rank_fn, gate_thresh=3.0, stats_log_on_exit=False)

    cost = cost_fn.compute_cost({}, {}, 15.0, {"stitch_thresh": 3.0})

    assert cost == pytest.approx(4.2)
    assert gate_fn.calls == 1
    assert rank_fn.calls == 0
    assert cost_fn.gate_evaluations == 1
    assert cost_fn.gate_rejections == 1


def test_accept_path():
    gate_fn = MockCostFunction(value=2.1)
    rank_fn = MockCostFunction(value=0.7)
    cost_fn = GateRankCostFunction(gate_fn=gate_fn, rank_fn=rank_fn, gate_thresh=3.0, stats_log_on_exit=False)

    cost = cost_fn.compute_cost({}, {}, 15.0, {"stitch_thresh": 3.0})

    assert cost == pytest.approx(0.7)
    assert gate_fn.calls == 1
    assert rank_fn.calls == 1
    assert cost_fn.gate_evaluations == 1
    assert cost_fn.gate_rejections == 0
    assert cost_fn.rank_evaluations == 1


@pytest.mark.parametrize("bad_rank_cost", [float("nan"), float("inf")])
def test_nan_inf_fallback(bad_rank_cost):
    gate_fn = MockCostFunction(value=2.4)
    rank_fn = MockCostFunction(value=bad_rank_cost)
    cost_fn = GateRankCostFunction(gate_fn=gate_fn, rank_fn=rank_fn, gate_thresh=3.0, stats_log_on_exit=False)

    cost = cost_fn.compute_cost({}, {}, 15.0, {"stitch_thresh": 3.0})

    assert cost == pytest.approx(2.4)
    assert math.isfinite(cost)
    assert cost_fn.rank_fallbacks == 1


def test_factory_aliases(monkeypatch):
    original_create = CostFunctionFactory.create

    def patched_create(config):
        if config.get("type") == "mock":
            return MockCostFunction(value=0.0)
        return original_create(config)

    monkeypatch.setattr(CostFunctionFactory, "create", staticmethod(patched_create))

    gate_rank = CostFunctionFactory.create(
        {
            "type": "gate_rank",
            "gate": {"type": "mock"},
            "rank": {"type": "mock"},
            "stats_log_on_exit": False,
        }
    )
    hybrid = CostFunctionFactory.create(
        {
            "type": "hybrid",
            "gate": {"type": "mock"},
            "rank": {"type": "mock"},
            "stats_log_on_exit": False,
        }
    )

    assert isinstance(gate_rank, GateRankCostFunction)
    assert isinstance(hybrid, GateRankCostFunction)


def test_default_threshold():
    gate_fn = MockCostFunction(value=3.5)
    rank_fn = MockCostFunction(value=0.6)
    cost_fn = GateRankCostFunction(gate_fn=gate_fn, rank_fn=rank_fn, gate_thresh=None, stats_log_on_exit=False)

    accepted = cost_fn.compute_cost({}, {}, 15.0, {"stitch_thresh": 4.0})
    rejected = cost_fn.compute_cost({}, {}, 15.0, {"stitch_thresh": 3.0})

    assert accepted == pytest.approx(0.6)
    assert rejected == pytest.approx(3.5)
