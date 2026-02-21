from pathlib import Path
import importlib.util
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.heteroscedastic_model import HeteroscedasticTrajectoryModel

MODULE_PATH = PROJECT_ROOT / "utils" / "stitch_cost_interface.py"
MODULE_SPEC = importlib.util.spec_from_file_location("stitch_cost_interface_for_hetero_tests", MODULE_PATH)
STITCH_COST_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(STITCH_COST_MODULE)
CostFunctionFactory = STITCH_COST_MODULE.CostFunctionFactory


def _track(
    t,
    x,
    y,
    *,
    direction=1,
    velocity=None,
    frag_id="f",
):
    return {
        "_id": frag_id,
        "timestamp": list(t),
        "x_position": list(x),
        "y_position": list(y),
        "direction": direction,
        "velocity": velocity,
        "length": 15.0,
        "width": 6.0,
        "height": 5.0,
        "detection_confidence": [0.9 for _ in t],
    }


def _write_ckpt(path: Path) -> None:
    model = HeteroscedasticTrajectoryModel(
        input_size=8,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        dropout=0.1,
        pred_hidden=64,
        cov_eps=1e-3,
    )
    ckpt = {
        "model_type": "heteroscedastic",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": 8,
            "d_model": 32,
            "nhead": 4,
            "dim_feedforward": 64,
            "num_layers": 1,
            "dropout": 0.1,
            "pred_hidden": 64,
            "cov_eps": 1e-3,
        },
        "seq_mean": [0.0] * 8,
        "seq_std": [1.0] * 8,
    }
    torch.save(ckpt, str(path))


def test_factory_creates_heteroscedastic_and_returns_finite_cost(tmp_path: Path):
    ckpt_path = tmp_path / "hetero_test_ckpt.pth"
    _write_ckpt(ckpt_path)

    cfg = {
        "type": "heteroscedastic",
        "checkpoint_path": str(ckpt_path),
        "device": "cpu",
        "n_query_points": 3,
        "cost_reduction": "mean",
        "cache_max_size": 8,
        "stats_log_on_exit": False,
    }
    fn = CostFunctionFactory.create(cfg)

    a = _track(
        t=[0.00, 0.04, 0.08, 0.12],
        x=[0.0, 1.0, 2.0, 3.0],
        y=[4.0, 4.0, 4.1, 4.1],
        velocity=[25.0, 25.0, 25.0, 25.0],
        frag_id="a",
    )
    b = _track(
        t=[0.16, 0.20, 0.24],
        x=[4.0, 5.0, 6.0],
        y=[4.1, 4.1, 4.2],
        velocity=[25.0, 25.0, 25.0],
        frag_id="b",
    )

    cost = fn.compute_cost(a, b, TIME_WIN=15.0, param={})
    assert isinstance(cost, float)
    assert cost < 1e6


def test_cache_hits_increase_on_repeat_calls(tmp_path: Path):
    ckpt_path = tmp_path / "hetero_test_ckpt.pth"
    _write_ckpt(ckpt_path)

    cfg = {
        "type": "heteroscedastic",
        "checkpoint_path": str(ckpt_path),
        "device": "cpu",
        "n_query_points": 3,
        "cache_max_size": 2,
        "stats_log_on_exit": False,
    }
    fn = CostFunctionFactory.create(cfg)

    a = _track([0.00, 0.04, 0.08], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], frag_id="a")
    b = _track([0.12, 0.16, 0.20], [2.8, 3.8, 4.8], [0.0, 0.0, 0.1], frag_id="b")

    _ = fn.compute_cost(a, b, TIME_WIN=15.0, param={})
    _ = fn.compute_cost(a, b, TIME_WIN=15.0, param={})

    stats = fn.get_stats()
    assert int(stats["cache_hits"]) >= 1


def test_invalid_gap_returns_high_cost(tmp_path: Path):
    ckpt_path = tmp_path / "hetero_test_ckpt.pth"
    _write_ckpt(ckpt_path)

    cfg = {
        "type": "heteroscedastic",
        "checkpoint_path": str(ckpt_path),
        "device": "cpu",
        "stats_log_on_exit": False,
    }
    fn = CostFunctionFactory.create(cfg)

    a = _track([1.00, 1.04], [1.0, 2.0], [0.0, 0.0], frag_id="a")
    b = _track([0.50, 0.54], [2.0, 3.0], [0.0, 0.0], frag_id="b")
    assert fn.compute_cost(a, b, TIME_WIN=15.0, param={}) == 1e6
