from pathlib import Path
import importlib.util
import json
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pinn_model import PhysicsInformedCostNetwork

MODULE_PATH = PROJECT_ROOT / "utils" / "stitch_cost_interface.py"
MODULE_SPEC = importlib.util.spec_from_file_location("stitch_cost_interface_for_pinn_tests", MODULE_PATH)
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


def test_factory_creates_physics_only_and_returns_finite_cost():
    cfg = {
        "type": "physics_only",
        "physics_weights": [1.0, 1.0, 0.75, 1.25, 0.5, 0.4],
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


def test_factory_creates_pinn_and_returns_finite_cost(tmp_path: Path):
    model = PhysicsInformedCostNetwork(
        input_size=8,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        dropout=0.1,
        endpoint_dim=4,
        use_correction=True,
    )
    ckpt = {
        "model_type": "pinn",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": 8,
            "d_model": 32,
            "nhead": 4,
            "dim_feedforward": 64,
            "num_layers": 1,
            "dropout": 0.1,
            "endpoint_dim": 4,
            "use_correction": True,
        },
        "seq_mean": [0.0] * 8,
        "seq_std": [1.0] * 8,
        "ep_mean": [0.0] * 4,
        "ep_std": [1.0] * 4,
        "train_config": {
            "dt_floor": 0.04,
            "min_points_for_fit": 3,
            "accel_limit": 15.0,
            "lane_tolerance": 6.0,
            "time_win": 15.0,
        },
    }
    ckpt_path = tmp_path / "pinn_test_ckpt.pth"
    torch.save(ckpt, ckpt_path)

    cfg = {
        "type": "pinn",
        "checkpoint_path": str(ckpt_path),
        "device": "cpu",
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


def test_invalid_gap_returns_high_cost(tmp_path: Path):
    cfg = {"type": "physics_only", "stats_log_on_exit": False}
    fn = CostFunctionFactory.create(cfg)
    a = _track([1.0, 1.04], [1.0, 2.0], [0.0, 0.0], frag_id="a")
    b = _track([0.5, 0.54], [2.0, 3.0], [0.0, 0.0], frag_id="b")
    assert fn.compute_cost(a, b, TIME_WIN=15.0, param={}) == 1e6


def test_pinn_logit_score_mappings_and_calibration(tmp_path: Path):
    model = PhysicsInformedCostNetwork(
        input_size=8,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        dropout=0.1,
        endpoint_dim=4,
        use_correction=True,
    )
    with torch.no_grad():
        for p in model.gate_head.parameters():
            p.zero_()
        model.gate_head[5].bias.fill_(2.0)

    ckpt = {
        "model_type": "pinn",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": 8,
            "d_model": 32,
            "nhead": 4,
            "dim_feedforward": 64,
            "num_layers": 1,
            "dropout": 0.1,
            "endpoint_dim": 4,
            "use_correction": True,
        },
        "seq_mean": [0.0] * 8,
        "seq_std": [1.0] * 8,
        "ep_mean": [0.0] * 4,
        "ep_std": [1.0] * 4,
        "train_config": {
            "dt_floor": 0.04,
            "min_points_for_fit": 3,
            "accel_limit": 15.0,
            "lane_tolerance": 6.0,
            "time_win": 15.0,
        },
    }
    ckpt_path = tmp_path / "pinn_logit_mapping_ckpt.pth"
    torch.save(ckpt, ckpt_path)

    calib_path = tmp_path / "pinn_calibration.json"
    calib_path.write_text(
        json.dumps(
            {
                "x_knots": [0.0, 1.0],
                "y_knots": [10.0, 20.0],
                "domain": [0.0, 1.0],
            }
        )
    )

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

    neg_fn = CostFunctionFactory.create(
        {
            "type": "pinn",
            "checkpoint_path": str(ckpt_path),
            "device": "cpu",
            "score_mapping": "neg_logit",
            "stats_log_on_exit": False,
        }
    )
    sp_fn = CostFunctionFactory.create(
        {
            "type": "pinn",
            "checkpoint_path": str(ckpt_path),
            "device": "cpu",
            "score_mapping": "softplus_neg_logit",
            "stats_log_on_exit": False,
        }
    )
    sp_cal_fn = CostFunctionFactory.create(
        {
            "type": "pinn",
            "checkpoint_path": str(ckpt_path),
            "device": "cpu",
            "score_mapping": "softplus_neg_logit",
            "calibration_mode": "isotonic",
            "calibration_path": str(calib_path),
            "stats_log_on_exit": False,
        }
    )

    neg_cost = neg_fn.compute_cost(a, b, TIME_WIN=15.0, param={})
    sp_cost = sp_fn.compute_cost(a, b, TIME_WIN=15.0, param={})
    sp_cal_cost = sp_cal_fn.compute_cost(a, b, TIME_WIN=15.0, param={})

    assert neg_cost < 0.0
    assert sp_cost > 0.0
    assert sp_cost > neg_cost
    assert sp_cal_cost > sp_cost
