import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.physics_residuals import (
    HIGH_RESIDUAL_VALUE,
    apply_robust_transforms,
    compute_fragment_stats,
    compute_physics_residuals,
    compute_wls_fit,
    invalid_residuals,
)


def _track(
    t,
    x,
    y,
    *,
    direction=1,
    velocity=None,
    length=15.0,
    width=6.0,
    frag_id="f",
):
    return {
        "_id": frag_id,
        "timestamp": np.asarray(t, dtype=np.float64).tolist(),
        "x_position": np.asarray(x, dtype=np.float64).tolist(),
        "y_position": np.asarray(y, dtype=np.float64).tolist(),
        "direction": direction,
        "velocity": velocity,
        "length": length,
        "width": width,
    }


def test_compute_wls_fit_short_track_fallback():
    t = np.array([0.0, 0.1], dtype=np.float64)
    x = np.array([5.0, 6.0], dtype=np.float64)
    y = np.array([2.0, 2.2], dtype=np.float64)
    fitx, fity = compute_wls_fit(t, x, y, min_points_for_fit=3)
    assert fitx[0] == 0.0
    assert fity[0] == 0.0
    assert np.isfinite(fitx[1])
    assert np.isfinite(fity[1])


def test_directional_velocity_clamp_non_negative_progress():
    # Direction=+1 but x decreases; x-slope should be clamped to 0.
    tr = _track(
        t=[0.0, 0.04, 0.08, 0.12, 0.16],
        x=[10.0, 9.8, 9.6, 9.4, 9.2],
        y=[2.0, 2.0, 2.1, 2.1, 2.2],
        direction=1,
    )
    stats = compute_fragment_stats(tr)
    assert stats.fit_end_x[0] == 0.0
    assert stats.fit_start_x[0] == 0.0


def test_compute_physics_residuals_invalid_gap():
    a = compute_fragment_stats(
        _track([0.0, 0.04, 0.08], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], frag_id="a")
    )
    b = compute_fragment_stats(
        _track([20.0, 20.04, 20.08], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0], frag_id="b")
    )
    r = compute_physics_residuals(a, b, time_win=15.0)
    assert np.allclose(r, invalid_residuals())


def test_compute_physics_residuals_finite_and_low_for_consistent_pair():
    a = compute_fragment_stats(
        _track(
            [0.00, 0.04, 0.08, 0.12],
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 4.0, 4.1, 4.1],
            velocity=[25.0, 25.0, 25.0, 25.0],
            frag_id="a",
        )
    )
    b = compute_fragment_stats(
        _track(
            [0.16, 0.20, 0.24],
            [4.0, 5.0, 6.0],
            [4.1, 4.1, 4.2],
            velocity=[25.0, 25.0, 25.0],
            frag_id="b",
        )
    )
    r = compute_physics_residuals(a, b, time_win=15.0)
    assert np.all(np.isfinite(r))
    assert (r[:3] < 5.0).all()


def test_apply_robust_transforms_shape_and_finiteness():
    r = np.array([0.5, 1.0, 2.0, 0.0, 3.0, 0.4], dtype=np.float64)
    t = apply_robust_transforms(r)
    assert t.shape == (6,)
    assert np.all(np.isfinite(t))
    assert np.all(t >= 0.0)


def test_apply_robust_transforms_handles_large_invalid_values():
    r = np.array([np.nan, np.inf, -np.inf, HIGH_RESIDUAL_VALUE, 1.0, 2.0], dtype=np.float64)
    t = apply_robust_transforms(r)
    assert t.shape == (6,)
    assert np.all(np.isfinite(t))
