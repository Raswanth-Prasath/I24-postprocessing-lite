"""
Deterministic physics residuals for trajectory stitching.

The residuals are computed from raw fragment data and transformed into a
bounded, threshold-friendly representation used by physics-only and PINN costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


RESIDUAL_DIM = 6
HIGH_RESIDUAL_VALUE = 1_000.0


@dataclass(frozen=True)
class FragmentStats:
    fragment_id: str
    t_ref: float
    first_t: float
    last_t: float
    first_x: float
    last_x: float
    first_y: float
    last_y: float
    v_start: float
    v_end: float
    mean_length: float
    mean_width: float
    direction: float
    n_points: int
    fit_end_x: Tuple[float, float]   # slope, intercept in local (t - t_ref) coordinates
    fit_end_y: Tuple[float, float]   # slope, intercept in local (t - t_ref) coordinates
    fit_start_x: Tuple[float, float] # slope, intercept in local (t - t_ref) coordinates
    fit_start_y: Tuple[float, float] # slope, intercept in local (t - t_ref) coordinates


def _safe_float_array(values: Any, default: float, *, length: Optional[int] = None) -> np.ndarray:
    if values is None:
        if length is None:
            return np.array([default], dtype=np.float64)
        return np.full(int(length), float(default), dtype=np.float64)

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        if length is None:
            arr = np.array([float(arr)], dtype=np.float64)
        else:
            arr = np.full(int(length), float(arr), dtype=np.float64)
    if length is not None and arr.size != int(length):
        if arr.size == 1:
            arr = np.full(int(length), float(arr.item()), dtype=np.float64)
        else:
            arr = arr[: int(length)]
            if arr.size < int(length):
                pad = np.full(int(length) - arr.size, float(arr[-1]), dtype=np.float64)
                arr = np.concatenate([arr, pad], axis=0)
    return np.nan_to_num(arr, nan=float(default), posinf=float(default), neginf=float(default))


def _mean_feature(values: Any, default: float) -> float:
    arr = _safe_float_array(values, default)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def _direction_sign(direction: Any) -> float:
    try:
        d = float(direction)
    except Exception:
        d = 1.0
    if d == 0.0 or not np.isfinite(d):
        return 1.0
    return 1.0 if d > 0 else -1.0


def _compute_velocity_from_positions(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = int(min(len(t), len(x)))
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    dt = np.diff(t[:n])
    dx = np.diff(x[:n])
    v = np.zeros(n, dtype=np.float64)
    v[1:] = dx / np.maximum(dt, 1e-6)
    v[0] = v[1]
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def _endpoint_velocity(track: Dict[str, Any], t: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    n = int(min(len(t), len(x)))
    if n <= 0:
        return 0.0, 0.0

    vel = track.get("velocity")
    if vel is not None:
        vel_arr = _safe_float_array(vel, 0.0, length=n)
    else:
        vel_arr = _compute_velocity_from_positions(t[:n], x[:n])

    if vel_arr.size < n:
        vel_arr = _compute_velocity_from_positions(t[:n], x[:n])

    return float(vel_arr[0]), float(vel_arr[n - 1])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    wsum = float(np.sum(weights))
    if wsum <= 1e-12:
        return float(np.mean(values))
    return float(np.sum(values * weights) / wsum)


def compute_wls_fit(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    min_points_for_fit: int = 3,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Fit x(t) and y(t) as weighted linear models using pure NumPy.
    """
    n = int(min(len(t), len(x), len(y)))
    if n <= 0:
        return (0.0, 0.0), (0.0, 0.0)

    t = np.asarray(t[:n], dtype=np.float64)
    x = np.asarray(x[:n], dtype=np.float64)
    y = np.asarray(y[:n], dtype=np.float64)
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights[:n], dtype=np.float64)
    w = np.clip(np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0), 1e-8, None)

    # Fallback for very short fragments.
    if n < int(min_points_for_fit):
        bx = _weighted_mean(x, w)
        by = _weighted_mean(y, w)
        return (0.0, float(bx)), (0.0, float(by))

    # Weighted least squares via sqrt(w) pre-multiplication.
    sw = np.sqrt(w)
    a = np.column_stack([t, np.ones(n, dtype=np.float64)])
    aw = a * sw[:, None]

    try:
        coef_x, *_ = np.linalg.lstsq(aw, x * sw, rcond=None)
        coef_y, *_ = np.linalg.lstsq(aw, y * sw, rcond=None)
        mx, bx = float(coef_x[0]), float(coef_x[1])
        my, by = float(coef_y[0]), float(coef_y[1])
    except Exception:
        mx, bx = 0.0, _weighted_mean(x, w)
        my, by = 0.0, _weighted_mean(y, w)

    if not np.isfinite(mx):
        mx = 0.0
    if not np.isfinite(my):
        my = 0.0
    if not np.isfinite(bx):
        bx = _weighted_mean(x, w)
    if not np.isfinite(by):
        by = _weighted_mean(y, w)

    return (mx, bx), (my, by)


def _window_size_for_fit(t: np.ndarray, dt_floor: float) -> int:
    if len(t) < 2:
        return int(max(1, len(t)))
    dt = np.diff(t)
    valid = dt[np.isfinite(dt) & (dt > 1e-6)]
    step = float(np.median(valid)) if valid.size > 0 else float(dt_floor)
    step = max(step, float(dt_floor))
    n = int(round(1.0 / step))
    return max(1, n)


def get_fragment_cache_key(track: Dict[str, Any]) -> str:
    for key in ("_id", "local_fragment_id", "fragment_id", "calib_id"):
        if key in track and track[key] is not None:
            return str(track[key])
    return str(id(track))


def compute_fragment_stats(
    track: Dict[str, Any],
    *,
    dt_floor: float = 0.04,
    min_points_for_fit: int = 3,
) -> FragmentStats:
    """
    Precompute per-fragment fit/endpoint stats for O(1) pair residual assembly.
    """
    t = _safe_float_array(track.get("timestamp"), 0.0)
    x = _safe_float_array(track.get("x_position"), 0.0, length=len(t))
    y = _safe_float_array(track.get("y_position"), 0.0, length=len(t))

    n = int(min(len(t), len(x), len(y)))
    if n <= 0:
        t = np.array([0.0], dtype=np.float64)
        x = np.array([0.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)
        n = 1
    t = t[:n]
    x = x[:n]
    y = y[:n]

    t_ref = float(t[0])
    t_local = t - t_ref
    n_fit = min(n, _window_size_for_fit(t, dt_floor))

    # End fit: heavier weight near fragment end.
    t_end = t_local[-n_fit:]
    x_end = x[-n_fit:]
    y_end = y[-n_fit:]
    w_end = np.linspace(1e-6, 1.0, num=n_fit, dtype=np.float64)
    fit_end_x, fit_end_y = compute_wls_fit(
        t_end,
        x_end,
        y_end,
        w_end,
        min_points_for_fit=min_points_for_fit,
    )

    # Start fit: heavier weight near fragment start.
    t_start = t_local[:n_fit]
    x_start = x[:n_fit]
    y_start = y[:n_fit]
    w_start = np.linspace(1.0, 1e-6, num=n_fit, dtype=np.float64)
    fit_start_x, fit_start_y = compute_wls_fit(
        t_start,
        x_start,
        y_start,
        w_start,
        min_points_for_fit=min_points_for_fit,
    )

    direction = _direction_sign(track.get("direction", 1.0))
    m_end_x, b_end_x = fit_end_x
    if m_end_x * direction < 0:
        m_end_x = 0.0
        b_end_x = _weighted_mean(x_end, w_end)

    m_start_x, b_start_x = fit_start_x
    if m_start_x * direction < 0:
        m_start_x = 0.0
        b_start_x = _weighted_mean(x_start, w_start)

    v_start, v_end = _endpoint_velocity(track, t, x)

    return FragmentStats(
        fragment_id=get_fragment_cache_key(track),
        t_ref=float(t_ref),
        first_t=float(t[0]),
        last_t=float(t[-1]),
        first_x=float(x[0]),
        last_x=float(x[-1]),
        first_y=float(y[0]),
        last_y=float(y[-1]),
        v_start=float(v_start),
        v_end=float(v_end),
        mean_length=_mean_feature(track.get("length"), 15.0),
        mean_width=_mean_feature(track.get("width"), 6.0),
        direction=direction,
        n_points=n,
        fit_end_x=(float(m_end_x), float(b_end_x)),
        fit_end_y=(float(fit_end_y[0]), float(fit_end_y[1])),
        fit_start_x=(float(m_start_x), float(b_start_x)),
        fit_start_y=(float(fit_start_y[0]), float(fit_start_y[1])),
    )


def invalid_residuals() -> np.ndarray:
    return np.full(RESIDUAL_DIM, HIGH_RESIDUAL_VALUE, dtype=np.float64)


def compute_physics_residuals(
    stats_a: FragmentStats,
    stats_b: FragmentStats,
    *,
    time_win: float,
    dt_floor: float = 0.04,
    accel_limit: float = 15.0,
    lane_tolerance: float = 6.0,
) -> np.ndarray:
    """
    Deterministic residual vector between two fragments [a -> b].

    Residual order:
      [proj_err_x, proj_err_y, vel_continuity, accel_bound, lane_consistency, dim_match]
    """
    if stats_a.direction != stats_b.direction:
        return invalid_residuals()

    gap = float(stats_b.first_t - stats_a.last_t)
    if (not np.isfinite(gap)) or gap < 0.0 or gap > float(time_win):
        return invalid_residuals()

    dt = max(float(gap), float(dt_floor))
    query_t = float(stats_b.first_t - stats_a.t_ref)

    mx, bx = stats_a.fit_end_x
    my, by = stats_a.fit_end_y
    pred_x = mx * query_t + bx
    pred_y = my * query_t + by

    proj_err_x = abs(float(pred_x - stats_b.first_x))
    proj_err_y = abs(float(pred_y - stats_b.first_y))
    vel_continuity = abs(float(stats_a.v_end - stats_b.v_start))
    implied_acc = abs(float((stats_b.v_start - stats_a.v_end) / dt))
    accel_bound = max(0.0, implied_acc - float(accel_limit))
    lane_consistency = max(0.0, abs(float(stats_b.first_y - stats_a.last_y)) - float(lane_tolerance))
    dim_match = abs(float(stats_a.mean_length - stats_b.mean_length)) + abs(
        float(stats_a.mean_width - stats_b.mean_width)
    )

    raw = np.array(
        [proj_err_x, proj_err_y, vel_continuity, accel_bound, lane_consistency, dim_match],
        dtype=np.float64,
    )
    return np.nan_to_num(raw, nan=HIGH_RESIDUAL_VALUE, posinf=HIGH_RESIDUAL_VALUE, neginf=HIGH_RESIDUAL_VALUE)


def apply_robust_transforms(residuals: Sequence[float]) -> np.ndarray:
    """
    Apply per-residual robust transforms (rho_i).
    """
    r = np.asarray(residuals, dtype=np.float64)
    if r.shape[-1] != RESIDUAL_DIM:
        raise ValueError(f"Expected residual vector with last dim {RESIDUAL_DIM}, got shape {r.shape}.")

    out = np.zeros_like(r, dtype=np.float64)
    out[..., 0] = np.log1p(np.abs(r[..., 0]))  # proj_err_x
    out[..., 1] = np.log1p(np.abs(r[..., 1]))  # proj_err_y
    out[..., 2] = np.log1p(np.abs(r[..., 2]))  # vel continuity
    out[..., 3] = np.sqrt(np.maximum(r[..., 3], 0.0))  # accel bound
    out[..., 4] = np.maximum(r[..., 4], 0.0)  # lane consistency
    out[..., 5] = np.log1p(np.maximum(r[..., 5], 0.0))  # dim mismatch

    return np.nan_to_num(out, nan=HIGH_RESIDUAL_VALUE, posinf=HIGH_RESIDUAL_VALUE, neginf=HIGH_RESIDUAL_VALUE)

