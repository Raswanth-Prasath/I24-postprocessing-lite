"""
Physics-only cost model for trajectory stitching.
"""

from __future__ import annotations

import atexit
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    from models.physics_residuals import (
        RESIDUAL_DIM,
        apply_robust_transforms,
        compute_fragment_stats,
        compute_physics_residuals,
        get_fragment_cache_key,
    )
except ImportError:
    from physics_residuals import (  # type: ignore
        RESIDUAL_DIM,
        apply_robust_transforms,
        compute_fragment_stats,
        compute_physics_residuals,
        get_fragment_cache_key,
    )


DEFAULT_PHYSICS_WEIGHTS = np.array([1.0, 1.0, 0.75, 1.25, 0.5, 0.4], dtype=np.float64)


class PhysicsOnlyCostModel:
    """
    Deterministic physics cost:
        cost = sum_i w_i * rho_i(r_i) + time_penalty * gap
    """

    def __init__(
        self,
        *,
        physics_weights: Optional[Sequence[float]] = None,
        dt_floor: float = 0.04,
        min_points_for_fit: int = 3,
        accel_limit: float = 15.0,
        lane_tolerance: float = 6.0,
        time_penalty: float = 0.0,
        stats_log_on_exit: bool = True,
    ):
        if physics_weights is None:
            weights = DEFAULT_PHYSICS_WEIGHTS.copy()
        else:
            weights = np.asarray(physics_weights, dtype=np.float64)
        if weights.shape != (RESIDUAL_DIM,):
            raise ValueError(f"physics_weights must have shape ({RESIDUAL_DIM},), got {weights.shape}")

        self.physics_weights = np.clip(weights, 0.0, None)
        self.dt_floor = float(dt_floor)
        self.min_points_for_fit = int(min_points_for_fit)
        self.accel_limit = float(accel_limit)
        self.lane_tolerance = float(lane_tolerance)
        self.time_penalty = float(time_penalty)

        self._fragment_cache: Dict[str, Any] = {}
        self.eval_count = 0
        self.invalid_count = 0
        self._stats_log_on_exit = bool(stats_log_on_exit)
        if self._stats_log_on_exit:
            atexit.register(self.log_stats)

    @staticmethod
    def _gap(track1: dict, track2: dict) -> float:
        if "first_timestamp" in track2 and "last_timestamp" in track1:
            return float(track2["first_timestamp"] - track1["last_timestamp"])
        return float(track2["timestamp"][0] - track1["timestamp"][-1])

    def _get_fragment_stats(self, track: dict):
        key = get_fragment_cache_key(track)
        stats = self._fragment_cache.get(key)
        if stats is None:
            stats = compute_fragment_stats(
                track,
                dt_floor=self.dt_floor,
                min_points_for_fit=self.min_points_for_fit,
            )
            self._fragment_cache[key] = stats
        return stats

    def log_stats(self) -> None:
        data = {
            "eval_count": int(self.eval_count),
            "invalid_count": int(self.invalid_count),
            "cache_size": int(len(self._fragment_cache)),
            "physics_weights": [float(x) for x in self.physics_weights.tolist()],
            "dt_floor": float(self.dt_floor),
            "accel_limit": float(self.accel_limit),
            "lane_tolerance": float(self.lane_tolerance),
            "time_penalty": float(self.time_penalty),
        }
        print(f"[PhysicsOnlyCostModel] Stats: {json.dumps(data, sort_keys=True)}")

    def compute_cost(self, track1: dict, track2: dict, time_win: float) -> float:
        self.eval_count += 1
        try:
            gap = self._gap(track1, track2)
            if (not np.isfinite(gap)) or gap < 0.0 or gap > float(time_win):
                self.invalid_count += 1
                return 1e6

            stats_a = self._get_fragment_stats(track1)
            stats_b = self._get_fragment_stats(track2)
            residuals = compute_physics_residuals(
                stats_a,
                stats_b,
                time_win=float(time_win),
                dt_floor=self.dt_floor,
                accel_limit=self.accel_limit,
                lane_tolerance=self.lane_tolerance,
            )
            residuals_t = apply_robust_transforms(residuals)
            cost = float(np.dot(self.physics_weights, residuals_t) + self.time_penalty * gap)
            if not np.isfinite(cost):
                self.invalid_count += 1
                return 1e6
            return cost
        except Exception:
            self.invalid_count += 1
            return 1e6
