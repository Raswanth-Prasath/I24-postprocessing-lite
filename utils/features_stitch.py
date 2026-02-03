"""
Centralized Fragment Pair Feature Extraction for Stitching

This module provides deterministic feature extraction that exactly matches
the training feature extraction from enhanced_dataset_creation.py.

CRITICAL: Features extracted during inference MUST match training extraction.
This module replicates the logic from enhanced_dataset_creation.py:90-159.

Supported Modes:
- 'basic': 28 core features (temporal, spatial, kinematic, vehicle)
- 'advanced': 47 features (basic + Bhattacharyya + projection + curvature)
- 'selected': Specific subset of features for inference

Key Design Principles:
1. Deterministic: Same input → same output every time
2. Consistent Ordering: Features always in same order (not dict iteration)
3. Robust: Handle missing fields gracefully (height, confidence)
4. Validated: Unit tests compare training vs inference extraction
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import statsmodels.api as sm
except Exception:
    sm = None


class StitchFeatureExtractor:
    """
    Centralized feature extraction for fragment pair classification.

    Ensures training features match inference features exactly.
    """

    def __init__(
        self,
        mode: str = "basic",
        selected_features: Optional[List[str]] = None,
    ):
        """
        Initialize feature extractor.

        Args:
            mode: Feature extraction mode ('basic', 'advanced', 'selected')
            selected_features: List of feature names for 'selected' mode
        """
        self.mode = mode
        self.selected_features = selected_features

        # Define canonical feature ordering for each mode
        self.basic_features = [
            "time_gap",
            "duration_a",
            "duration_b",
            "duration_ratio",
            "spatial_gap",
            "y_mean_a",
            "y_mean_b",
            "y_diff",
            "y_std_a",
            "y_std_b",
            "vel_a_mean",
            "vel_b_mean",
            "vel_diff",
            "vel_ratio",
            "length_a",
            "length_b",
            "length_diff",
            "width_a",
            "width_b",
            "width_diff",
            "height_a",
            "height_b",
            "height_diff",
            "conf_a_mean",
            "conf_b_mean",
            "conf_a_min",
            "conf_b_min",
            "direction_match",
        ]

        # Advanced mode = basic + Bhattacharyya + projection + velocity + curvature features
        # (47 total features expected from enhanced_dataset_creation.py)
        self.advanced_features = self.basic_features + [
            "projection_error_x_mean",
            "projection_error_x_std",
            "projection_error_x_max",
            "projection_error_y_mean",
            "projection_error_y_std",
            "projection_error_y_max",
            "bhattacharyya_distance",
            "bhattacharyya_coeff",
            "velocity_fit_x",
            "velocity_fit_y",
            "velocity_actual_x_mean",
            "velocity_actual_y_mean",
            "velocity_mismatch_x",
            "velocity_mismatch_y",
            "curvature_a_mean",
            "curvature_a_std",
            "curvature_b_mean",
            "curvature_b_std",
            "curvature_diff",
        ]

        # Validate mode
        if mode not in ["basic", "advanced", "selected"]:
            raise ValueError(f"Unknown mode: {mode}. Use 'basic', 'advanced', or 'selected'")

        if mode == "selected" and selected_features is None:
            raise ValueError("selected_features required for 'selected' mode")

    def extract_feature_vector(
        self, frag_a: Dict, frag_b: Dict
    ) -> np.ndarray:
        """
        Extract features as numpy array for model inference.

        CRITICAL: Must match training feature extraction exactly.

        Args:
            frag_a: First fragment dictionary
            frag_b: Second fragment dictionary

        Returns:
            1D numpy array with features in consistent order
        """
        # Extract basic features first
        features_dict = self._extract_basic_features(frag_a, frag_b)

        # Add advanced features if needed
        if self.mode == "advanced":
            advanced_dict = self._extract_advanced_features(frag_a, frag_b)
            features_dict.update(advanced_dict)

        # Build feature vector in canonical order
        if self.mode == "selected":
            feature_order = self.selected_features
        elif self.mode == "advanced":
            feature_order = self.advanced_features
        else:  # basic
            feature_order = self.basic_features

        # Extract features in order
        feature_vector = []
        for feature_name in feature_order:
            value = features_dict.get(feature_name, 0.0)
            feature_vector.append(float(value))

        return np.array(feature_vector, dtype=np.float32)

    def _extract_basic_features(
        self, frag_a: Dict, frag_b: Dict
    ) -> Dict[str, float]:
        """
        Extract 28 basic geometric and kinematic features.

        This MUST match enhanced_dataset_creation.py:90-159 exactly.

        Args:
            frag_a: First fragment
            frag_b: Second fragment

        Returns:
            Dictionary with feature values
        """
        features = {}

        # Temporal features
        features["time_gap"] = float(
            frag_b["first_timestamp"] - frag_a["last_timestamp"]
        )
        features["duration_a"] = float(
            frag_a["last_timestamp"] - frag_a["first_timestamp"]
        )
        features["duration_b"] = float(
            frag_b["last_timestamp"] - frag_b["first_timestamp"]
        )
        features["duration_ratio"] = features["duration_a"] / (
            features["duration_b"] + 1e-6
        )

        # Spatial features (direction-aware)
        if frag_a.get("direction", 1) == 1:  # Eastbound
            features["spatial_gap"] = float(
                frag_b["starting_x"] - frag_a["ending_x"]
            )
        else:  # Westbound
            features["spatial_gap"] = float(
                frag_a["ending_x"] - frag_b["starting_x"]
            )

        # Lateral features
        y_a = np.array(frag_a["y_position"])
        y_b = np.array(frag_b["y_position"])
        features["y_mean_a"] = float(np.mean(y_a))
        features["y_mean_b"] = float(np.mean(y_b))
        features["y_diff"] = float(abs(features["y_mean_b"] - features["y_mean_a"]))
        features["y_std_a"] = float(np.std(y_a))
        features["y_std_b"] = float(np.std(y_b))

        # Kinematic features
        if "velocity" in frag_a and frag_a["velocity"] is not None:
            vel_a = np.array(frag_a["velocity"])
            vel_a_mean = np.mean(vel_a) if len(vel_a) > 0 else 0.0
            vel_a_end = vel_a[-1] if len(vel_a) > 0 else 0.0
        else:
            vel_a_mean = 0.0
            vel_a_end = 0.0

        if "velocity" in frag_b and frag_b["velocity"] is not None:
            vel_b = np.array(frag_b["velocity"])
            vel_b_mean = np.mean(vel_b) if len(vel_b) > 0 else 0.0
            vel_b_start = vel_b[0] if len(vel_b) > 0 else 0.0
        else:
            vel_b_mean = 0.0
            vel_b_start = 0.0

        features["vel_a_mean"] = float(vel_a_mean)
        features["vel_b_mean"] = float(vel_b_mean)
        features["vel_diff"] = float(abs(vel_a_end - vel_b_start))
        features["vel_ratio"] = float(
            vel_a_end / (vel_b_start + 1e-6)
        )

        # Vehicle dimension features
        len_a = np.array(frag_a["length"])
        len_b = np.array(frag_b["length"])
        features["length_a"] = float(np.mean(len_a))
        features["length_b"] = float(np.mean(len_b))
        features["length_diff"] = float(
            abs(features["length_a"] - features["length_b"])
        )

        width_a = np.array(frag_a["width"])
        width_b = np.array(frag_b["width"])
        features["width_a"] = float(np.mean(width_a))
        features["width_b"] = float(np.mean(width_b))
        features["width_diff"] = float(
            abs(features["width_a"] - features["width_b"])
        )

        # Height (may not be present in all datasets)
        if "height" in frag_a and frag_a["height"] is not None:
            height_a = np.array(frag_a["height"])
            features["height_a"] = float(np.mean(height_a))
        else:
            features["height_a"] = 0.0

        if "height" in frag_b and frag_b["height"] is not None:
            height_b = np.array(frag_b["height"])
            features["height_b"] = float(np.mean(height_b))
        else:
            features["height_b"] = 0.0

        features["height_diff"] = float(
            abs(features["height_a"] - features["height_b"])
        )

        # Detection confidence features
        if (
            "detection_confidence" in frag_a
            and frag_a["detection_confidence"] is not None
        ):
            conf_a = np.array(frag_a["detection_confidence"])
            features["conf_a_mean"] = float(np.mean(conf_a))
            features["conf_a_min"] = float(np.min(conf_a))
        else:
            features["conf_a_mean"] = 1.0
            features["conf_a_min"] = 1.0

        if (
            "detection_confidence" in frag_b
            and frag_b["detection_confidence"] is not None
        ):
            conf_b = np.array(frag_b["detection_confidence"])
            features["conf_b_mean"] = float(np.mean(conf_b))
            features["conf_b_min"] = float(np.min(conf_b))
        else:
            features["conf_b_mean"] = 1.0
            features["conf_b_min"] = 1.0

        # Direction matching
        features["direction_match"] = float(
            1 if frag_a.get("direction", 1) == frag_b.get("direction", 1) else 0
        )

        return features

    def _extract_advanced_features(
        self, frag_a: Dict, frag_b: Dict
    ) -> Dict[str, float]:
        """
        Extract advanced features (Bhattacharyya, projection, curvature).

        Args:
            frag_a: First fragment
            frag_b: Second fragment

        Returns:
            Dictionary with advanced feature values
        """
        features = self._compute_bhattacharyya_features(frag_a, frag_b)

        curv_a_mean, curv_a_std = self._compute_trajectory_curvature(frag_a)
        curv_b_mean, curv_b_std = self._compute_trajectory_curvature(frag_b)
        features["curvature_a_mean"] = curv_a_mean
        features["curvature_a_std"] = curv_a_std
        features["curvature_b_mean"] = curv_b_mean
        features["curvature_b_std"] = curv_b_std
        features["curvature_diff"] = abs(curv_a_mean - curv_b_mean)

        return features

    @staticmethod
    def _bhattacharyya_distance(mu1: np.ndarray, mu2: np.ndarray,
                                cov1: np.ndarray, cov2: np.ndarray) -> float:
        """Compute Bhattacharyya distance between two Gaussian distributions."""
        try:
            mu = mu1 - mu2
            cov = (cov1 + cov2) / 2

            det = np.linalg.det(cov)
            det1 = np.linalg.det(cov1)
            det2 = np.linalg.det(cov2)

            if det <= 0 or det1 <= 0 or det2 <= 0:
                return 999.0

            term1 = 0.125 * np.dot(np.dot(mu.T, np.linalg.inv(cov)), mu)
            term2 = 0.5 * np.log(det / (np.sqrt(det1) * np.sqrt(det2)))
            dist = term1 + term2

            if np.isnan(dist) or np.isinf(dist) or dist < -999:
                return 999.0

            return float(dist)
        except Exception:
            return 999.0

    def _weighted_least_squares(
        self, t: np.ndarray, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> Tuple[List[float], List[float]]:
        """Fit linear model using weighted least squares."""
        try:
            if sm is not None:
                t_const = sm.add_constant(t)
                modelx = sm.WLS(x, t_const, weights=weights)
                resx = modelx.fit()
                fitx = [resx.params[1], resx.params[0]]

                modely = sm.WLS(y, t_const, weights=weights)
                resy = modely.fit()
                fity = [resy.params[1], resy.params[0]]
                return fitx, fity

            # Fallback: weighted polyfit if statsmodels unavailable
            if weights is not None:
                fitx = np.polyfit(t, x, 1, w=weights)
                fity = np.polyfit(t, y, 1, w=weights)
            else:
                fitx = np.polyfit(t, x, 1)
                fity = np.polyfit(t, y, 1)
            return [float(fitx[0]), float(fitx[1])], [float(fity[0]), float(fity[1])]
        except Exception:
            return [0.0, float(np.mean(x))], [0.0, float(np.mean(y))]

    def _compute_bhattacharyya_features(self, frag_a: Dict, frag_b: Dict) -> Dict[str, float]:
        """Compute Bhattacharyya distance-based features between two fragments."""
        features: Dict[str, float] = {}

        try:
            t1 = np.array(frag_a["timestamp"])
            t2 = np.array(frag_b["timestamp"])
            x1 = np.array(frag_a["x_position"])
            x2 = np.array(frag_b["x_position"])
            y1 = np.array(frag_a["y_position"])
            y2 = np.array(frag_b["y_position"])

            toffset = min(t1[0], t2[0])
            t1 = t1 - toffset
            t2 = t2 - toffset

            n1 = min(len(t1), 25)
            t1_fit = t1[-n1:]
            x1_fit = x1[-n1:]
            y1_fit = y1[-n1:]

            n2 = min(len(t2), 25)
            t2_meas = t2[:n2]
            x2_meas = x2[:n2]
            y2_meas = y2[:n2]

            weights1 = np.linspace(1e-6, 1, len(t1_fit))
            fitx, fity = self._weighted_least_squares(t1_fit, x1_fit, y1_fit, weights1)

            x_projected = fitx[0] * t2_meas + fitx[1]
            y_projected = fity[0] * t2_meas + fity[1]

            x_errors = x2_meas - x_projected
            y_errors = y2_meas - y_projected

            features["projection_error_x_mean"] = float(np.mean(x_errors))
            features["projection_error_x_std"] = float(np.std(x_errors))
            features["projection_error_x_max"] = float(np.max(np.abs(x_errors)))

            features["projection_error_y_mean"] = float(np.mean(y_errors))
            features["projection_error_y_std"] = float(np.std(y_errors))
            features["projection_error_y_max"] = float(np.max(np.abs(y_errors)))

            mu1 = np.array([np.mean(x_projected), np.mean(y_projected)])
            mu2 = np.array([np.mean(x2_meas), np.mean(y2_meas)])

            var_x1 = np.var(x_projected) + 1e-6
            var_y1 = np.var(y_projected) + 1e-6
            var_x2 = np.var(x2_meas) + 1e-6
            var_y2 = np.var(y2_meas) + 1e-6

            cov1 = np.diag([var_x1, var_y1])
            cov2 = np.diag([var_x2, var_y2])

            bd = self._bhattacharyya_distance(mu1, mu2, cov1, cov2)
            features["bhattacharyya_distance"] = float(bd)
            features["bhattacharyya_coeff"] = float(np.exp(-bd) if bd < 100 else 0.0)

            features["velocity_fit_x"] = float(fitx[0])
            features["velocity_fit_y"] = float(fity[0])

            if len(t2) > 1:
                dt_b = np.diff(t2)
                dx_b = np.diff(x2)
                dy_b = np.diff(y2)
                vx_b = dx_b / (dt_b + 1e-6)
                vy_b = dy_b / (dt_b + 1e-6)

                features["velocity_actual_x_mean"] = float(np.mean(vx_b[:n2]))
                features["velocity_actual_y_mean"] = float(np.mean(vy_b[:n2]))
                features["velocity_mismatch_x"] = float(abs(fitx[0] - np.mean(vx_b[:n2])))
                features["velocity_mismatch_y"] = float(abs(fity[0] - np.mean(vy_b[:n2])))
            else:
                features["velocity_actual_x_mean"] = 0.0
                features["velocity_actual_y_mean"] = 0.0
                features["velocity_mismatch_x"] = 0.0
                features["velocity_mismatch_y"] = 0.0

        except Exception:
            features["projection_error_x_mean"] = 999.0
            features["projection_error_x_std"] = 999.0
            features["projection_error_x_max"] = 999.0
            features["projection_error_y_mean"] = 999.0
            features["projection_error_y_std"] = 999.0
            features["projection_error_y_max"] = 999.0
            features["bhattacharyya_distance"] = 999.0
            features["bhattacharyya_coeff"] = 0.0
            features["velocity_fit_x"] = 0.0
            features["velocity_fit_y"] = 0.0
            features["velocity_actual_x_mean"] = 0.0
            features["velocity_actual_y_mean"] = 0.0
            features["velocity_mismatch_x"] = 999.0
            features["velocity_mismatch_y"] = 999.0

        return features

    @staticmethod
    def _compute_trajectory_curvature(frag: Dict) -> Tuple[float, float]:
        """Compute trajectory curvature features (mean, std)."""
        try:
            x = np.array(frag["x_position"])
            y = np.array(frag["y_position"])

            if len(x) < 3:
                return 0.0, 0.0

            dx = np.gradient(x)
            dy = np.gradient(y)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

            numerator = np.abs(dx * ddy - dy * ddx)
            denominator = np.power(dx**2 + dy**2, 1.5) + 1e-6
            curvature = numerator / denominator

            return float(np.mean(curvature)), float(np.std(curvature))
        except Exception:
            return 0.0, 0.0

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names in extraction order.

        Returns:
            List of feature names
        """
        if self.mode == "selected":
            return self.selected_features
        elif self.mode == "advanced":
            return self.advanced_features
        else:
            return self.basic_features


# Convenience function
def extract_features(
    frag_a: Dict,
    frag_b: Dict,
    mode: str = "basic",
    selected_features: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract features from fragment pair (convenience function).

    Args:
        frag_a: First fragment
        frag_b: Second fragment
        mode: Feature extraction mode
        selected_features: List of selected feature names (for 'selected' mode)

    Returns:
        1D numpy array of features
    """
    extractor = StitchFeatureExtractor(mode=mode, selected_features=selected_features)
    return extractor.extract_feature_vector(frag_a, frag_b)
