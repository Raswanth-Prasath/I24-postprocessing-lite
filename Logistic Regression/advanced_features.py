"""
Advanced Feature Extraction for Fragment Association

Includes sophisticated features based on the baseline cost function:
- Bhattacharyya distance (simplified version)
- Velocity-based trajectory projection
- Statistical distribution features
"""

import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def bhattacharyya_distance(mu1, mu2, cov1, cov2):
    """
    Compute Bhattacharyya distance between two Gaussian distributions

    Args:
        mu1, mu2: Mean vectors
        cov1, cov2: Covariance matrices

    Returns:
        Bhattacharyya distance
    """
    try:
        mu = mu1 - mu2
        cov = (cov1 + cov2) / 2

        det = np.linalg.det(cov)
        det1 = np.linalg.det(cov1)
        det2 = np.linalg.det(cov2)

        if det <= 0 or det1 <= 0 or det2 <= 0:
            return 999.0  # Invalid covariance

        # Bhattacharyya distance formula
        term1 = 0.125 * np.dot(np.dot(mu.T, np.linalg.inv(cov)), mu)
        term2 = 0.5 * np.log(det / (np.sqrt(det1) * np.sqrt(det2)))

        dist = term1 + term2

        if np.isnan(dist) or np.isinf(dist) or dist < -999:
            return 999.0

        return float(dist)

    except Exception:
        return 999.0


def weighted_least_squares(t, x, y, weights=None):
    """
    Fit linear model using weighted least squares

    Args:
        t: Time array
        x: X-position array
        y: Y-position array
        weights: Weights for each point

    Returns:
        (fitx, fity): Tuples of (slope, intercept) for x and y
    """
    try:
        t_const = sm.add_constant(t)

        # Fit x
        modelx = sm.WLS(x, t_const, weights=weights)
        resx = modelx.fit()
        fitx = [resx.params[1], resx.params[0]]  # [slope, intercept]

        # Fit y
        modely = sm.WLS(y, t_const, weights=weights)
        resy = modely.fit()
        fity = [resy.params[1], resy.params[0]]  # [slope, intercept]

        return fitx, fity

    except Exception:
        # Fallback to simple mean if WLS fails
        return [0.0, np.mean(x)], [0.0, np.mean(y)]


def compute_bhattacharyya_features(frag_a, frag_b):
    """
    Compute Bhattacharyya distance-based features between two fragments

    This is a simplified version of the baseline cost function
    """
    features = {}

    try:
        t1 = np.array(frag_a['timestamp'])
        t2 = np.array(frag_b['timestamp'])
        x1 = np.array(frag_a['x_position'])
        x2 = np.array(frag_b['x_position'])
        y1 = np.array(frag_a['y_position'])
        y2 = np.array(frag_b['y_position'])

        # Normalize timestamps
        toffset = min(t1[0], t2[0])
        t1 = t1 - toffset
        t2 = t2 - toffset

        # Use last 25 points of frag_a (or all if fewer)
        n1 = min(len(t1), 25)
        t1_fit = t1[-n1:]
        x1_fit = x1[-n1:]
        y1_fit = y1[-n1:]

        # Use first 25 points of frag_b (or all if fewer)
        n2 = min(len(t2), 25)
        t2_meas = t2[:n2]
        x2_meas = x2[:n2]
        y2_meas = y2[:n2]

        # Fit linear model to frag_a (weighted towards end)
        weights1 = np.linspace(1e-6, 1, len(t1_fit))
        fitx, fity = weighted_least_squares(t1_fit, x1_fit, y1_fit, weights1)

        # Project frag_a forward to frag_b timestamps
        x_projected = fitx[0] * t2_meas + fitx[1]
        y_projected = fity[0] * t2_meas + fity[1]

        # Compute projection errors
        x_errors = x2_meas - x_projected
        y_errors = y2_meas - y_projected

        features['projection_error_x_mean'] = np.mean(x_errors)
        features['projection_error_x_std'] = np.std(x_errors)
        features['projection_error_x_max'] = np.max(np.abs(x_errors))

        features['projection_error_y_mean'] = np.mean(y_errors)
        features['projection_error_y_std'] = np.std(y_errors)
        features['projection_error_y_max'] = np.max(np.abs(y_errors))

        # Compute Bhattacharyya distance (simplified)
        # Use 2D Gaussian approximation
        mu1 = np.array([np.mean(x_projected), np.mean(y_projected)])
        mu2 = np.array([np.mean(x2_meas), np.mean(y2_meas)])

        var_x1 = np.var(x_projected) + 1e-6
        var_y1 = np.var(y_projected) + 1e-6
        var_x2 = np.var(x2_meas) + 1e-6
        var_y2 = np.var(y2_meas) + 1e-6

        cov1 = np.diag([var_x1, var_y1])
        cov2 = np.diag([var_x2, var_y2])

        bd = bhattacharyya_distance(mu1, mu2, cov1, cov2)
        features['bhattacharyya_distance'] = bd
        features['bhattacharyya_coeff'] = np.exp(-bd) if bd < 100 else 0.0

        # Velocity consistency
        features['velocity_fit_x'] = fitx[0]  # Slope in x direction
        features['velocity_fit_y'] = fity[0]  # Slope in y direction

        # Compute actual velocity of frag_b
        if len(t2) > 1:
            dt_b = np.diff(t2)
            dx_b = np.diff(x2)
            dy_b = np.diff(y2)
            vx_b = dx_b / (dt_b + 1e-6)
            vy_b = dy_b / (dt_b + 1e-6)

            features['velocity_actual_x_mean'] = np.mean(vx_b[:n2])
            features['velocity_actual_y_mean'] = np.mean(vy_b[:n2])
            features['velocity_mismatch_x'] = abs(fitx[0] - np.mean(vx_b[:n2]))
            features['velocity_mismatch_y'] = abs(fity[0] - np.mean(vy_b[:n2]))
        else:
            features['velocity_actual_x_mean'] = 0.0
            features['velocity_actual_y_mean'] = 0.0
            features['velocity_mismatch_x'] = 0.0
            features['velocity_mismatch_y'] = 0.0

    except Exception as e:
        # If computation fails, return default values
        features['projection_error_x_mean'] = 999.0
        features['projection_error_x_std'] = 999.0
        features['projection_error_x_max'] = 999.0
        features['projection_error_y_mean'] = 999.0
        features['projection_error_y_std'] = 999.0
        features['projection_error_y_max'] = 999.0
        features['bhattacharyya_distance'] = 999.0
        features['bhattacharyya_coeff'] = 0.0
        features['velocity_fit_x'] = 0.0
        features['velocity_fit_y'] = 0.0
        features['velocity_actual_x_mean'] = 0.0
        features['velocity_actual_y_mean'] = 0.0
        features['velocity_mismatch_x'] = 999.0
        features['velocity_mismatch_y'] = 999.0

    return features


def compute_trajectory_curvature(frag):
    """
    Compute trajectory curvature features
    """
    try:
        x = np.array(frag['x_position'])
        y = np.array(frag['y_position'])

        if len(x) < 3:
            return 0.0, 0.0

        # Compute derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2, 1.5) + 1e-6
        curvature = numerator / denominator

        return np.mean(curvature), np.std(curvature)

    except Exception:
        return 0.0, 0.0


def extract_advanced_features(frag_a, frag_b):
    """
    Extract all advanced features for a fragment pair

    Returns:
        Dictionary of advanced features
    """
    features = {}

    # Bhattacharyya distance and projection features
    bhatt_features = compute_bhattacharyya_features(frag_a, frag_b)
    features.update(bhatt_features)

    # Trajectory curvature
    curv_a_mean, curv_a_std = compute_trajectory_curvature(frag_a)
    curv_b_mean, curv_b_std = compute_trajectory_curvature(frag_b)

    features['curvature_a_mean'] = curv_a_mean
    features['curvature_a_std'] = curv_a_std
    features['curvature_b_mean'] = curv_b_mean
    features['curvature_b_std'] = curv_b_std
    features['curvature_diff'] = abs(curv_a_mean - curv_b_mean)

    return features


if __name__ == "__main__":
    # Test the feature extraction
    print("Advanced feature extraction module loaded successfully")
    print("\nFeatures that will be extracted:")
    print("1. Bhattacharyya distance")
    print("2. Bhattacharyya coefficient")
    print("3. Projection errors (x and y)")
    print("4. Velocity fit and mismatch")
    print("5. Trajectory curvature")
    print("\nTotal: ~20 additional features")
