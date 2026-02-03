"""
VIF (Variance Inflation Factor) Analysis for Multicollinearity Detection

Quantifies how much the variance of a feature is inflated due to multicollinearity.
VIF = 1 / (1 - R²_i) where R²_i is R² from regressing feature i on all others

Thresholds:
- VIF < 5: Low multicollinearity (keep)
- 5 ≤ VIF < 10: Moderate (monitor)
- VIF ≥ 10: High (remove)

Reference: Standard practice in regression analysis (O'Brien 2007, James et al. 2013)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, List, Tuple
import json


class VIFAnalyzer:
    """
    Compute and manage Variance Inflation Factors for feature multicollinearity.
    """

    def __init__(self, threshold: float = 10.0):
        """
        Args:
            threshold: VIF threshold above which features are considered too multicollinear (default 10)
        """
        self.threshold = threshold
        self.vif_scores = {}
        self.removed_features = []

    def compute_vif(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Compute VIF for each feature.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature name -> VIF score
        """
        if X.shape[0] < X.shape[1] + 1:
            print(f"Warning: {X.shape[0]} samples < {X.shape[1]} features")

        vif_scores = {}

        for i, feature_name in enumerate(feature_names):
            # Regress feature i on all other features
            X_i = np.delete(X, i, axis=1)
            y_i = X[:, i]

            try:
                lr = LinearRegression()
                lr.fit(X_i, y_i)
                r_squared = lr.score(X_i, y_i)

                # VIF = 1 / (1 - R²)
                # Handle edge case where R² ≈ 1 (perfect multicollinearity)
                vif = 1.0 / (1.0 - r_squared + 1e-8)
                vif_scores[feature_name] = vif

            except Exception as e:
                print(f"Error computing VIF for {feature_name}: {e}")
                vif_scores[feature_name] = np.inf

        self.vif_scores = vif_scores
        return vif_scores

    def remove_multicollinear_features(
        self, X: np.ndarray, feature_names: List[str], threshold: float = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Iteratively remove features with VIF > threshold.

        Algorithm:
        1. Compute VIF for all features
        2. Find feature with highest VIF
        3. If VIF > threshold:
           - Remove feature
           - Recompute VIF for remaining features
           - Repeat until all VIF < threshold
        4. Return filtered feature set

        Args:
            X: Feature matrix
            feature_names: List of feature names
            threshold: VIF threshold (uses self.threshold if None)

        Returns:
            (X_filtered, feature_names_filtered)
        """
        if threshold is None:
            threshold = self.threshold

        X_current = X.copy()
        features_current = list(feature_names)
        self.removed_features = []

        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- VIF Iteration {iteration} ---")
            print(f"Features remaining: {len(features_current)}")

            # Compute VIF
            vif_scores = self.compute_vif(X_current, features_current)

            # Find max VIF
            max_vif_feature = max(vif_scores, key=vif_scores.get)
            max_vif_value = vif_scores[max_vif_feature]

            print(f"Max VIF: {max_vif_value:.2f} ({max_vif_feature})")

            # Check stopping condition
            if max_vif_value <= threshold:
                print(f"\nAll features have VIF ≤ {threshold}. Stopping.")
                break

            # Remove feature with highest VIF
            print(f"Removing {max_vif_feature} (VIF={max_vif_value:.2f})")
            idx_to_remove = features_current.index(max_vif_feature)
            X_current = np.delete(X_current, idx_to_remove, axis=1)
            features_current.remove(max_vif_feature)
            self.removed_features.append((max_vif_feature, max_vif_value))

        return X_current, features_current

    def identify_correlation_groups(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        Identify groups of highly correlated features.

        Uses correlation matrix with threshold 0.8.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Dictionary mapping representative feature -> list of correlated features
        """
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(X_scaled.T)
        corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)

        # Find groups
        groups = {}
        processed = set()

        for i, feature_a in enumerate(feature_names):
            if feature_a in processed:
                continue

            # Find all features correlated with feature_a
            correlated = [feature_a]
            for j, feature_b in enumerate(feature_names):
                if i != j and feature_b not in processed:
                    corr = abs(corr_df.loc[feature_a, feature_b])
                    if corr > 0.8:
                        correlated.append(feature_b)

            if len(correlated) > 1:
                groups[feature_a] = correlated
                processed.update(correlated)

        return groups, corr_df


def load_training_data(npz_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load training data from NPZ file.

    Args:
        npz_path: Path to .npz training data file

    Returns:
        (X, feature_names)
    """
    data = np.load(npz_path)
    X = data['X']
    feature_names = list(data['feature_names'])

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Features: {feature_names}")

    return X, feature_names


def main(npz_path: str = "training_dataset_advanced.npz"):
    """
    Main VIF analysis pipeline.

    Args:
        npz_path: Path to training data NPZ file
    """
    print("=" * 80)
    print("VIF ANALYSIS - MULTICOLLINEARITY DETECTION")
    print("=" * 80)

    # Load data
    X, feature_names = load_training_data(npz_path)

    # Remove any NaN values
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    print(f"After removing NaNs: {X.shape[0]} samples")

    # Initialize analyzer
    analyzer = VIFAnalyzer(threshold=10.0)

    # Iteratively remove multicollinear features
    print("\n" + "=" * 80)
    print("ITERATIVE FEATURE REMOVAL")
    print("=" * 80)
    X_filtered, features_filtered = analyzer.remove_multicollinear_features(
        X, feature_names, threshold=10.0
    )

    # Final VIF scores
    print("\n" + "=" * 80)
    print("FINAL VIF SCORES")
    print("=" * 80)
    final_vif = analyzer.compute_vif(X_filtered, features_filtered)
    vif_df = pd.DataFrame(
        list(final_vif.items()), columns=["Feature", "VIF"]
    ).sort_values("VIF", ascending=False)
    print(vif_df.to_string(index=False))

    # Identify correlation groups
    print("\n" + "=" * 80)
    print("CORRELATION GROUPS (r > 0.8)")
    print("=" * 80)
    groups, corr_df = analyzer.identify_correlation_groups(X_filtered, features_filtered)
    for feature_a, correlated_features in groups.items():
        print(f"\nGroup: {feature_a}")
        for feature_b in correlated_features[1:]:
            corr = corr_df.loc[feature_a, feature_b]
            print(f"  ↔ {feature_b}: r={corr:.3f}")

    # Save results
    output_dir = Path("feature_selection_outputs")
    output_dir.mkdir(exist_ok=True)

    # VIF CSV
    vif_df.to_csv(output_dir / "vif_analysis.csv", index=False)
    print(f"\n✓ Saved VIF scores to vif_analysis.csv")

    # Removed features
    if analyzer.removed_features:
        removed_df = pd.DataFrame(
            analyzer.removed_features, columns=["Feature", "VIF_at_removal"]
        )
        removed_df.to_csv(output_dir / "vif_removed_features.csv", index=False)
        print(f"✓ Saved removed features to vif_removed_features.csv")

    # Recommended features
    recommended_txt = "\n".join(features_filtered)
    (output_dir / "vif_recommended_features.txt").write_text(recommended_txt)
    print(f"✓ Saved recommended features ({len(features_filtered)} total)")

    # Summary report
    summary = f"""VIF ANALYSIS SUMMARY
====================

Original features: {len(feature_names)}
Removed features: {len(analyzer.removed_features)}
Remaining features: {len(features_filtered)}

Removed (due to VIF > 10):
"""
    for feat, vif in analyzer.removed_features:
        summary += f"  - {feat}: VIF={vif:.2f}\n"

    summary += f"""
Recommended Features ({len(features_filtered)} total):
"""
    for feat in features_filtered:
        vif = final_vif.get(feat, np.inf)
        summary += f"  - {feat}: VIF={vif:.2f}\n"

    (output_dir / "vif_analysis_summary.txt").write_text(summary)
    print(f"✓ Saved summary report to vif_analysis_summary.txt")

    print("\n" + "=" * 80)
    print("VIF ANALYSIS COMPLETE")
    print("=" * 80)

    return X_filtered, features_filtered, analyzer


if __name__ == "__main__":
    main()
