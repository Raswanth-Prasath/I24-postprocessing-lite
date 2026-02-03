"""
Unified Feature Selection Pipeline

Combines multiple feature selection methods with consensus voting:
1. VIF Filtering (mandatory) - removes multicollinear features
2. L1 Lasso Selection - sparse feature selection
3. Permutation Importance - model-agnostic importance
4. RFE (Recursive Feature Elimination) - iterative selection
5. Statistical Testing (p-values) - significance filtering
6. Consensus Voting - select features approved by multiple methods

Recommendation: Features selected by ≥2 methods are reliable
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
from collections import defaultdict

from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import RFE, permutation_importance
from sklearn.preprocessing import StandardScaler
from scipy import stats


class UnifiedFeatureSelector:
    """
    Multi-method feature selection with consensus voting.
    """

    def __init__(self, vif_threshold: float = 10.0, consensus_votes: int = 2):
        """
        Args:
            vif_threshold: VIF threshold for removing multicollinear features
            consensus_votes: Minimum number of methods agreeing on a feature (for consensus)
        """
        self.vif_threshold = vif_threshold
        self.consensus_votes = consensus_votes

        self.vif_features = []
        self.lasso_features = []
        self.permutation_features = []
        self.rfe_features = []
        self.pvalue_features = []

        self.method_scores = {}

    def stage1_vif_filtering(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Stage 1: VIF-based filtering to remove multicollinear features.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            (X_filtered, features_remaining, features_removed)
        """
        from sklearn.linear_model import LinearRegression

        print("\n" + "=" * 80)
        print("STAGE 1: VIF FILTERING (MANDATORY)")
        print("=" * 80)

        vif_scores = {}
        features_current = list(feature_names)
        X_current = X.copy()
        removed = []

        # Iteratively remove highest VIF features
        max_iterations = len(feature_names)
        for iteration in range(max_iterations):
            # Compute VIF
            for i, feature_name in enumerate(features_current):
                X_i = np.delete(X_current, i, axis=1)
                y_i = X_current[:, i]

                lr = LinearRegression()
                lr.fit(X_i, y_i)
                r_squared = lr.score(X_i, y_i)
                vif = 1.0 / (1.0 - r_squared + 1e-8)
                vif_scores[feature_name] = vif

            # Find max VIF
            max_vif_feature = max(vif_scores, key=vif_scores.get)
            max_vif = vif_scores[max_vif_feature]

            if max_vif <= self.vif_threshold:
                break

            # Remove feature
            idx = features_current.index(max_vif_feature)
            X_current = np.delete(X_current, idx, axis=1)
            features_current.remove(max_vif_feature)
            removed.append(max_vif_feature)
            print(f"Removed {max_vif_feature} (VIF={max_vif:.2f})")

        print(f"\nVIF Filtering complete:")
        print(f"  Removed: {len(removed)} features")
        print(f"  Remaining: {len(features_current)} features")

        self.vif_features = features_current
        return X_current, features_current, removed

    def stage2_parallel_methods(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> None:
        """
        Stage 2-5: Run four parallel feature selection methods.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        print("\n" + "=" * 80)
        print("STAGE 2-5: PARALLEL FEATURE SELECTION METHODS")
        print("=" * 80)

        n_features_target = 10  # Target feature count for RFE

        # Method 1: L1 Lasso (elastic net)
        print("\n--- Method 1: L1 Lasso Selection ---")
        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000)
        lasso_cv.fit(X, y)
        lasso_coef = np.abs(lasso_cv.coef_)
        lasso_threshold = np.percentile(lasso_coef, 60)  # Top 40%
        self.lasso_features = [
            feature_names[i]
            for i in np.where(lasso_coef >= lasso_threshold)[0]
        ]
        print(f"Selected {len(self.lasso_features)} features with Lasso")
        print(f"  Features: {self.lasso_features}")

        # Method 2: Permutation Importance
        print("\n--- Method 2: Permutation Importance ---")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        perm_result = permutation_importance(
            lr, X, y, n_repeats=10, random_state=42
        )
        importance_threshold = np.percentile(perm_result.importances_mean, 60)
        self.permutation_features = [
            feature_names[i]
            for i in np.where(perm_result.importances_mean >= importance_threshold)[
                0
            ]
        ]
        print(f"Selected {len(self.permutation_features)} features with permutation importance")
        print(f"  Features: {self.permutation_features}")

        # Method 3: RFE
        print("\n--- Method 3: Recursive Feature Elimination (RFE) ---")
        rfe = RFE(lr, n_features_to_select=n_features_target, step=1)
        rfe.fit(X, y)
        self.rfe_features = [
            feature_names[i] for i, selected in enumerate(rfe.support_) if selected
        ]
        print(f"Selected {len(self.rfe_features)} features with RFE")
        print(f"  Features: {self.rfe_features}")

        # Method 4: Statistical Significance (p-values)
        print("\n--- Method 4: Statistical Testing (p-values) ---")
        lr_coef = lr.coef_[0]
        # Approximate p-values using z-scores
        # Assumes normal approximation for logistic regression
        pvalues = [
            2 * (1 - stats.norm.cdf(abs(coef)))
            for coef in lr_coef
        ]
        pvalue_threshold = 0.05
        self.pvalue_features = [
            feature_names[i]
            for i, pval in enumerate(pvalues)
            if pval < pvalue_threshold
        ]
        print(
            f"Selected {len(self.pvalue_features)} features with p < {pvalue_threshold}"
        )
        print(f"  Features: {self.pvalue_features}")

    def stage6_consensus_voting(self, feature_names: List[str]) -> Dict[str, int]:
        """
        Stage 6: Consensus voting across all five methods.

        Methods:
        1. VIF filtering (mandatory - features must pass)
        2. Lasso
        3. Permutation importance
        4. RFE
        5. P-values

        Args:
            feature_names: All features to consider

        Returns:
            Dictionary mapping feature -> vote count
        """
        print("\n" + "=" * 80)
        print("STAGE 6: CONSENSUS VOTING")
        print("=" * 80)

        # Count votes for each feature
        votes = defaultdict(int)

        # All methods start with features that passed VIF
        for feature in self.vif_features:
            votes[feature] += 1  # VIF pass counts as baseline

        # Add votes from other methods
        for feature in self.lasso_features:
            votes[feature] += 1

        for feature in self.permutation_features:
            votes[feature] += 1

        for feature in self.rfe_features:
            votes[feature] += 1

        for feature in self.pvalue_features:
            votes[feature] += 1

        # Sort by votes
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

        print("\nVotes by feature:")
        print("Feature | Votes | Methods")
        print("-" * 60)
        for feature, vote_count in sorted_votes:
            method_names = []
            if feature in self.vif_features:
                method_names.append("VIF")
            if feature in self.lasso_features:
                method_names.append("Lasso")
            if feature in self.permutation_features:
                method_names.append("Perm")
            if feature in self.rfe_features:
                method_names.append("RFE")
            if feature in self.pvalue_features:
                method_names.append("PVal")

            methods_str = ", ".join(method_names)
            print(f"{feature:30s} | {vote_count:5d} | {methods_str}")

        return dict(sorted_votes)

    def get_recommended_sets(self, votes: Dict[str, int]) -> Dict[str, List[str]]:
        """
        Generate recommended feature sets at different sizes.

        Args:
            votes: Dictionary mapping feature -> vote count

        Returns:
            Dictionary with keys: 'minimal' (5), 'optimal' (10), 'maximal' (15)
        """
        sorted_features = sorted(votes.items(), key=lambda x: x[1], reverse=True)

        recommended_sets = {
            "minimal": [f for f, _ in sorted_features[:5]],
            "optimal": [f for f, _ in sorted_features[:10]],
            "maximal": [f for f, _ in sorted_features[:15]],
        }

        return recommended_sets


def load_training_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load training data from NPZ file."""
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    feature_names = list(data["feature_names"])

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y, feature_names


def main(npz_path: str = "training_dataset_advanced.npz"):
    """
    Main unified feature selection pipeline.

    Args:
        npz_path: Path to training data NPZ file
    """
    print("=" * 80)
    print("UNIFIED FEATURE SELECTION PIPELINE")
    print("=" * 80)

    # Load data
    X, y, feature_names = load_training_data(npz_path)

    # Remove NaN
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    print(f"After removing NaNs: {X.shape[0]} samples")

    # Standardize (important for many methods)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize selector
    selector = UnifiedFeatureSelector(vif_threshold=10.0, consensus_votes=2)

    # Stage 1: VIF filtering
    X_vif, features_after_vif, removed_features = selector.stage1_vif_filtering(
        X_scaled, feature_names
    )

    # Update feature names and data after VIF filtering
    feature_names_vif = features_after_vif

    # Stage 2-5: Run parallel methods
    selector.stage2_parallel_methods(X_vif, y, feature_names_vif)

    # Stage 6: Consensus voting
    votes = selector.stage6_consensus_voting(feature_names_vif)

    # Get recommended sets
    recommended_sets = selector.get_recommended_sets(votes)

    # Save results
    output_dir = Path("feature_selection_outputs")
    output_dir.mkdir(exist_ok=True)

    # Voting results CSV
    votes_df = pd.DataFrame(
        [(feature, count) for feature, count in sorted(votes.items(), key=lambda x: -x[1])],
        columns=["Feature", "Votes"]
    )
    votes_df.to_csv(output_dir / "unified_selection_votes.csv", index=False)
    print(f"\n✓ Saved voting results")

    # Recommended feature sets
    for set_name, features in recommended_sets.items():
        output_file = output_dir / f"recommended_features_{set_name}.txt"
        output_file.write_text("\n".join(features))
        print(f"✓ Saved {set_name} feature set ({len(features)} features)")

    # Comprehensive report
    report = f"""UNIFIED FEATURE SELECTION REPORT
=================================

Pipeline Configuration:
  - VIF threshold: 10.0
  - Consensus vote threshold: 2+

Stage 1 Results (VIF Filtering):
  - Original features: {len(feature_names)}
  - Removed: {len(removed_features)}
  - Remaining: {len(feature_names_vif)}

Stage 2-5 Results (Parallel Methods):
  - Lasso selected: {len(selector.lasso_features)} features
  - Permutation importance selected: {len(selector.permutation_features)} features
  - RFE selected: {len(selector.rfe_features)} features
  - P-value (p<0.05) selected: {len(selector.pvalue_features)} features

Stage 6 Results (Consensus Voting):
  - Max votes: {max(votes.values()) if votes else 0}
  - Min votes: {min(votes.values()) if votes else 0}
  - Mean votes: {np.mean(list(votes.values())):.2f}

Recommended Feature Sets:
"""
    for set_name, features in recommended_sets.items():
        report += f"\n{set_name.upper()} ({len(features)} features):\n"
        for feature in features:
            votes_count = votes.get(feature, 0)
            report += f"  - {feature} ({votes_count} votes)\n"

    (output_dir / "unified_selection_report.md").write_text(report)
    print(f"✓ Saved comprehensive report")

    print("\n" + "=" * 80)
    print("UNIFIED FEATURE SELECTION COMPLETE")
    print("=" * 80)

    return selector, recommended_sets, votes


if __name__ == "__main__":
    main()
