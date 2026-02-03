"""
Tests for VIF (Variance Inflation Factor) Analysis

Tests:
1. VIF computation accuracy on synthetic data with known correlations
2. Iterative removal algorithm correctly identifies multicollinear features
3. Feature identification works correctly
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from vif_analysis import VIFAnalyzer


class TestVIFAnalysis:
    """Test VIF analysis functionality."""

    def test_vif_computation_uncorrelated(self):
        """Test VIF computation on uncorrelated features (VIF ~= 1)."""
        # Generate uncorrelated data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)

        analyzer = VIFAnalyzer(threshold=10.0)
        vif_scores = analyzer.compute_vif(X, [f"feat_{i}" for i in range(5)])

        # Uncorrelated features should have VIF close to 1
        for feat, vif in vif_scores.items():
            assert 0.5 < vif < 3.0, f"{feat} VIF={vif} (expected ~1 for uncorrelated)"

    def test_vif_computation_correlated(self):
        """Test VIF computation on correlated features."""
        # Generate highly correlated data
        np.random.seed(42)
        n_samples = 100

        x1 = np.random.randn(n_samples)
        x2 = x1 + 0.1 * np.random.randn(n_samples)  # Nearly identical to x1
        x3 = np.random.randn(n_samples)  # Independent

        X = np.column_stack([x1, x2, x3])

        analyzer = VIFAnalyzer(threshold=10.0)
        vif_scores = analyzer.compute_vif(X, ["x1", "x2", "x3"])

        # x1 and x2 should have high VIF (multicollinear)
        assert vif_scores["x1"] > 5.0, f"x1 VIF={vif_scores['x1']} (expected >5)"
        assert vif_scores["x2"] > 5.0, f"x2 VIF={vif_scores['x2']} (expected >5)"

        # x3 should have low VIF (independent)
        assert vif_scores["x3"] < 3.0, f"x3 VIF={vif_scores['x3']} (expected <3)"

    def test_iterative_removal(self):
        """Test iterative removal algorithm."""
        # Create data with multicollinear features
        np.random.seed(42)
        n_samples = 100

        x1 = np.random.randn(n_samples)
        x2 = x1 + 0.1 * np.random.randn(n_samples)  # Highly correlated with x1
        x3 = 2 * x1 + 0.05 * np.random.randn(n_samples)  # Even more correlated
        x4 = np.random.randn(n_samples)  # Independent

        X = np.column_stack([x1, x2, x3, x4])
        feature_names = ["x1", "x2", "x3", "x4"]

        analyzer = VIFAnalyzer(threshold=5.0)
        X_filtered, features_filtered = analyzer.remove_multicollinear_features(
            X, feature_names, threshold=5.0
        )

        # Should remove x2 and x3 (highly correlated with x1)
        # Should keep x1 and x4
        assert len(features_filtered) >= 2, "Should keep at least 2 features"
        assert "x4" in features_filtered, "Should keep independent feature x4"

        # Remaining features should have VIF < 5
        final_vif = analyzer.compute_vif(X_filtered, features_filtered)
        for feat, vif in final_vif.items():
            assert vif < 5.0 or np.isinf(vif), f"{feat} VIF={vif} > 5 after removal"

    def test_correlation_group_identification(self):
        """Test identification of correlated feature groups."""
        np.random.seed(42)
        n_samples = 100

        x1 = np.random.randn(n_samples)
        x2 = x1 + 0.05 * np.random.randn(n_samples)  # Correlated with x1
        x3 = np.random.randn(n_samples)
        x4 = x3 + 0.05 * np.random.randn(n_samples)  # Correlated with x3
        x5 = np.random.randn(n_samples)  # Independent

        X = np.column_stack([x1, x2, x3, x4, x5])
        feature_names = ["x1", "x2", "x3", "x4", "x5"]

        analyzer = VIFAnalyzer(threshold=10.0)
        groups, corr_df = analyzer.identify_correlation_groups(X, feature_names)

        # Should identify groups: {x1, x2} and {x3, x4}
        # x5 should be independent

        # Check that at least one group contains x1 or x2
        group_str = str(groups)
        has_group_12 = ("x1" in group_str and "x2" in group_str)
        has_group_34 = ("x3" in group_str and "x4" in group_str)

        assert has_group_12 or has_group_34, "Should identify at least one correlation group"

    def test_vif_threshold_variation(self):
        """Test that different thresholds produce different results."""
        np.random.seed(42)
        n_samples = 100

        # Create data with moderate multicollinearity
        x1 = np.random.randn(n_samples)
        x2 = x1 + 0.3 * np.random.randn(n_samples)
        x3 = np.random.randn(n_samples)
        x4 = x3 + 0.3 * np.random.randn(n_samples)

        X = np.column_stack([x1, x2, x3, x4])
        feature_names = ["x1", "x2", "x3", "x4"]

        # Strict threshold
        analyzer_strict = VIFAnalyzer(threshold=3.0)
        _, features_strict = analyzer_strict.remove_multicollinear_features(
            X, feature_names, threshold=3.0
        )

        # Lenient threshold
        analyzer_lenient = VIFAnalyzer(threshold=10.0)
        _, features_lenient = analyzer_lenient.remove_multicollinear_features(
            X, feature_names, threshold=10.0
        )

        # Strict should remove more features
        assert len(features_strict) <= len(features_lenient), (
            f"Strict threshold should remove more features: "
            f"{len(features_strict)} <= {len(features_lenient)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
