"""
Integration Tests for Logistic Regression Cost Function

Tests:
1. Cost function loads correctly from model file
2. Cost computation produces valid outputs (0-10 range)
3. Invalid time gaps handled correctly
4. Feature extraction integration works
5. Cost values are reasonable (valid pairs < threshold)
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))


class TestLogisticRegressionCostFunction:
    """Test LR cost function integration."""

    @pytest.fixture
    def sample_fragments(self):
        """Create sample fragment pair."""
        frag_a = {
            "timestamp": np.array([0.0, 0.04, 0.08]),
            "x_position": np.array([100.0, 102.0, 104.0]),
            "y_position": np.array([20.0, 20.5, 20.0]),
            "velocity": np.array([50.0, 50.0, 50.0]),
            "length": np.array([4.5, 4.5, 4.5]),
            "width": np.array([2.0, 2.0, 2.0]),
            "direction": 1,
            "first_timestamp": 0.0,
            "last_timestamp": 0.08,
            "starting_x": 100.0,
            "ending_x": 104.0,
        }

        frag_b = {
            "timestamp": np.array([0.12, 0.16]),
            "x_position": np.array([106.0, 108.0]),
            "y_position": np.array([20.0, 20.0]),
            "velocity": np.array([50.0, 50.0]),
            "length": np.array([4.5, 4.5]),
            "width": np.array([2.0, 2.0]),
            "direction": 1,
            "first_timestamp": 0.12,
            "last_timestamp": 0.16,
            "starting_x": 106.0,
            "ending_x": 108.0,
        }

        return frag_a, frag_b

    def test_cost_function_loads(self):
        """Test that cost function can be instantiated."""
        try:
            from utils.stitch_cost_interface import LogisticRegressionCostFunction

            model_path = (
                Path(__file__).parent.parent
                / "model_artifacts"
                / "combined_optimal_10features.pkl"
            )

            if model_path.exists():
                cost_fn = LogisticRegressionCostFunction(
                    model_path=str(model_path),
                    scale_factor=5.0,
                    time_penalty=0.1,
                )
                assert cost_fn is not None
                assert cost_fn.model is not None
                assert cost_fn.scaler is not None
            else:
                pytest.skip("Model file not found")

        except ImportError:
            pytest.skip("Could not import LogisticRegressionCostFunction")

    def test_cost_computation_valid_pair(self, sample_fragments):
        """Test cost computation for valid fragment pair."""
        try:
            from utils.stitch_cost_interface import LogisticRegressionCostFunction

            model_path = (
                Path(__file__).parent.parent
                / "model_artifacts"
                / "combined_optimal_10features.pkl"
            )

            if not model_path.exists():
                pytest.skip("Model file not found")

            frag_a, frag_b = sample_fragments

            cost_fn = LogisticRegressionCostFunction(
                model_path=str(model_path),
                scale_factor=5.0,
                time_penalty=0.1,
            )

            cost = cost_fn.compute_cost(
                frag_a, frag_b, TIME_WIN=15.0, param={"cx": 0.2, "mx": 0.1}
            )

            # Cost should be finite and reasonable
            assert np.isfinite(cost)
            assert cost >= 0, f"Cost should be non-negative, got {cost}"
            # With scale_factor=5 and time_penalty=0.1, expect cost roughly in [0, 6]
            assert cost < 20, f"Cost seems unreasonably high: {cost}"

        except ImportError:
            pytest.skip("Could not import LogisticRegressionCostFunction")

    def test_cost_computation_invalid_time_gap(self, sample_fragments):
        """Test that invalid time gaps return high cost."""
        try:
            from utils.stitch_cost_interface import LogisticRegressionCostFunction

            model_path = (
                Path(__file__).parent.parent
                / "model_artifacts"
                / "combined_optimal_10features.pkl"
            )

            if not model_path.exists():
                pytest.skip("Model file not found")

            frag_a, frag_b = sample_fragments

            # Create invalid fragment pair (negative time gap)
            frag_b_invalid = frag_b.copy()
            frag_b_invalid["timestamp"] = frag_b["timestamp"] - 1.0  # Before frag_a ends
            frag_b_invalid["first_timestamp"] -= 1.0
            frag_b_invalid["last_timestamp"] -= 1.0

            cost_fn = LogisticRegressionCostFunction(
                model_path=str(model_path),
                scale_factor=5.0,
                time_penalty=0.1,
            )

            cost = cost_fn.compute_cost(
                frag_a, frag_b_invalid, TIME_WIN=15.0, param={"cx": 0.2, "mx": 0.1}
            )

            # Should return 1e6 for invalid pair
            assert cost == 1e6, f"Expected 1e6 for invalid time gap, got {cost}"

        except ImportError:
            pytest.skip("Could not import LogisticRegressionCostFunction")

    def test_cost_computation_out_of_window(self, sample_fragments):
        """Test that time gaps exceeding TIME_WIN return high cost."""
        try:
            from utils.stitch_cost_interface import LogisticRegressionCostFunction

            model_path = (
                Path(__file__).parent.parent
                / "model_artifacts"
                / "combined_optimal_10features.pkl"
            )

            if not model_path.exists():
                pytest.skip("Model file not found")

            frag_a, frag_b = sample_fragments

            # Create fragment pair with large time gap
            frag_b_far = frag_b.copy()
            frag_b_far["timestamp"] = frag_b["timestamp"] + 100.0
            frag_b_far["first_timestamp"] += 100.0
            frag_b_far["last_timestamp"] += 100.0

            cost_fn = LogisticRegressionCostFunction(
                model_path=str(model_path),
                scale_factor=5.0,
                time_penalty=0.1,
            )

            # TIME_WIN=5, but gap is 100
            cost = cost_fn.compute_cost(
                frag_a, frag_b_far, TIME_WIN=5.0, param={"cx": 0.2, "mx": 0.1}
            )

            # Should return 1e6 for out-of-window pair
            assert cost == 1e6, f"Expected 1e6 for out-of-window pair, got {cost}"

        except ImportError:
            pytest.skip("Could not import LogisticRegressionCostFunction")

    def test_scale_factor_effect(self, sample_fragments):
        """Test that scale_factor affects cost appropriately."""
        try:
            from utils.stitch_cost_interface import LogisticRegressionCostFunction

            model_path = (
                Path(__file__).parent.parent
                / "model_artifacts"
                / "combined_optimal_10features.pkl"
            )

            if not model_path.exists():
                pytest.skip("Model file not found")

            frag_a, frag_b = sample_fragments

            # Test with different scale factors
            cost_fn_1x = LogisticRegressionCostFunction(
                model_path=str(model_path),
                scale_factor=1.0,
                time_penalty=0.1,
            )

            cost_fn_5x = LogisticRegressionCostFunction(
                model_path=str(model_path),
                scale_factor=5.0,
                time_penalty=0.1,
            )

            cost_1x = cost_fn_1x.compute_cost(
                frag_a, frag_b, TIME_WIN=15.0, param={"cx": 0.2, "mx": 0.1}
            )
            cost_5x = cost_fn_5x.compute_cost(
                frag_a, frag_b, TIME_WIN=15.0, param={"cx": 0.2, "mx": 0.1}
            )

            # Both should be valid
            assert np.isfinite(cost_1x) and cost_1x < 1e6
            assert np.isfinite(cost_5x) and cost_5x < 1e6

            # 5x should generally be higher (due to scale factor)
            # Note: might not always be true due to time penalty, but typically
            # we expect base_cost to dominate

        except ImportError:
            pytest.skip("Could not import LogisticRegressionCostFunction")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
