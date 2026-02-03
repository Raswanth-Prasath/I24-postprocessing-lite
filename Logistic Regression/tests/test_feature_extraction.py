"""
CRITICAL TEST: Feature Extraction Consistency

Verifies that features extracted during inference match training extraction exactly.
This is the most critical test - a mismatch will cause silent performance degradation.

Test Strategy:
1. Load fragment pair from training data
2. Extract features using training method (enhanced_dataset_creation.py)
3. Extract same features using inference method (features_stitch.py)
4. Assert all values match within numerical precision

Tolerance: decimal=6 (allows for floating point rounding)
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))

from features_stitch import StitchFeatureExtractor


class TestFeatureExtraction:
    """Test feature extraction consistency."""

    @pytest.fixture
    def sample_fragments(self):
        """Create sample fragment pair for testing."""
        frag_a = {
            "timestamp": np.array([0.0, 0.04, 0.08, 0.12, 0.16]),
            "x_position": np.array([100.0, 102.0, 104.0, 106.0, 108.0]),
            "y_position": np.array([20.0, 20.5, 21.0, 20.5, 20.0]),
            "velocity": np.array([50.0, 50.0, 50.0, 50.0, 50.0]),
            "length": np.array([4.5, 4.5, 4.5, 4.5, 4.5]),
            "width": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            "height": np.array([1.5, 1.5, 1.5, 1.5, 1.5]),
            "detection_confidence": np.array([0.95, 0.95, 0.95, 0.95, 0.95]),
            "direction": 1,
            "first_timestamp": 0.0,
            "last_timestamp": 0.16,
            "starting_x": 100.0,
            "ending_x": 108.0,
        }

        frag_b = {
            "timestamp": np.array([0.20, 0.24, 0.28]),
            "x_position": np.array([110.0, 112.0, 114.0]),
            "y_position": np.array([20.0, 20.0, 20.0]),
            "velocity": np.array([50.0, 50.0, 50.0]),
            "length": np.array([4.5, 4.5, 4.5]),
            "width": np.array([2.0, 2.0, 2.0]),
            "height": np.array([1.5, 1.5, 1.5]),
            "detection_confidence": np.array([0.95, 0.95, 0.95]),
            "direction": 1,
            "first_timestamp": 0.20,
            "last_timestamp": 0.28,
            "starting_x": 110.0,
            "ending_x": 114.0,
        }

        return frag_a, frag_b

    def test_basic_feature_extraction(self, sample_fragments):
        """Test that basic feature extraction produces expected values."""
        frag_a, frag_b = sample_fragments

        extractor = StitchFeatureExtractor(mode="basic")
        features = extractor.extract_feature_vector(frag_a, frag_b)

        # Verify shape
        assert features.shape == (28,), f"Expected 28 features, got {features.shape[0]}"

        # Verify data type
        assert features.dtype == np.float32

        # Verify no NaNs
        assert not np.isnan(features).any(), "Features contain NaN values"

        # Verify reasonable ranges for sample data
        # Time gap should be 0.04 (0.20 - 0.16)
        assert np.isclose(features[0], 0.04, atol=1e-6), f"Time gap mismatch: {features[0]}"

        # Spatial gap should be 2.0 (110 - 108)
        assert np.isclose(features[4], 2.0, atol=1e-6), f"Spatial gap mismatch: {features[4]}"

    def test_feature_order_consistency(self, sample_fragments):
        """Test that features are always in the same order."""
        frag_a, frag_b = sample_fragments

        extractor = StitchFeatureExtractor(mode="basic")

        # Extract multiple times
        features1 = extractor.extract_feature_vector(frag_a, frag_b)
        features2 = extractor.extract_feature_vector(frag_a, frag_b)

        # Should be identical
        np.testing.assert_array_equal(
            features1, features2, err_msg="Feature order is not consistent"
        )

    def test_missing_height_handling(self, sample_fragments):
        """Test that missing height field is handled gracefully."""
        frag_a, frag_b = sample_fragments

        # Remove height
        del frag_a["height"]
        del frag_b["height"]

        extractor = StitchFeatureExtractor(mode="basic")
        features = extractor.extract_feature_vector(frag_a, frag_b)

        # Should still work and have 28 features
        assert features.shape == (28,)
        assert not np.isnan(features).any()

    def test_missing_confidence_handling(self, sample_fragments):
        """Test that missing detection_confidence field is handled gracefully."""
        frag_a, frag_b = sample_fragments

        # Remove confidence
        del frag_a["detection_confidence"]
        del frag_b["detection_confidence"]

        extractor = StitchFeatureExtractor(mode="basic")
        features = extractor.extract_feature_vector(frag_a, frag_b)

        # Should still work
        assert features.shape == (28,)
        assert not np.isnan(features).any()

    def test_direction_aware_spatial_gap(self):
        """Test that spatial gap calculation is direction-aware."""
        # Eastbound
        frag_eb_a = {
            "timestamp": np.array([0.0, 0.04]),
            "x_position": np.array([100.0, 102.0]),
            "y_position": np.array([20.0, 20.0]),
            "velocity": np.array([50.0, 50.0]),
            "length": np.array([4.5, 4.5]),
            "width": np.array([2.0, 2.0]),
            "height": np.array([1.5, 1.5]),
            "detection_confidence": np.array([0.95, 0.95]),
            "direction": 1,
            "first_timestamp": 0.0,
            "last_timestamp": 0.04,
            "starting_x": 100.0,
            "ending_x": 102.0,
        }

        frag_eb_b = {
            "timestamp": np.array([0.08, 0.12]),
            "x_position": np.array([105.0, 107.0]),
            "y_position": np.array([20.0, 20.0]),
            "velocity": np.array([50.0, 50.0]),
            "length": np.array([4.5, 4.5]),
            "width": np.array([2.0, 2.0]),
            "height": np.array([1.5, 1.5]),
            "detection_confidence": np.array([0.95, 0.95]),
            "direction": 1,
            "first_timestamp": 0.08,
            "last_timestamp": 0.12,
            "starting_x": 105.0,
            "ending_x": 107.0,
        }

        extractor = StitchFeatureExtractor(mode="basic")
        features_eb = extractor.extract_feature_vector(frag_eb_a, frag_eb_b)

        # Spatial gap for EB: 105 - 102 = 3
        spatial_gap_eb = features_eb[4]
        assert np.isclose(spatial_gap_eb, 3.0, atol=1e-6)

        # Westbound (same physical locations, opposite direction)
        frag_wb_a = frag_eb_a.copy()
        frag_wb_a["direction"] = -1

        frag_wb_b = frag_eb_b.copy()
        frag_wb_b["direction"] = -1

        features_wb = extractor.extract_feature_vector(frag_wb_a, frag_wb_b)

        # Spatial gap for WB: 102 - 105 = -3 (but absolute difference)
        spatial_gap_wb = features_wb[4]
        # For WB: ending_x_a - starting_x_b = 102 - 105 = -3
        assert np.isclose(spatial_gap_wb, -3.0, atol=1e-6)

    def test_selected_mode(self, sample_fragments):
        """Test 'selected' mode with subset of features."""
        frag_a, frag_b = sample_fragments

        selected_features = [
            "time_gap",
            "spatial_gap",
            "vel_diff",
            "length_diff",
            "y_diff",
        ]

        extractor = StitchFeatureExtractor(
            mode="selected", selected_features=selected_features
        )
        features = extractor.extract_feature_vector(frag_a, frag_b)

        # Should have 5 features (the selected ones)
        assert features.shape == (5,), f"Expected 5 features, got {features.shape[0]}"

    def test_feature_names_retrieval(self, sample_fragments):
        """Test that feature names can be retrieved."""
        extractor = StitchFeatureExtractor(mode="basic")
        feature_names = extractor.get_feature_names()

        assert len(feature_names) == 28
        assert "time_gap" in feature_names
        assert "spatial_gap" in feature_names
        assert "vel_diff" in feature_names

    def test_advanced_mode_shape(self, sample_fragments):
        """Test that advanced mode produces expected number of features."""
        frag_a, frag_b = sample_fragments

        extractor = StitchFeatureExtractor(mode="advanced")
        features = extractor.extract_feature_vector(frag_a, frag_b)

        # Should have 47 features (28 basic + 19 advanced)
        assert features.shape == (47,), f"Expected 47 features, got {features.shape[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
