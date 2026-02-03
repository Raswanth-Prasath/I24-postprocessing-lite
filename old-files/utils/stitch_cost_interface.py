"""
Stitch Cost Function Interface

Abstraction layer for fragment stitching cost functions using Strategy pattern.
Allows switching between Bhattacharyya distance and learned Siamese network.
"""

from abc import ABC, abstractmethod
import sys
from pathlib import Path


class StitchCostFunction(ABC):
    """Abstract base class for stitch cost functions"""

    @abstractmethod
    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        """
        Compute cost of stitching two fragments.

        Args:
            track1: First fragment dictionary with keys:
                    timestamp, x_position, y_position, velocity, direction, etc.
            track2: Second fragment dictionary
            TIME_WIN: Maximum time window for valid stitching
            param: Additional parameters (cx, mx, cy, my, stitch_thresh, etc.)

        Returns:
            Cost value (lower = better match). Returns 1e6 for invalid pairs.
        """
        pass


class BhattacharyyaCostFunction(StitchCostFunction):
    """
    Original Bhattacharyya distance-based cost function.

    Wraps the existing stitch_cost() from utils_stitcher_cost.py
    """

    def __init__(self):
        from utils.utils_stitcher_cost import stitch_cost
        self._stitch_cost = stitch_cost

    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        """Compute Bhattacharyya distance-based cost"""
        return self._stitch_cost(track1, track2, TIME_WIN, param)


class SiameseCostFunction(StitchCostFunction):
    """
    Learned Siamese network-based cost function.

    Uses trained neural network to compute similarity between fragment pairs.
    """

    def __init__(self, checkpoint_path: str, device: str = None, model_config: dict = None):
        """
        Initialize Siamese cost function.

        Args:
            checkpoint_path: Path to trained model checkpoint (.pth file)
            device: Device for inference ('cuda', 'cpu', or None for auto)
            model_config: Model architecture configuration (optional)
        """
        import torch
        import numpy as np

        self.np = np
        self.torch = torch

        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Add Siamese-Network directory to path for imports
        siamese_dir = Path(__file__).parent.parent / "Siamese-Network"
        if str(siamese_dir) not in sys.path:
            sys.path.insert(0, str(siamese_dir))

        from siamese_model import SiameseTrajectoryNetwork

        # Default model configuration (must match training)
        if model_config is None:
            model_config = {
                'input_size': 4,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': True,
                'similarity_hidden_dim': 64,
                'endpoint_feature_dim': 4,
                'use_endpoint_features': True
            }

        # Load model
        self.model = SiameseTrajectoryNetwork(**model_config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[SiameseCostFunction] Loaded model from {checkpoint_path}")
        print(f"[SiameseCostFunction] Using device: {self.device}")

    def _extract_sequence(self, fragment: dict):
        """
        Extract trajectory sequence from fragment dictionary.

        Returns:
            Array of shape (seq_len, 4) with [x, y, velocity, time_normalized]
        """
        np = self.np

        timestamps = np.array(fragment['timestamp'])
        x_pos = np.array(fragment['x_position'])
        y_pos = np.array(fragment['y_position'])
        seq_len = len(timestamps)

        # Get or compute velocity - ensure same length as other arrays
        if 'velocity' in fragment and fragment['velocity'] is not None:
            vel_arr = np.array(fragment['velocity'])
            if len(vel_arr) == seq_len:
                velocity = vel_arr
            else:
                # Length mismatch - recompute velocity
                velocity = self._compute_velocity(timestamps, x_pos)
        else:
            velocity = self._compute_velocity(timestamps, x_pos)

        # Normalize timestamps to start at 0
        t_norm = timestamps - timestamps[0]

        # Build sequence
        sequence = np.column_stack([x_pos, y_pos, velocity, t_norm])

        return sequence.astype(np.float32)

    def _compute_velocity(self, timestamps, x_pos):
        """Compute velocity from positions and timestamps."""
        np = self.np
        dt = np.diff(timestamps)
        dx = np.diff(x_pos)
        velocity = np.zeros(len(timestamps))
        velocity[1:] = dx / (dt + 1e-6)
        velocity[0] = velocity[1] if len(velocity) > 1 else 0
        return velocity

    @property
    def _inference_mode(self):
        """Context manager for inference (no gradients)"""
        return self.torch.no_grad()

    def _compute_endpoint_features(self, track1: dict, track2: dict):
        """
        Compute endpoint features for the similarity head.

        Returns:
            Array of shape (4,) with [time_gap, x_gap, y_gap, velocity_diff]
        """
        np = self.np

        t1 = np.array(track1['timestamp'])
        t2 = np.array(track2['timestamp'])
        x1 = np.array(track1['x_position'])
        x2 = np.array(track2['x_position'])
        y1 = np.array(track1['y_position'])
        y2 = np.array(track2['y_position'])

        # Time gap
        time_gap = t2[0] - t1[-1]

        # Spatial gaps
        x_gap = x2[0] - x1[-1]
        y_gap = y2[0] - y1[-1]

        # Velocity difference
        v1 = self._compute_velocity(t1, x1)
        v2 = self._compute_velocity(t2, x2)
        velocity_diff = v2[0] - v1[-1]

        return np.array([time_gap, x_gap, y_gap, velocity_diff], dtype=np.float32)

    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        """
        Compute cost using Siamese network.

        Similarity is converted to cost: cost = (1 - similarity) + 0.1 * time_gap
        """
        np = self.np
        torch = self.torch

        try:
            # Check time gap
            t1 = np.array(track1['timestamp'])
            t2 = np.array(track2['timestamp'])
            gap = t2[0] - t1[-1]

            if gap < 0 or gap > TIME_WIN:
                return 1e6  # Invalid pair

            # Extract sequences
            seq_a = self._extract_sequence(track1)
            seq_b = self._extract_sequence(track2)

            # Compute endpoint features
            endpoint_features = self._compute_endpoint_features(track1, track2)

            # Convert to tensors
            with self._inference_mode:
                seq_a_t = torch.FloatTensor(seq_a).unsqueeze(0).to(self.device)
                seq_b_t = torch.FloatTensor(seq_b).unsqueeze(0).to(self.device)
                len_a = torch.LongTensor([seq_a_t.size(1)]).to(self.device)
                len_b = torch.LongTensor([seq_b_t.size(1)]).to(self.device)
                endpoint_t = torch.FloatTensor(endpoint_features).unsqueeze(0).to(self.device)

                # Forward pass with endpoint features
                similarity, _, _ = self.model(seq_a_t, len_a, seq_b_t, len_b, endpoint_t)
                similarity = similarity.item()

            # Convert similarity to cost
            # similarity in [0, 1], higher = better match
            # cost should be lower for better matches
            base_cost = 1.0 - similarity

            # Add time penalty (same as original Bhattacharyya)
            time_cost = 0.1 * gap

            total_cost = base_cost + time_cost

            return total_cost

        except Exception as e:
            print(f"[SiameseCostFunction] Error computing cost: {e}")
            return 1e6  # Return high cost on error


class CostFunctionFactory:
    """Factory for creating cost function instances"""

    @staticmethod
    def create(config: dict) -> StitchCostFunction:
        """
        Create a cost function based on configuration.

        Args:
            config: Configuration dictionary with keys:
                - type: 'bhattacharyya' or 'siamese'
                - checkpoint_path: Path to model checkpoint (for siamese)
                - device: Device for inference (for siamese, optional)
                - model_config: Model configuration (for siamese, optional)

        Returns:
            StitchCostFunction instance
        """
        cost_type = config.get('type', 'bhattacharyya').lower()

        if cost_type == 'bhattacharyya':
            return BhattacharyyaCostFunction()

        elif cost_type == 'siamese':
            checkpoint_path = config.get('checkpoint_path')
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required for siamese cost function")

            device = config.get('device', None)
            model_config = config.get('model_config', None)

            return SiameseCostFunction(
                checkpoint_path=checkpoint_path,
                device=device,
                model_config=model_config
            )

        else:
            raise ValueError(f"Unknown cost function type: {cost_type}. "
                           f"Supported: 'bhattacharyya', 'siamese'")


# Convenience function for backward compatibility
def get_cost_function(parameters: dict) -> StitchCostFunction:
    """
    Get cost function from parameters configuration.

    Args:
        parameters: Full parameters dictionary (from parameters.json)

    Returns:
        StitchCostFunction instance
    """
    cost_config = parameters.get('cost_function', {'type': 'bhattacharyya'})
    return CostFunctionFactory.create(cost_config)
