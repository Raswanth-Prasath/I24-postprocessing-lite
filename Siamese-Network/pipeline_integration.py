"""
Pipeline Integration for Siamese Network

Integrates the trained Siamese network into the I24-postprocessing-lite pipeline
to replace the Bhattacharyya distance cost function.
"""

import torch
import numpy as np
from pathlib import Path
from siamese_model import SiameseTrajectoryNetwork


class SiameseCostFunction:
    """
    Wrapper for Siamese network to use as cost function in NCC algorithm

    Replaces the Bhattacharyya distance in utils/utils_stitcher_cost.py
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 model_config: dict = None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
            model_config: Model architecture configuration
        """
        self.device = torch.device(device)

        # Default model configuration (must match training)
        if model_config is None:
            model_config = {
                'input_size': 4,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': True,
                'similarity_hidden_dim': 64
            }

        # Load model
        self.model = SiameseTrajectoryNetwork(**model_config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded Siamese network from {checkpoint_path}")
        print(f"Using device: {self.device}")

        # Normalization statistics (should match training)
        # These should ideally be saved with the model
        self.norm_stats = None

    def _extract_sequence(self, fragment: dict) -> np.ndarray:
        """
        Extract trajectory sequence from fragment dictionary

        Args:
            fragment: Fragment dictionary from I24-postprocessing-lite

        Returns:
            Array of shape (seq_len, 4) with [x_rel, y_rel, velocity, time_normalized]
        """
        # Get data
        timestamps = np.array(fragment['timestamp'])
        x_pos = np.array(fragment['x_position'])
        y_pos = np.array(fragment['y_position'])

        # 2. Use Relative Coordinates (Translation Invariance)
        x_rel = x_pos - x_pos[0]
        y_rel = y_pos - y_pos[0]

        # Get or compute velocity
        if 'velocity' in fragment and len(fragment['velocity']) > 0:
            velocity = np.array(fragment['velocity'])
        else:
            dt = np.diff(timestamps)
            dx = np.diff(x_pos)
            velocity = np.zeros(len(timestamps))
            velocity[1:] = dx / (dt + 1e-6)
            velocity[0] = velocity[1]

        # Normalize timestamps to start at 0
        t_norm = timestamps - timestamps[0]

        # Build sequence
        sequence = np.column_stack([x_rel, y_rel, velocity, t_norm])

        return sequence.astype(np.float32)

    def _extract_endpoint_features(self, frag_a: dict, frag_b: dict) -> np.ndarray:
        """Extract endpoint features (gap metrics) for similarity head"""
        t_a_end = frag_a['timestamp'][-1]
        t_b_start = frag_b['timestamp'][0]
        time_gap = t_b_start - t_a_end

        x_a_end = frag_a['x_position'][-1]
        x_b_start = frag_b['x_position'][0]
        x_gap = x_b_start - x_a_end

        y_gap = frag_b['y_position'][0] - frag_a['y_position'][-1]

        # Velocity difference
        if 'velocity' in frag_a and len(frag_a['velocity']) > 0:
            v_a_end = frag_a['velocity'][-1]
        else:
            dt = frag_a['timestamp'][-1] - frag_a['timestamp'][-2] if len(frag_a['timestamp']) > 1 else 1e-6
            dx = frag_a['x_position'][-1] - frag_a['x_position'][-2] if len(frag_a['timestamp']) > 1 else 0
            v_a_end = dx / (dt + 1e-6)

        if 'velocity' in frag_b and len(frag_b['velocity']) > 0:
            v_b_start = frag_b['velocity'][0]
        else:
            dt = frag_b['timestamp'][1] - frag_b['timestamp'][0] if len(frag_b['timestamp']) > 1 else 1e-6
            dx = frag_b['x_position'][1] - frag_b['x_position'][0] if len(frag_b['timestamp']) > 1 else 0
            v_b_start = dx / (dt + 1e-6)

        velocity_diff = v_b_start - v_a_end

        return np.array([time_gap, x_gap, y_gap, velocity_diff], dtype=np.float32)

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize sequence (if normalization stats available)"""
        if self.norm_stats is not None:
            return (sequence - self.norm_stats['mean']) / self.norm_stats['std']
        return sequence

    @torch.no_grad()
    def compute_similarity(self, track1: dict, track2: dict) -> float:
        """
        Compute similarity score between two fragments using Siamese network

        Args:
            track1: First fragment dictionary
            track2: Second fragment dictionary

        Returns:
            Similarity score in [0, 1] (higher = more likely same vehicle)
        """
        # Extract sequences
        seq_a = self._extract_sequence(track1)
        seq_b = self._extract_sequence(track2)
        
        # Extract endpoint features
        endpoints = self._extract_endpoint_features(track1, track2)

        # Normalize
        seq_a = self._normalize_sequence(seq_a)
        seq_b = self._normalize_sequence(seq_b)

        # Convert to tensors
        seq_a = torch.FloatTensor(seq_a).unsqueeze(0).to(self.device)  # (1, len_a, 4)
        seq_b = torch.FloatTensor(seq_b).unsqueeze(0).to(self.device)  # (1, len_b, 4)
        len_a = torch.LongTensor([seq_a.size(1)]).to(self.device)
        len_b = torch.LongTensor([seq_b.size(1)]).to(self.device)
        endpoints = torch.FloatTensor(endpoints).unsqueeze(0).to(self.device)

        # Forward pass
        similarity, _, _ = self.model(seq_a, len_a, seq_b, len_b, endpoints)

        # Return as float
        return similarity.item()

    def stitch_cost(self, track1: dict, track2: dict, TIME_WIN: float, param: dict) -> float:
        """
        Cost function for fragment stitching (replaces original stitch_cost)

        This function signature matches the original Bhattacharyya distance cost function
        in utils/utils_stitcher_cost.py

        Args:
            track1: First fragment
            track2: Second fragment
            TIME_WIN: Maximum time window for stitching
            param: Parameters (not used, kept for compatibility)

        Returns:
            Cost value (lower = more likely to stitch)
        """
        try:
            # Check time gap
            t1 = np.array(track1['timestamp'])
            t2 = np.array(track2['timestamp'])
            gap = t2[0] - t1[-1]

            if gap < 0 or gap > TIME_WIN:
                return 1e6  # Invalid pair

            # Compute similarity using Siamese network
            similarity = self.compute_similarity(track1, track2)

            # Convert similarity to cost
            # similarity ∈ [0,1], higher is better
            # cost should be lower for better matches
            base_cost = 1.0 - similarity

            # Add time penalty (same as original)
            time_cost = 0.1 * gap

            total_cost = base_cost + time_cost

            return total_cost

        except Exception as e:
            print(f"Error in Siamese cost computation: {e}")
            return 1e6  # Return high cost on error


def create_integration_example():
    """
    Example of how to integrate Siamese cost function into the pipeline
    """
    print("\n" + "="*60)
    print("SIAMESE NETWORK PIPELINE INTEGRATION EXAMPLE")
    print("="*60)

    # Path to trained model - use relative path
    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / "outputs" / "best_accuracy.pth"

    if not checkpoint_path.exists():
        print(f"\nError: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_siamese.py")
        return

    # Create cost function wrapper
    siamese_cost = SiameseCostFunction(str(checkpoint_path))

    print("\n✓ Siamese cost function initialized")
    print("\nIntegration instructions:")
    print("1. Copy this file to: 02_Code/I24-postprocessing-lite/utils/")
    print("2. Modify min_cost_flow.py to import SiameseCostFunction")
    print("3. Replace stitch_cost calls with siamese_cost.stitch_cost")

    print("\nExample usage in min_cost_flow.py:")
    print("```python")
    print("from utils.pipeline_integration import SiameseCostFunction")
    print("")
    print("# Initialize once at module level")
    print("siamese_cost = SiameseCostFunction('path/to/best_accuracy.pth')")
    print("")
    print("# Use in MOTGraphSingle class")
    print("cost = siamese_cost.stitch_cost(track1, track2, TIME_WIN, param)")
    print("```")

    # Test with dummy data
    print("\n" + "="*60)
    print("TESTING WITH DUMMY DATA")
    print("="*60)

    dummy_fragment_1 = {
        'timestamp': np.linspace(0, 2, 50),
        'x_position': np.linspace(0, 100, 50),
        'y_position': np.ones(50) * 10,
        'velocity': np.ones(50) * 50
    }

    dummy_fragment_2 = {
        'timestamp': np.linspace(2.5, 4, 40),
        'x_position': np.linspace(110, 200, 40),
        'y_position': np.ones(40) * 10.5,
        'velocity': np.ones(40) * 48
    }

    # Compute cost
    cost = siamese_cost.stitch_cost(
        dummy_fragment_1,
        dummy_fragment_2,
        TIME_WIN=5.0,
        param={}
    )

    print(f"\nTest cost computed: {cost:.4f}")
    print("(Lower cost = more likely to be same vehicle)")

    print("\n" + "="*60)
    print("INTEGRATION READY!")
    print("="*60)


def create_modified_utils_stitcher_cost():
    """
    Create a modified version of utils_stitcher_cost.py that uses Siamese network

    This function generates the code to replace the original cost function
    """
    code = '''
"""
Modified utils_stitcher_cost.py with Siamese Network Integration

This version uses a learned Siamese neural network instead of Bhattacharyya distance
"""

import numpy as np
from pipeline_integration import SiameseCostFunction

# Initialize Siamese cost function (load model once)
SIAMESE_COST = None

def initialize_siamese_cost(checkpoint_path: str):
    """Initialize the Siamese cost function (call this once at startup)"""
    global SIAMESE_COST
    SIAMESE_COST = SiameseCostFunction(checkpoint_path)
    print("Siamese cost function initialized")


def stitch_cost(track1, track2, TIME_WIN, param):
    """
    Compute stitching cost using Siamese neural network

    This replaces the original Bhattacharyya distance cost function
    """
    global SIAMESE_COST

    if SIAMESE_COST is None:
        # Fallback to default checkpoint if not initialized
        checkpoint_path = r"D:\\ASU Academics\\Thesis & Research\\02_Code\\Siamese-Network\\outputs\\best_accuracy.pth"
        initialize_siamese_cost(checkpoint_path)

    return SIAMESE_COST.stitch_cost(track1, track2, TIME_WIN, param)


# Keep original Bhattacharyya cost as backup
def stitch_cost_bhattacharyya(track1, track2, TIME_WIN, param):
    """Original Bhattacharyya distance cost (kept for comparison)"""
    # ... [original implementation] ...
    pass
'''

    output_path = Path(r"D:\ASU Academics\Thesis & Research\02_Code\Siamese-Network\utils_stitcher_cost_siamese.py")
    with open(output_path, 'w') as f:
        f.write(code)

    print(f"\nModified cost function saved to: {output_path}")
    print("Copy this to: 02_Code/I24-postprocessing-lite/utils/utils_stitcher_cost.py")


if __name__ == "__main__":
    # Run integration example
    create_integration_example()

    # Generate modified cost function
    print("\n" + "="*60)
    print("GENERATING MODIFIED COST FUNCTION")
    print("="*60)
    create_modified_utils_stitcher_cost()
