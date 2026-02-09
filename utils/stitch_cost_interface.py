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

    def __init__(self, checkpoint_path: str, device: str = None, model_config: dict = None,
                 scale_factor: float = 5.0, time_penalty: float = 0.1):
        """
        Initialize Siamese cost function.

        Args:
            checkpoint_path: Path to trained model checkpoint (.pth file)
            device: Device for inference ('cuda', 'cpu', or None for auto)
            model_config: Model architecture configuration (optional)
            scale_factor: Scale factor to adjust cost range (default 5.0)
            time_penalty: Weight for time gap penalty (default 0.1)
        """
        import torch
        import numpy as np

        self.np = np
        self.torch = torch
        self.scale_factor = scale_factor
        self.time_penalty = time_penalty

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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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

            # Convert similarity to cost (scaled to match Bhattacharyya range)
            # similarity in [0, 1], higher = better match
            # cost should be lower for better matches
            base_cost = (1.0 - similarity) * self.scale_factor

            # Add time penalty
            time_cost = self.time_penalty * gap

            total_cost = base_cost + time_cost

            return total_cost

        except Exception as e:
            print(f"[SiameseCostFunction] Error computing cost: {e}")
            return 1e6  # Return high cost on error


class LogisticRegressionCostFunction(StitchCostFunction):
    """
    Logistic Regression-based cost function.

    Uses a pre-trained sklearn logistic regression model to predict the probability
    that two fragments belong to the same vehicle. Converts probability to cost.

    Cost Formula:
        cost = (1 - probability) * scale_factor + time_penalty * time_gap
    """

    def __init__(
        self,
        model_path: str,
        scale_factor: float = 5.0,
        time_penalty: float = 0.1,
        use_logit: bool = False,
        logit_offset: float = 5.0,
    ):
        """
        Initialize logistic regression cost function.

        Args:
            model_path: Path to pickled model package (.pkl file)
                       Expected: {'model': fitted_lr, 'scaler': fitted_scaler, 'features': feature_list}
            scale_factor: Scale factor to adjust cost range (default 5.0)
            time_penalty: Weight for time gap penalty (default 0.1)
            use_logit: If True, use raw logit (decision_function) instead of probability.
                      Produces costs on Bhattacharyya scale (0-10+) so existing thresholds work.
            logit_offset: Offset added to negative logit for cost calculation (default 5.0).
                         cost = logit_offset - logit + time_penalty * gap
        """
        import pickle
        import numpy as np

        self.np = np
        self.scale_factor = scale_factor
        self.time_penalty = time_penalty
        self.use_logit = use_logit
        self.logit_offset = logit_offset

        # Resolve relative path
        model_path = self._resolve_path(model_path)

        # Load model package
        try:
            with open(model_path, "rb") as f:
                model_pkg = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

        # Extract components
        self.model = model_pkg.get("model")
        self.scaler = model_pkg.get("scaler")
        # Support both 'features' and 'feature_names' keys for compatibility
        self.selected_features = model_pkg.get("features") or model_pkg.get("feature_names", [])

        if self.model is None:
            raise ValueError("Model package missing 'model' key")
        if self.scaler is None:
            raise ValueError("Model package missing 'scaler' key")

        # Initialize feature extractor
        from utils.features_stitch import StitchFeatureExtractor
        self.feature_extractor = StitchFeatureExtractor(
            mode="selected", selected_features=self.selected_features
        )

        print(f"[LogisticRegressionCostFunction] Loaded model from {model_path}")
        print(f"[LogisticRegressionCostFunction] Features: {len(self.selected_features)}")
        if self.use_logit:
            print(f"[LogisticRegressionCostFunction] Mode: LOGIT (offset={logit_offset}), Time penalty: {time_penalty}")
        else:
            print(f"[LogisticRegressionCostFunction] Mode: probability, Scale factor: {scale_factor}, Time penalty: {time_penalty}")

    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths to absolute paths based on project root."""
        if path is None:
            return None
        if Path(path).is_absolute():
            return path
        project_root = Path(__file__).parent.parent
        resolved = project_root / path
        if resolved.exists():
            return str(resolved)
        return path

    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        """
        Compute cost using logistic regression model.

        Args:
            track1: First fragment dictionary
            track2: Second fragment dictionary
            TIME_WIN: Maximum time window for valid stitching
            param: Additional parameters

        Returns:
            Cost value (lower = better match). Returns 1e6 for invalid pairs.
        """
        np = self.np

        try:
            # Check time gap validity
            if "first_timestamp" in track2 and "last_timestamp" in track1:
                gap = track2["first_timestamp"] - track1["last_timestamp"]
            else:
                gap = track2["timestamp"][0] - track1["timestamp"][-1]

            if gap < 0 or gap > TIME_WIN:
                return 1e6  # Invalid pair

            # Extract features
            features = self.feature_extractor.extract_feature_vector(track1, track2)

            # Ensure correct shape for scaler/model
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Scale features
            features_scaled = self.scaler.transform(features)

            if self.use_logit:
                # Logit-based cost: maps to Bhattacharyya scale (0-10+)
                # logit > 0 means P(same) > 0.5, logit < 0 means P(same) < 0.5
                logit = self.model.decision_function(features_scaled)[0]
                base_cost = self.logit_offset - logit
            else:
                # Probability-based cost (original)
                probability = self.model.predict_proba(features_scaled)[0, 1]
                base_cost = (1.0 - probability) * self.scale_factor

            # Add time penalty for larger gaps
            time_cost = self.time_penalty * gap

            total_cost = base_cost + time_cost

            return total_cost

        except Exception as e:
            print(f"[LogisticRegressionCostFunction] Error computing cost: {e}")
            return 1e6  # Return high cost on error


class TorchLogisticCostFunction(StitchCostFunction):
    """
    Torch-based logistic regression scorer. Supports loading either a torch
    state dict (.pth) or converting from the existing sklearn pickle on the fly.
    """

    def __init__(
        self,
        model_path: str,
        scale_factor: float = 5.0,
        time_penalty: float = 0.1,
        device: str = "cpu",
        save_converted: bool = True,
        use_logit: bool = False,
        logit_offset: float = 5.0,
    ):
        import torch
        import pickle
        import numpy as np

        self.torch = torch
        self.np = np
        self.device = torch.device(device)
        self.scale_factor = scale_factor
        self.time_penalty = time_penalty
        self.use_logit = use_logit
        self.logit_offset = logit_offset

        resolved = self._resolve_path(model_path)
        path = Path(resolved)

        if path.suffix == ".pth" and path.exists():
            state = torch.load(path, map_location=self.device)
        else:
            # Convert from sklearn pickle
            with open(path, "rb") as f:
                model_pkg = pickle.load(f)
            lr = model_pkg["model"]
            scaler = model_pkg["scaler"]
            # Support both 'features' and 'feature_names' keys for compatibility
            features = model_pkg.get("features") or model_pkg.get("feature_names", [])

            state = {
                "weight": lr.coef_.astype("float32"),
                "bias": lr.intercept_.astype("float32"),
                "scaler_mean": scaler.mean_.astype("float32"),
                "scaler_scale": scaler.scale_.astype("float32"),
                "features": features,
            }

            # Optionally persist converted file next to original
            if save_converted:
                out_path = path.with_suffix(".pth")
                try:
                    torch.save(state, out_path)
                    print(f"[TorchLogisticCostFunction] Saved converted model to {out_path}")
                except Exception as e:
                    print(f"[TorchLogisticCostFunction] Warning: could not save converted model: {e}")

        # Build tensors
        self.weight = torch.as_tensor(state["weight"], device=self.device)  # shape (1, F)
        self.bias = torch.as_tensor(state["bias"].reshape(1), device=self.device)  # shape (1,)
        self.scaler_mean = torch.as_tensor(state["scaler_mean"], device=self.device)
        self.scaler_scale = torch.as_tensor(state["scaler_scale"], device=self.device)
        self.selected_features = state.get("features", [])

        from utils.features_stitch import StitchFeatureExtractor
        self.feature_extractor = StitchFeatureExtractor(
            mode="selected", selected_features=self.selected_features
        )

        print(f"[TorchLogisticCostFunction] Loaded model from {path}")
        print(f"[TorchLogisticCostFunction] Features: {len(self.selected_features)}")
        if self.use_logit:
            print(f"[TorchLogisticCostFunction] Mode: LOGIT (offset={logit_offset}), Time penalty: {time_penalty}")
        else:
            print(f"[TorchLogisticCostFunction] Mode: probability, Scale factor: {scale_factor}, Time penalty: {time_penalty}")
        print(f"[TorchLogisticCostFunction] Device: {self.device}")

    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths to absolute paths based on project root."""
        if path is None:
            return None
        if Path(path).is_absolute():
            return path
        project_root = Path(__file__).parent.parent
        resolved = project_root / path
        if resolved.exists():
            return str(resolved)
        return path

    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        torch = self.torch
        np = self.np

        try:
            if "first_timestamp" in track2 and "last_timestamp" in track1:
                gap = track2["first_timestamp"] - track1["last_timestamp"]
            else:
                gap = track2["timestamp"][0] - track1["timestamp"][-1]

            if gap < 0 or gap > TIME_WIN:
                return 1e6

            feats = self.feature_extractor.extract_feature_vector(track1, track2)
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)

            feats_t = torch.as_tensor(feats, device=self.device, dtype=torch.float32)
            feats_scaled = (feats_t - self.scaler_mean) / (self.scaler_scale + 1e-8)

            with torch.no_grad():
                logits = torch.matmul(feats_scaled, self.weight.T) + self.bias

                if self.use_logit:
                    base_cost = self.logit_offset - logits.item()
                else:
                    prob = torch.sigmoid(logits).item()
                    base_cost = (1.0 - prob) * self.scale_factor

            time_cost = self.time_penalty * gap
            return float(base_cost + time_cost)

        except Exception as e:
            print(f"[TorchLogisticCostFunction] Error computing cost: {e}")
            return 1e6


class MLPCostFunction(StitchCostFunction):
    """
    MLP-based cost function using raw summary features.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu',
                 scale_factor: float = 5.0, time_penalty: float = 0.1):
        import torch
        import numpy as np

        self.torch = torch
        self.np = np
        self.device = torch.device(device)
        self.scale_factor = scale_factor
        self.time_penalty = time_penalty

        # Add models directory to path
        models_dir = Path(__file__).parent.parent / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))

        from mlp_model import MLPStitchModel, extract_pair_features, TOTAL_INPUT_DIM
        self._extract_pair_features = extract_pair_features

        # Load checkpoint
        checkpoint_path = self._resolve_path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        input_dim = ckpt.get('input_dim', TOTAL_INPUT_DIM)

        self.model = MLPStitchModel(input_dim=input_dim)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Normalization stats
        self.feat_mean = torch.as_tensor(ckpt.get('feat_mean', np.zeros(input_dim)),
                                          device=self.device, dtype=torch.float32)
        self.feat_std = torch.as_tensor(ckpt.get('feat_std', np.ones(input_dim)),
                                         device=self.device, dtype=torch.float32)

        print(f"[MLPCostFunction] Loaded model from {checkpoint_path}")
        print(f"[MLPCostFunction] Scale factor: {scale_factor}, Time penalty: {time_penalty}")

    def _resolve_path(self, path: str) -> str:
        if path is None:
            return None
        if Path(path).is_absolute():
            return path
        project_root = Path(__file__).parent.parent
        resolved = project_root / path
        if resolved.exists():
            return str(resolved)
        return path

    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        np = self.np
        torch = self.torch

        try:
            if "first_timestamp" in track2 and "last_timestamp" in track1:
                gap = track2["first_timestamp"] - track1["last_timestamp"]
            else:
                gap = track2["timestamp"][0] - track1["timestamp"][-1]

            if gap < 0 or gap > TIME_WIN:
                return 1e6

            features = self._extract_pair_features(track1, track2)
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            feat_t = torch.as_tensor(features, device=self.device, dtype=torch.float32).unsqueeze(0)

            # Normalize
            feat_t = (feat_t - self.feat_mean) / (self.feat_std + 1e-8)

            with torch.no_grad():
                prob = self.model(feat_t).item()

            base_cost = (1.0 - prob) * self.scale_factor
            time_cost = self.time_penalty * gap
            return float(base_cost + time_cost)

        except Exception as e:
            print(f"[MLPCostFunction] Error: {e}")
            return 1e6


class TCNCostFunction(StitchCostFunction):
    """
    TCN (Temporal Convolutional Network) based cost function.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu',
                 scale_factor: float = 5.0, time_penalty: float = 0.1):
        import torch
        import numpy as np

        self.torch = torch
        self.np = np
        self.device = torch.device(device)
        self.scale_factor = scale_factor
        self.time_penalty = time_penalty

        models_dir = Path(__file__).parent.parent / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))

        from tcn_model import SiameseTCN
        from rich_sequence_dataset import extract_rich_sequence, extract_endpoint_features
        self._extract_sequence = extract_rich_sequence
        self._extract_endpoint = extract_endpoint_features

        checkpoint_path = self._resolve_path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model_config = ckpt.get('model_config', {})
        self.model = SiameseTCN(**model_config)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Normalization
        self.seq_mean = torch.as_tensor(ckpt.get('seq_mean', np.zeros(8)),
                                         device=self.device, dtype=torch.float32)
        self.seq_std = torch.as_tensor(ckpt.get('seq_std', np.ones(8)),
                                        device=self.device, dtype=torch.float32)

        print(f"[TCNCostFunction] Loaded model from {checkpoint_path}")

    def _resolve_path(self, path: str) -> str:
        if path is None:
            return None
        if Path(path).is_absolute():
            return path
        project_root = Path(__file__).parent.parent
        resolved = project_root / path
        if resolved.exists():
            return str(resolved)
        return path

    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        np = self.np
        torch = self.torch

        try:
            t1 = np.array(track1['timestamp'])
            t2 = np.array(track2['timestamp'])
            gap = t2[0] - t1[-1]

            if gap < 0 or gap > TIME_WIN:
                return 1e6

            seq_a = self._extract_sequence(track1)
            seq_b = self._extract_sequence(track2)
            endpoint = self._extract_endpoint(track1, track2)

            # Normalize
            seq_a_t = torch.FloatTensor(seq_a).unsqueeze(0).to(self.device)
            seq_b_t = torch.FloatTensor(seq_b).unsqueeze(0).to(self.device)
            seq_a_t = (seq_a_t - self.seq_mean) / (self.seq_std + 1e-8)
            seq_b_t = (seq_b_t - self.seq_mean) / (self.seq_std + 1e-8)

            endpoint_t = torch.FloatTensor(endpoint).unsqueeze(0).to(self.device)

            with torch.no_grad():
                similarity = self.model(seq_a_t, seq_b_t, endpoint_t).item()

            base_cost = (1.0 - similarity) * self.scale_factor
            time_cost = self.time_penalty * gap
            return float(base_cost + time_cost)

        except Exception as e:
            print(f"[TCNCostFunction] Error: {e}")
            return 1e6


class TransformerCostFunction(StitchCostFunction):
    """
    Transformer-based cost function for trajectory stitching.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu',
                 scale_factor: float = 5.0, time_penalty: float = 0.1):
        import torch
        import numpy as np

        self.torch = torch
        self.np = np
        self.device = torch.device(device)
        self.scale_factor = scale_factor
        self.time_penalty = time_penalty

        models_dir = Path(__file__).parent.parent / "models"
        if str(models_dir) not in sys.path:
            sys.path.insert(0, str(models_dir))

        from transformer_model import SiameseTransformerNetwork
        from rich_sequence_dataset import extract_rich_sequence, extract_endpoint_features
        self._extract_sequence = extract_rich_sequence
        self._extract_endpoint = extract_endpoint_features

        checkpoint_path = self._resolve_path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model_config = ckpt.get('model_config', {})
        self.model = SiameseTransformerNetwork(**model_config)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.seq_mean = torch.as_tensor(ckpt.get('seq_mean', np.zeros(8)),
                                         device=self.device, dtype=torch.float32)
        self.seq_std = torch.as_tensor(ckpt.get('seq_std', np.ones(8)),
                                        device=self.device, dtype=torch.float32)

        print(f"[TransformerCostFunction] Loaded model from {checkpoint_path}")

    def _resolve_path(self, path: str) -> str:
        if path is None:
            return None
        if Path(path).is_absolute():
            return path
        project_root = Path(__file__).parent.parent
        resolved = project_root / path
        if resolved.exists():
            return str(resolved)
        return path

    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        np = self.np
        torch = self.torch

        try:
            t1 = np.array(track1['timestamp'])
            t2 = np.array(track2['timestamp'])
            gap = t2[0] - t1[-1]

            if gap < 0 or gap > TIME_WIN:
                return 1e6

            seq_a = self._extract_sequence(track1)
            seq_b = self._extract_sequence(track2)
            endpoint = self._extract_endpoint(track1, track2)

            seq_a_t = torch.FloatTensor(seq_a).unsqueeze(0).to(self.device)
            seq_b_t = torch.FloatTensor(seq_b).unsqueeze(0).to(self.device)
            seq_a_t = (seq_a_t - self.seq_mean) / (self.seq_std + 1e-8)
            seq_b_t = (seq_b_t - self.seq_mean) / (self.seq_std + 1e-8)

            # Padding masks (all False = no padding for single sequences)
            mask_a = torch.zeros(1, seq_a_t.size(1), dtype=torch.bool, device=self.device)
            mask_b = torch.zeros(1, seq_b_t.size(1), dtype=torch.bool, device=self.device)
            endpoint_t = torch.FloatTensor(endpoint).unsqueeze(0).to(self.device)

            with torch.no_grad():
                similarity = self.model(seq_a_t, mask_a, seq_b_t, mask_b, endpoint_t).item()

            base_cost = (1.0 - similarity) * self.scale_factor
            time_cost = self.time_penalty * gap
            return float(base_cost + time_cost)

        except Exception as e:
            print(f"[TransformerCostFunction] Error: {e}")
            return 1e6


class CostFunctionFactory:
    """Factory for creating cost function instances"""

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve relative paths to absolute paths based on project root."""
        if path is None:
            return None
        if Path(path).is_absolute():
            return path
        project_root = Path(__file__).parent.parent
        resolved = project_root / path
        if resolved.exists():
            return str(resolved)
        return path

    @staticmethod
    def create(config: dict) -> StitchCostFunction:
        """
        Create a cost function based on configuration.

        Args:
            config: Configuration dictionary with keys:
                - type: 'bhattacharyya', 'siamese', or 'logistic_regression'
                - checkpoint_path: Path to model checkpoint (for siamese)
                - model_path: Path to model package (for logistic_regression)
                - device: Device for inference (for siamese, optional)
                - model_config: Model configuration (for siamese, optional)
                - scale_factor: Cost scaling (for logistic_regression, default 5.0)
                - time_penalty: Time penalty weight (for logistic_regression, default 0.1)

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

            checkpoint_path = CostFunctionFactory._resolve_path(checkpoint_path)
            device = config.get('device', None)
            model_config = config.get('model_config', None)
            scale_factor = config.get('scale_factor', 5.0)
            time_penalty = config.get('time_penalty', 0.1)

            return SiameseCostFunction(
                checkpoint_path=checkpoint_path,
                device=device,
                model_config=model_config,
                scale_factor=scale_factor,
                time_penalty=time_penalty
            )

        elif cost_type in ('logistic_regression', 'lr'):
            model_path = config.get('model_path')
            if model_path is None:
                raise ValueError("model_path required for logistic_regression cost function")

            model_path = CostFunctionFactory._resolve_path(model_path)
            scale_factor = config.get('scale_factor', 5.0)
            time_penalty = config.get('time_penalty', 0.1)
            use_logit = config.get('use_logit', False)
            logit_offset = config.get('logit_offset', 5.0)

            return LogisticRegressionCostFunction(
                model_path=model_path,
                scale_factor=scale_factor,
                time_penalty=time_penalty,
                use_logit=use_logit,
                logit_offset=logit_offset,
            )

        elif cost_type in ('torch_logistic', 'torch_lr', 'torch-logistic'):
            model_path = config.get('model_path')
            if model_path is None:
                raise ValueError("model_path required for torch_logistic cost function")

            model_path = CostFunctionFactory._resolve_path(model_path)
            scale_factor = config.get('scale_factor', 5.0)
            time_penalty = config.get('time_penalty', 0.1)
            device = config.get('device', 'cpu')
            save_converted = config.get('save_converted', True)
            use_logit = config.get('use_logit', False)
            logit_offset = config.get('logit_offset', 5.0)

            return TorchLogisticCostFunction(
                model_path=model_path,
                scale_factor=scale_factor,
                time_penalty=time_penalty,
                device=device,
                save_converted=save_converted,
                use_logit=use_logit,
                logit_offset=logit_offset,
            )

        elif cost_type == 'mlp':
            checkpoint_path = config.get('checkpoint_path')
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required for mlp cost function")
            checkpoint_path = CostFunctionFactory._resolve_path(checkpoint_path)
            return MLPCostFunction(
                checkpoint_path=checkpoint_path,
                device=config.get('device', 'cpu'),
                scale_factor=config.get('scale_factor', 5.0),
                time_penalty=config.get('time_penalty', 0.1),
            )

        elif cost_type == 'tcn':
            checkpoint_path = config.get('checkpoint_path')
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required for tcn cost function")
            checkpoint_path = CostFunctionFactory._resolve_path(checkpoint_path)
            return TCNCostFunction(
                checkpoint_path=checkpoint_path,
                device=config.get('device', 'cpu'),
                scale_factor=config.get('scale_factor', 5.0),
                time_penalty=config.get('time_penalty', 0.1),
            )

        elif cost_type == 'transformer':
            checkpoint_path = config.get('checkpoint_path')
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required for transformer cost function")
            checkpoint_path = CostFunctionFactory._resolve_path(checkpoint_path)
            return TransformerCostFunction(
                checkpoint_path=checkpoint_path,
                device=config.get('device', 'cpu'),
                scale_factor=config.get('scale_factor', 5.0),
                time_penalty=config.get('time_penalty', 0.1),
            )

        else:
            raise ValueError(
                f"Unknown cost function type: {cost_type}. "
                f"Supported: 'bhattacharyya', 'siamese', 'logistic_regression', "
                f"'torch_logistic', 'mlp', 'tcn', 'transformer'"
            )


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
