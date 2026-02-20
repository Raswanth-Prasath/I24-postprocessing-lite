"""
Stitch Cost Function Interface

Abstraction layer for fragment stitching cost functions using Strategy pattern.
Allows switching between Bhattacharyya distance and learned Siamese network.
"""

from abc import ABC, abstractmethod
import atexit
import json
import math
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

    def __init__(self, checkpoint_path: str, device: str = None,
                 scale_factor: float = 5.0, time_penalty: float = 0.1,
                 calibration_mode: str = None,
                 calibration_path: str = None,
                 similarity_mapping: str = "linear",
                 similarity_power: float = 1.0,
                 similarity_clip_eps: float = 1e-2,
                 training_objective: str = "classification",
                 score_mapping: str = None):
        import torch
        import numpy as np

        self.torch = torch
        self.np = np
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.scale_factor = scale_factor
        self.time_penalty = time_penalty
        self.invalid_cost_count = 0
        self.calibration_applied_count = 0

        self.training_objective = str(training_objective or "classification").lower()
        if self.training_objective not in {"classification", "ranking"}:
            print(
                f"[TransformerCostFunction] Warning: unknown training_objective='{self.training_objective}'. "
                "Falling back to 'classification'."
            )
            self.training_objective = "classification"

        if score_mapping is None:
            score_mapping = "direct_cost" if self.training_objective == "ranking" else "legacy_similarity"
        self.score_mapping = str(score_mapping).lower()
        if self.score_mapping not in {"legacy_similarity", "direct_cost"}:
            print(
                f"[TransformerCostFunction] Warning: unknown score_mapping='{self.score_mapping}'. "
                "Falling back to objective-appropriate default."
            )
            self.score_mapping = "direct_cost" if self.training_objective == "ranking" else "legacy_similarity"
        self.use_explicit_time_penalty = not (
            self.training_objective == "ranking" and self.score_mapping == "direct_cost"
        )
        if not self.use_explicit_time_penalty:
            print(
                "[TransformerCostFunction] Info: skipping explicit time penalty for "
                "ranking+direct_cost to avoid double-counting gap cost."
            )

        default_calibration_mode = "off" if self.training_objective == "ranking" else "isotonic"
        self.calibration_mode = (calibration_mode or default_calibration_mode).lower()
        self.calibration_path = calibration_path
        self.calibration = None
        self.similarity_mapping = str(similarity_mapping or "linear").lower()
        if self.similarity_mapping not in {"linear", "odds", "power"}:
            print(
                f"[TransformerCostFunction] Warning: unknown similarity_mapping='{self.similarity_mapping}'. "
                "Falling back to 'linear'."
            )
            self.similarity_mapping = "linear"
        self.similarity_power = float(similarity_power)
        if self.similarity_power <= 0:
            print(
                f"[TransformerCostFunction] Warning: similarity_power={self.similarity_power} must be > 0. "
                "Using 1.0."
            )
            self.similarity_power = 1.0
        self.similarity_clip_eps = float(similarity_clip_eps)
        if self.similarity_clip_eps <= 0 or self.similarity_clip_eps >= 0.5:
            print(
                f"[TransformerCostFunction] Warning: similarity_clip_eps={self.similarity_clip_eps} out of range. "
                "Using 1e-2."
            )
            self.similarity_clip_eps = 1e-2

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
        checkpoint_objective = str(model_config.get("training_objective", "classification")).lower()
        if checkpoint_objective not in {"classification", "ranking"}:
            checkpoint_objective = "classification"
        if checkpoint_objective != self.training_objective:
            print(
                f"[TransformerCostFunction] Warning: runtime training_objective={self.training_objective} "
                f"differs from checkpoint training_objective={checkpoint_objective}; using runtime value."
            )
        model_config["training_objective"] = self.training_objective
        self.model = SiameseTransformerNetwork(**model_config)
        load_result = self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        legacy_time_bias_keys = {
            "encoder.time_bias.proj.0.weight",
            "encoder.time_bias.proj.0.bias",
            "encoder.time_bias.proj.2.weight",
            "encoder.time_bias.proj.2.bias",
        }
        missing_set = set(load_result.missing_keys)
        unexpected_set = set(load_result.unexpected_keys)
        if missing_set and missing_set.issubset(legacy_time_bias_keys) and not unexpected_set:
            print(
                "[TransformerCostFunction] Info: legacy checkpoint detected (no time_bias weights); "
                "using safe zero-initialized time bias."
            )
        else:
            if load_result.missing_keys:
                print(f"[TransformerCostFunction] Warning: missing checkpoint keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"[TransformerCostFunction] Warning: unexpected checkpoint keys: {load_result.unexpected_keys}")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.seq_mean = torch.as_tensor(ckpt.get('seq_mean', np.zeros(8)),
                                         device=self.device, dtype=torch.float32)
        self.seq_std = torch.as_tensor(ckpt.get('seq_std', np.ones(8)),
                                        device=self.device, dtype=torch.float32).clamp_min(1e-6)
        self.ep_mean = torch.as_tensor(ckpt.get('ep_mean', np.zeros(4)),
                                       device=self.device, dtype=torch.float32)
        self.ep_std = torch.as_tensor(ckpt.get('ep_std', np.ones(4)),
                                      device=self.device, dtype=torch.float32).clamp_min(1e-6)

        print(f"[TransformerCostFunction] Loaded model from {checkpoint_path}")
        self._initialize_calibration()

    def _similarity_to_raw_base_cost(self, similarity: float) -> float:
        """
        Convert model similarity to an uncalibrated base cost before time penalty.
        Mapping is monotonic so ordering is preserved while dynamic range can be widened.
        """
        sim = float(self.np.clip(similarity, self.similarity_clip_eps, 1.0 - self.similarity_clip_eps))
        if self.similarity_mapping == "odds":
            # Emphasize mid/low similarities: d = (1-s)/s
            dissimilarity = (1.0 - sim) / sim
        elif self.similarity_mapping == "power":
            dissimilarity = (1.0 - sim) ** self.similarity_power
        else:
            dissimilarity = (1.0 - sim)
        return float(dissimilarity * self.scale_factor)

    def _score_to_raw_base_cost(self, score: float) -> float:
        """
        Convert model output to an uncalibrated non-negative base cost.

        - legacy_similarity: model output is a similarity in [0, 1]
        - direct_cost: model output is an unconstrained scalar and is mapped via softplus
        """
        if self.score_mapping == "direct_cost":
            x = float(score)
            softplus = self.np.log1p(self.np.exp(-abs(x))) + max(x, 0.0)
            return float(softplus)
        return self._similarity_to_raw_base_cost(score)

    def _to_raw_total_cost(self, raw_base_cost: float, gap: float) -> float:
        """
        Convert model base score to raw total cost used by stitching.

        Ranking+direct_cost outputs are trained against Bhattacharyya total cost,
        so adding an extra explicit gap penalty would double-count time.
        """
        if self.use_explicit_time_penalty:
            return float(raw_base_cost + self.time_penalty * gap)
        return float(raw_base_cost)

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

    def _default_calibration_path(self) -> str:
        project_root = Path(__file__).parent.parent
        return str(project_root / "models" / "outputs" / "transformer_calibration.json")

    def _initialize_calibration(self):
        if self.calibration_mode in {"linear", "off"}:
            return

        path = self.calibration_path or self._default_calibration_path()
        resolved_path = self._resolve_path(path)
        calibration = self._load_calibration_artifact(resolved_path)
        if calibration is None:
            print(
                f"[TransformerCostFunction] Calibration mode '{self.calibration_mode}' requested but "
                f"artifact unavailable at {resolved_path}. Falling back to linear mapping."
            )
            self.calibration_mode = "linear"
            return

        self.calibration = calibration
        print(
            f"[TransformerCostFunction] Loaded calibration mode='{self.calibration_mode}' "
            f"from {resolved_path}"
        )

        fit_params = calibration.get("fit_params", {})
        fit_scale = fit_params.get("scale_factor", None)
        fit_time_penalty = fit_params.get("time_penalty", None)
        if fit_scale is not None and abs(float(fit_scale) - float(self.scale_factor)) > 1e-6:
            print(
                f"[TransformerCostFunction] Warning: checkpoint runtime scale_factor={self.scale_factor} "
                f"differs from calibration fit scale_factor={fit_scale}."
            )
        if fit_time_penalty is not None and abs(float(fit_time_penalty) - float(self.time_penalty)) > 1e-6:
            print(
                f"[TransformerCostFunction] Warning: runtime time_penalty={self.time_penalty} "
                f"differs from calibration fit time_penalty={fit_time_penalty}."
            )
        fit_mapping = fit_params.get("similarity_mapping", None)
        fit_power = fit_params.get("similarity_power", None)
        if fit_mapping is not None and str(fit_mapping).lower() != self.similarity_mapping:
            print(
                f"[TransformerCostFunction] Warning: runtime similarity_mapping={self.similarity_mapping} "
                f"differs from calibration fit similarity_mapping={fit_mapping}."
            )
        if fit_power is not None and abs(float(fit_power) - float(self.similarity_power)) > 1e-9:
            print(
                f"[TransformerCostFunction] Warning: runtime similarity_power={self.similarity_power} "
                f"differs from calibration fit similarity_power={fit_power}."
            )

    def _load_calibration_artifact(self, path: str):
        try:
            if path is None or not Path(path).exists():
                return None

            with open(path, "r") as f:
                artifact = json.load(f)

            x_knots = self.np.asarray(artifact.get("x_knots", []), dtype=self.np.float64)
            y_knots = self.np.asarray(artifact.get("y_knots", []), dtype=self.np.float64)
            if len(x_knots) < 2 or len(y_knots) != len(x_knots):
                raise ValueError("Calibration artifact must contain matching x_knots/y_knots with >=2 points.")

            order = self.np.argsort(x_knots)
            x_knots = x_knots[order]
            y_knots = y_knots[order]

            x_knots, unique_idx = self.np.unique(x_knots, return_index=True)
            y_knots = y_knots[unique_idx]
            if len(x_knots) < 2:
                raise ValueError("Calibration x_knots are not sufficiently distinct.")

            domain = artifact.get("domain", [float(x_knots[0]), float(x_knots[-1])])
            if not isinstance(domain, (list, tuple)) or len(domain) != 2:
                domain = [float(x_knots[0]), float(x_knots[-1])]
            domain = [float(domain[0]), float(domain[1])]
            if domain[0] >= domain[1]:
                domain = [float(x_knots[0]), float(x_knots[-1])]

            fit_params = artifact.get("fit_params", {})
            target_mapping = fit_params.get("target_mapping", "")
            if target_mapping and target_mapping != "raw_total_to_bhat_total":
                print(
                    f"[TransformerCostFunction] Warning: calibration artifact uses "
                    f"target_mapping='{target_mapping}', expected 'raw_total_to_bhat_total'."
                )

            fit_metrics = artifact.get("fit_metrics", {})
            spearman = fit_metrics.get("spearman", None)
            try:
                spearman_value = float(spearman) if spearman is not None else None
            except (TypeError, ValueError):
                spearman_value = None
            if (
                spearman_value is not None
                and self.np.isfinite(spearman_value)
                and spearman_value < 0.5
            ):
                print(
                    f"[TransformerCostFunction] Warning: weak calibration fit "
                    f"(spearman={spearman_value:.3f})."
                )

            artifact["x_knots"] = x_knots
            artifact["y_knots"] = y_knots
            artifact["domain"] = domain
            return artifact
        except Exception as exc:
            print(f"[TransformerCostFunction] Failed to load calibration artifact '{path}': {exc}")
            return None

    def _apply_calibration(self, raw_cost: float) -> float:
        if self.calibration_mode in {"linear", "off"} or self.calibration is None:
            return float(raw_cost)

        x_knots = self.calibration["x_knots"]
        y_knots = self.calibration["y_knots"]
        domain_min, domain_max = self.calibration["domain"]
        clipped = float(self.np.clip(raw_cost, domain_min, domain_max))
        calibrated = self.np.interp(clipped, x_knots, y_knots)
        return float(calibrated)

    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        np = self.np
        torch = self.torch

        try:
            t1 = np.array(track1['timestamp'])
            t2 = np.array(track2['timestamp'])
            gap = t2[0] - t1[-1]

            if gap < 0 or gap > TIME_WIN:
                self.invalid_cost_count += 1
                return 1e6

            seq_a = self._extract_sequence(track1)
            seq_b = self._extract_sequence(track2)
            endpoint = self._extract_endpoint(track1, track2)

            seq_a_t = torch.FloatTensor(seq_a).unsqueeze(0).to(self.device)
            seq_b_t = torch.FloatTensor(seq_b).unsqueeze(0).to(self.device)
            seq_a_t = (seq_a_t - self.seq_mean) / self.seq_std
            seq_b_t = (seq_b_t - self.seq_mean) / self.seq_std

            # Padding masks (all False = no padding for single sequences)
            mask_a = torch.zeros(1, seq_a_t.size(1), dtype=torch.bool, device=self.device)
            mask_b = torch.zeros(1, seq_b_t.size(1), dtype=torch.bool, device=self.device)
            endpoint_t = torch.FloatTensor(endpoint).unsqueeze(0).to(self.device)
            endpoint_t = (endpoint_t - self.ep_mean) / self.ep_std

            with torch.no_grad():
                model_output = self.model(seq_a_t, mask_a, seq_b_t, mask_b, endpoint_t).item()

            raw_base_cost = self._score_to_raw_base_cost(model_output)
            raw_total_cost = self._to_raw_total_cost(raw_base_cost, gap)
            if not self.np.isfinite(raw_total_cost):
                self.invalid_cost_count += 1
                return 1e6
            calibrated_total = self._apply_calibration(raw_total_cost)
            if self.calibration_mode not in {"linear", "off"} and self.calibration is not None:
                self.calibration_applied_count += 1
            if not self.np.isfinite(calibrated_total):
                self.invalid_cost_count += 1
                return 1e6
            return float(calibrated_total)

        except Exception as e:
            print(f"[TransformerCostFunction] Error: {e}")
            self.invalid_cost_count += 1
            return 1e6


class GateRankCostFunction(StitchCostFunction):
    """Two-stage cost: physics gate first, learned ranker second."""

    def __init__(self, gate_fn: StitchCostFunction, rank_fn: StitchCostFunction,
                 gate_thresh: float = None, stats_log_on_exit: bool = True):
        self.gate_fn = gate_fn
        self.rank_fn = rank_fn
        self._gate_thresh_override = float(gate_thresh) if gate_thresh is not None else None
        self._stats_log_on_exit = bool(stats_log_on_exit)

        # Basic observability counters for per-process runs.
        self.gate_evaluations = 0
        self.gate_rejections = 0
        self.rank_evaluations = 0
        self.rank_fallbacks = 0
        self.gate_failures = 0

        if self._stats_log_on_exit:
            atexit.register(self.log_stats)

    def _resolve_gate_thresh(self, param: dict) -> float:
        if self._gate_thresh_override is not None:
            return self._gate_thresh_override
        return float(param["stitch_thresh"])

    def get_stats(self) -> dict:
        accepted = self.gate_evaluations - self.gate_rejections
        acceptance_rate = accepted / self.gate_evaluations if self.gate_evaluations else 0.0
        return {
            "gate_evaluations": self.gate_evaluations,
            "gate_rejections": self.gate_rejections,
            "gate_acceptances": accepted,
            "acceptance_rate": acceptance_rate,
            "rank_evaluations": self.rank_evaluations,
            "rank_fallbacks": self.rank_fallbacks,
            "gate_failures": self.gate_failures,
        }

    def log_stats(self):
        stats = self.get_stats()
        print(f"[GateRankCostFunction] Stats: {json.dumps(stats, sort_keys=True)}")

    def compute_cost(self, track1: dict, track2: dict,
                    TIME_WIN: float, param: dict) -> float:
        self.gate_evaluations += 1
        try:
            gate_cost = float(self.gate_fn.compute_cost(track1, track2, TIME_WIN, param))
        except Exception as exc:
            self.gate_failures += 1
            print(f"[GateRankCostFunction] Gate error: {exc}")
            return 1e6

        if not math.isfinite(gate_cost):
            self.gate_failures += 1
            return 1e6

        try:
            effective_gate_thresh = self._resolve_gate_thresh(param)
        except Exception as exc:
            self.gate_failures += 1
            print(f"[GateRankCostFunction] Invalid gate threshold: {exc}")
            return 1e6

        if gate_cost > effective_gate_thresh:
            self.gate_rejections += 1
            return gate_cost

        self.rank_evaluations += 1
        try:
            rank_cost = float(self.rank_fn.compute_cost(track1, track2, TIME_WIN, param))
        except Exception as exc:
            self.rank_fallbacks += 1
            print(f"[GateRankCostFunction] Rank error, fallback to gate cost: {exc}")
            return gate_cost

        if not math.isfinite(rank_cost):
            self.rank_fallbacks += 1
            return gate_cost

        return rank_cost


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
            training_objective = str(config.get('training_objective', 'classification')).lower()
            default_score_mapping = 'direct_cost' if training_objective == 'ranking' else 'legacy_similarity'
            default_calibration_mode = 'off' if training_objective == 'ranking' else 'isotonic'
            return TransformerCostFunction(
                checkpoint_path=checkpoint_path,
                device=config.get('device', None),
                scale_factor=config.get('scale_factor', 5.0),
                time_penalty=config.get('time_penalty', 0.1),
                calibration_mode=config.get('calibration_mode', default_calibration_mode),
                calibration_path=config.get('calibration_path', None),
                similarity_mapping=config.get('similarity_mapping', 'linear'),
                similarity_power=config.get('similarity_power', 1.0),
                similarity_clip_eps=config.get('similarity_clip_eps', 1e-2),
                training_objective=training_objective,
                score_mapping=config.get('score_mapping', default_score_mapping),
            )

        elif cost_type in ('gate_rank', 'hybrid'):
            gate_config = config.get('gate')
            rank_config = config.get('rank')
            if gate_config is None or rank_config is None:
                raise ValueError("gate and rank sub-configs are required for gate_rank cost function")

            gate_type = str(gate_config.get('type', '')).lower()
            rank_type = str(rank_config.get('type', '')).lower()
            if gate_type in ('gate_rank', 'hybrid') or rank_type in ('gate_rank', 'hybrid'):
                raise ValueError("Nested gate_rank/hybrid cost function is not supported")

            gate_fn = CostFunctionFactory.create(gate_config)
            rank_fn = CostFunctionFactory.create(rank_config)
            return GateRankCostFunction(
                gate_fn=gate_fn,
                rank_fn=rank_fn,
                gate_thresh=config.get('gate_thresh', None),
                stats_log_on_exit=config.get('stats_log_on_exit', True),
            )

        else:
            raise ValueError(
                f"Unknown cost function type: {cost_type}. "
                f"Supported: 'bhattacharyya', 'siamese', 'logistic_regression', "
                f"'torch_logistic', 'mlp', 'tcn', 'transformer', 'gate_rank'/'hybrid'"
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
