"""
MLP Model for Trajectory Fragment Stitching

Uses fixed-size summary statistics from raw fragment data (no hand-crafted domain features).

Per-fragment: 8 raw features x 6 stats (mean, std, min, max, first, last) = 48
Pair-level: 4 gap features (time_gap, x_gap, y_gap, velocity_diff)
Total input: 48 + 48 + 4 = 100 raw summary features
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict


NUM_RAW_FEATURES = 8
NUM_STATS = 6  # mean, std, min, max, first, last
SUMMARY_DIM = NUM_RAW_FEATURES * NUM_STATS  # 48 per fragment
PAIR_GAP_DIM = 4
TOTAL_INPUT_DIM = SUMMARY_DIM * 2 + PAIR_GAP_DIM  # 100


def extract_fragment_summary(fragment: Dict) -> np.ndarray:
    """
    Aggregate raw features of a variable-length fragment into fixed-size summary.

    For each of 8 raw features, compute 6 statistics:
        mean, std, min, max, first, last

    Returns:
        Array of shape (48,)
    """
    timestamps = np.array(fragment['timestamp'], dtype=np.float64)
    seq_len = len(timestamps)
    t_norm = (timestamps - timestamps[0]).astype(np.float32)

    x_pos = np.array(fragment['x_position'], dtype=np.float32)
    y_pos = np.array(fragment['y_position'], dtype=np.float32)

    # Velocity
    if 'velocity' in fragment and fragment['velocity'] is not None:
        vel = np.array(fragment['velocity'], dtype=np.float32)
        if len(vel) != seq_len:
            vel = _compute_velocity(timestamps, x_pos)
    else:
        vel = _compute_velocity(timestamps, x_pos)

    # Scalar or array fields
    length = _to_array(fragment.get('length', 15.0), seq_len)
    width = _to_array(fragment.get('width', 6.0), seq_len)
    height = _to_array(fragment.get('height', 5.0), seq_len)

    if 'detection_confidence' in fragment and fragment['detection_confidence'] is not None:
        conf = np.array(fragment['detection_confidence'], dtype=np.float32)
        if len(conf) != seq_len:
            conf = np.ones(seq_len, dtype=np.float32)
    else:
        conf = np.ones(seq_len, dtype=np.float32)

    features = [x_pos, y_pos, vel, length, width, height, conf, t_norm]

    stats = []
    for feat in features:
        stats.extend([
            np.mean(feat),
            np.std(feat) if len(feat) > 1 else 0.0,
            np.min(feat),
            np.max(feat),
            feat[0],
            feat[-1],
        ])
    return np.array(stats, dtype=np.float32)


def extract_pair_features(frag_a: Dict, frag_b: Dict) -> np.ndarray:
    """
    Extract full MLP input vector for a fragment pair.

    Returns:
        Array of shape (100,) = 48 (frag_a) + 48 (frag_b) + 4 (gap)
    """
    summary_a = extract_fragment_summary(frag_a)
    summary_b = extract_fragment_summary(frag_b)
    gap = _extract_gap_features(frag_a, frag_b)
    return np.concatenate([summary_a, summary_b, gap])


def _extract_gap_features(frag_a: Dict, frag_b: Dict) -> np.ndarray:
    """Extract 4 gap features between fragment endpoints."""
    t_a_end = frag_a.get('last_timestamp', frag_a['timestamp'][-1])
    t_b_start = frag_b.get('first_timestamp', frag_b['timestamp'][0])
    time_gap = t_b_start - t_a_end

    x_a_end = frag_a.get('ending_x', frag_a['x_position'][-1])
    x_b_start = frag_b.get('starting_x', frag_b['x_position'][0])
    x_gap = x_b_start - x_a_end

    y_gap = frag_b['y_position'][0] - frag_a['y_position'][-1]

    v_a = _get_endpoint_velocity(frag_a, end=True)
    v_b = _get_endpoint_velocity(frag_b, end=False)
    velocity_diff = v_b - v_a

    return np.array([time_gap, x_gap, y_gap, velocity_diff], dtype=np.float32)


def _compute_velocity(timestamps, x_pos):
    dt = np.diff(timestamps)
    dx = np.diff(x_pos)
    velocity = np.zeros(len(timestamps), dtype=np.float32)
    velocity[1:] = dx / (dt + 1e-6)
    velocity[0] = velocity[1] if len(velocity) > 1 else 0
    return velocity


def _to_array(value, length):
    if isinstance(value, (list, np.ndarray)):
        arr = np.array(value, dtype=np.float32)
        if len(arr) == length:
            return arr
        return np.full(length, np.mean(arr), dtype=np.float32)
    return np.full(length, float(value), dtype=np.float32)


def _get_endpoint_velocity(frag, end=True):
    if 'velocity' in frag and frag['velocity'] is not None and len(frag['velocity']) > 0:
        return frag['velocity'][-1] if end else frag['velocity'][0]
    ts = frag['timestamp']
    xs = frag['x_position']
    if len(ts) >= 2:
        if end:
            return (xs[-1] - xs[-2]) / (ts[-1] - ts[-2] + 1e-6)
        else:
            return (xs[1] - xs[0]) / (ts[1] - ts[0] + 1e-6)
    return 0.0


class MLPStitchModel(nn.Module):
    """
    MLP for fragment pair classification.

    Input: 100 raw summary features
    Output: probability that fragments belong to the same vehicle
    """

    def __init__(self, input_dim: int = TOTAL_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 100) raw summary features
        Returns:
            (batch, 1) probability
        """
        return self.net(x)


if __name__ == "__main__":
    model = MLPStitchModel()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MLP parameters: {n_params:,}")
    print(model)

    x = torch.randn(8, TOTAL_INPUT_DIM)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Output values: {out.squeeze().detach().numpy()}")
