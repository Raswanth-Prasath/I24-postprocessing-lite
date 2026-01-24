"""
Siamese BiLSTM Trajectory Reconstruction Visualization

Generates time-space diagrams comparing:
1. Raw Fragments (before reconstruction)
2. Ground Truth (complete trajectories)
3. Siamese Network Predictions (reconstructed trajectories)

Author: Raswanth Prasath
"""

import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.cm import get_cmap
from pathlib import Path
import torch
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from siamese_model import SiameseTrajectoryNetwork
from siamese_dataset import TrajectoryPairDataset

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_accuracy.pth"

DATASET_DIR = Path(r"D:\ASU Academics\Thesis & Research\01_Papers\Datasets\I24-3D")

# ============================================================================
# SIAMESE NETWORK INFERENCE
# ============================================================================

class SiameseReconstructor:
    """Uses trained Siamese network to reconstruct trajectories from fragments."""

    def __init__(self, model_path: Path, device: str = None):
        """
        Initialize the reconstructor with a trained model.

        Args:
            model_path: Path to the trained model checkpoint
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model configuration (must match training)
        self.model_config = {
            'input_size': 4,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'similarity_hidden_dim': 64
        }

        # Load model
        self.model = SiameseTrajectoryNetwork(**self.model_config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded Siamese model from {model_path}")
        print(f"Using device: {self.device}")
        if 'epoch' in checkpoint:
            print(f"Model from epoch: {checkpoint['epoch']}")

    def _extract_sequence(self, fragment: Dict) -> np.ndarray:
        """
        Extract trajectory sequence from fragment dictionary.

        Args:
            fragment: Fragment dictionary with timestamp, x_position, y_position, velocity

        Returns:
            Array of shape (seq_len, 4) with [x, y, velocity, time_normalized]
        """
        timestamps = np.array(fragment['timestamp'])
        x_pos = np.array(fragment['x_position'])
        y_pos = np.array(fragment['y_position'])

        # Get or compute velocity
        if 'velocity' in fragment and len(fragment['velocity']) > 0:
            velocity = np.array(fragment['velocity'])
            # Ensure same length
            min_len = min(len(timestamps), len(velocity))
            timestamps = timestamps[:min_len]
            x_pos = x_pos[:min_len]
            y_pos = y_pos[:min_len]
            velocity = velocity[:min_len]
        else:
            # Compute velocity from positions
            dt = np.diff(timestamps)
            dx = np.diff(x_pos)
            velocity = np.zeros(len(timestamps))
            velocity[1:] = dx / (dt + 1e-6)
            velocity[0] = velocity[1] if len(velocity) > 1 else 0

        # Normalize timestamps to start at 0
        t_norm = timestamps - timestamps[0]

        # Build sequence: [x, y, velocity, time]
        sequence = np.column_stack([x_pos, y_pos, velocity, t_norm])

        return sequence.astype(np.float32)

    @torch.no_grad()
    def compute_similarity(self, frag_a: Dict, frag_b: Dict) -> float:
        """
        Compute similarity score between two fragments.

        Args:
            frag_a: First fragment
            frag_b: Second fragment

        Returns:
            Similarity score in [0, 1] (higher = more likely same vehicle)
        """
        # Extract sequences
        seq_a = self._extract_sequence(frag_a)
        seq_b = self._extract_sequence(frag_b)

        # Convert to tensors and add batch dimension
        seq_a = torch.FloatTensor(seq_a).unsqueeze(0).to(self.device)
        seq_b = torch.FloatTensor(seq_b).unsqueeze(0).to(self.device)
        len_a = torch.LongTensor([seq_a.size(1)]).to(self.device)
        len_b = torch.LongTensor([seq_b.size(1)]).to(self.device)

        # Forward pass
        similarity, _, _ = self.model(seq_a, len_a, seq_b, len_b)

        return similarity.item()

    def reconstruct_trajectories(self, fragments: List[Dict],
                                  threshold: float = 0.5,
                                  max_time_gap: float = 5.0,
                                  max_spatial_gap: float = 200.0,
                                  max_y_diff: float = 8.0) -> List[List[Dict]]:
        """
        Reconstruct vehicle trajectories by linking fragments.

        Uses greedy matching: for each unassigned fragment, find the best
        continuation based on Siamese similarity scores.

        Args:
            fragments: List of fragment dictionaries
            threshold: Minimum similarity score to link fragments
            max_time_gap: Maximum time gap between fragments (seconds)
            max_spatial_gap: Maximum spatial gap (feet)
            max_y_diff: Maximum lateral difference (feet)

        Returns:
            List of trajectories, where each trajectory is a list of fragments
        """
        print(f"\nReconstructing trajectories with Siamese Network...")
        print(f"  Threshold: {threshold}")
        print(f"  Max time gap: {max_time_gap}s")
        print(f"  Max spatial gap: {max_spatial_gap}ft")

        # Sort fragments by start time
        sorted_fragments = sorted(fragments, key=lambda x: x['first_timestamp'])
        n = len(sorted_fragments)

        trajectories = []
        used_fragments = set()

        # Track progress
        total_pairs_evaluated = 0

        for i in range(n):
            if i in used_fragments:
                continue

            # Start new trajectory with this fragment
            current_trajectory = [sorted_fragments[i]]
            current_frag = sorted_fragments[i]
            used_fragments.add(i)

            # Try to extend trajectory
            while True:
                best_match = None
                best_similarity = threshold

                # Look for candidate continuations
                for j in range(i + 1, n):
                    if j in used_fragments:
                        continue

                    frag_b = sorted_fragments[j]

                    # Skip if fragments overlap in time
                    if current_frag['last_timestamp'] >= frag_b['first_timestamp']:
                        continue

                    # Skip if different directions
                    if current_frag['direction'] != frag_b['direction']:
                        continue

                    # Time gap constraint
                    time_gap = frag_b['first_timestamp'] - current_frag['last_timestamp']
                    if time_gap > max_time_gap:
                        continue

                    # Spatial gap constraint
                    if current_frag['direction'] == 1:  # Eastbound
                        spatial_gap = frag_b['starting_x'] - current_frag['ending_x']
                    else:  # Westbound
                        spatial_gap = current_frag['starting_x'] - frag_b['ending_x']

                    if spatial_gap < -50 or spatial_gap > max_spatial_gap:
                        continue

                    # Lateral constraint
                    y_diff = abs(np.mean(frag_b['y_position']) - np.mean(current_frag['y_position']))
                    if y_diff > max_y_diff:
                        continue

                    # Compute Siamese similarity
                    try:
                        similarity = self.compute_similarity(current_frag, frag_b)
                        total_pairs_evaluated += 1
                    except Exception as e:
                        print(f"    Warning: Error computing similarity: {e}")
                        continue

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (j, frag_b, similarity)

                # Link best match if found
                if best_match is not None:
                    idx, matched_frag, sim = best_match
                    current_trajectory.append(matched_frag)
                    used_fragments.add(idx)
                    current_frag = matched_frag
                else:
                    break

            trajectories.append(current_trajectory)

            # Progress update
            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{n} fragments...")

        # Statistics
        multi_frag = sum(1 for t in trajectories if len(t) > 1)
        avg_len = np.mean([len(t) for t in trajectories])

        print(f"\n  Reconstruction complete:")
        print(f"    Input fragments: {n}")
        print(f"    Output trajectories: {len(trajectories)}")
        print(f"    Multi-fragment trajectories: {multi_frag}")
        print(f"    Average trajectory length: {avg_len:.2f} fragments")
        print(f"    Pairs evaluated: {total_pairs_evaluated}")

        return trajectories


# ============================================================================
# DATA LOADING
# ============================================================================

def load_scenario_data(scenario: str) -> Tuple[List[Dict], List[Dict]]:
    """Load ground truth and raw fragment data for a scenario."""
    gt_path = DATASET_DIR / f"GT_{scenario}.json"
    raw_path = DATASET_DIR / f"RAW_{scenario}.json"

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)

    print(f"\nLoaded scenario {scenario}:")
    print(f"  Ground truth vehicles: {len(gt_data)}")
    print(f"  Raw fragments: {len(raw_data)}")

    return gt_data, raw_data


def calculate_lane_bounds(gt_data: List[Dict], raw_data: List[Dict]):
    """Calculate lane boundaries from trajectory data."""
    eb_y = []
    wb_y = []
    all_t = []

    for data in [gt_data, raw_data]:
        for item in data:
            if item['direction'] == 1:
                eb_y.extend(item['y_position'])
            elif item['direction'] == -1:
                wb_y.extend(item['y_position'])
            all_t.extend(item['timestamp'])

    # Lane bounds (12 ft per lane)
    min_eb, max_eb = min(eb_y) if eb_y else 0, max(eb_y) if eb_y else 48
    min_wb, max_wb = min(wb_y) if wb_y else -48, max(wb_y) if wb_y else 0

    num_eb = max(1, int(np.ceil((max_eb - min_eb) / 12)))
    num_wb = max(1, int(np.ceil((max_wb - min_wb) / 12)))

    eb_bounds = [min_eb + i * 12 for i in range(num_eb + 1)]
    wb_bounds = [max_wb - i * 12 for i in range(num_wb + 1)][::-1]

    min_timestamp = min(all_t) if all_t else 0

    return eb_bounds, wb_bounds, num_eb, num_wb, min_timestamp


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def assign_lane_to_trajectory(trajectory_fragments, eb_lane_bounds, wb_lane_bounds):
    """Assign a lane to a trajectory based on average y-position."""
    all_y = []
    for frag in trajectory_fragments:
        all_y.extend(frag['y_position'])

    y_avg = np.mean(all_y)
    direction = trajectory_fragments[0]['direction']

    if direction == 1:
        for i in range(len(eb_lane_bounds) - 1):
            if eb_lane_bounds[i] <= y_avg < eb_lane_bounds[i + 1]:
                return i + 1, 'EB'
        return len(eb_lane_bounds) - 1, 'EB'
    elif direction == -1:
        for i in range(len(wb_lane_bounds) - 1):
            if wb_lane_bounds[i] <= y_avg < wb_lane_bounds[i + 1]:
                return i + 1, 'WB'
        return len(wb_lane_bounds) - 1, 'WB'

    return None, None


def plot_raw_fragments_diagram(raw_fragments, eb_lane_bounds, wb_lane_bounds,
                               num_lanes_eb, num_lanes_wb, max_lanes, min_timestamp, title_prefix=""):
    """Plot time-space diagram showing raw trajectory fragments (only non-empty lanes)."""

    cmap = get_cmap('tab20')

    raw_trajectories = {
        "EB": {f"Lane {i+1}": [] for i in range(max_lanes)},
        "WB": {f"Lane {i+1}": [] for i in range(max_lanes)}
    }

    for frag_idx, frag in enumerate(raw_fragments):
        y_avg = np.mean(frag['y_position'])
        direction = frag['direction']

        lane = None
        direction_str = None

        if direction == 1:
            direction_str = 'EB'
            for i in range(len(eb_lane_bounds) - 1):
                if eb_lane_bounds[i] <= y_avg < eb_lane_bounds[i + 1]:
                    lane = i + 1
                    break
            if lane is None:
                lane = len(eb_lane_bounds) - 1
        elif direction == -1:
            direction_str = 'WB'
            for i in range(len(wb_lane_bounds) - 1):
                if wb_lane_bounds[i] <= y_avg < wb_lane_bounds[i + 1]:
                    lane = i + 1
                    break
            if lane is None:
                lane = len(wb_lane_bounds) - 1

        if lane is None or lane > max_lanes or direction_str is None:
            continue

        timestamps = [t - min_timestamp for t in frag['timestamp']]
        x_positions = frag['x_position']

        color = cmap(frag_idx % 20)
        lane_key = f"Lane {lane}"
        raw_trajectories[direction_str][lane_key].append((timestamps, x_positions, color, frag_idx))

    # Find which lanes have data
    eb_populated_lanes = [i+1 for i in range(max_lanes) if len(raw_trajectories["EB"][f"Lane {i + 1}"]) > 0]
    wb_populated_lanes = [i+1 for i in range(max_lanes) if len(raw_trajectories["WB"][f"Lane {i + 1}"]) > 0]

    # Determine actual number of columns to display
    num_cols = max(len(eb_populated_lanes), len(wb_populated_lanes))
    if num_cols == 0:
        print("  No data to plot!")
        return None

    fig, axs = plt.subplots(2, num_cols, figsize=(8 * num_cols, 16),
                            sharex='all', sharey='row')

    if num_cols == 1:
        axs = np.array([axs]).T

    # Plot EB lanes (only populated ones)
    for plot_idx, lane_num in enumerate(eb_populated_lanes):
        axs[0, plot_idx].set_title(f"EB Lane {lane_num} (Raw Fragments)", fontsize=14, fontweight='bold', pad=15)
        for timestamps, x_positions, color, frag_id in raw_trajectories["EB"][f"Lane {lane_num}"]:
            axs[0, plot_idx].plot(timestamps, x_positions, 'o-', color=color, alpha=0.7,
                          linewidth=2.5, markersize=1)
        axs[0, plot_idx].grid(True, alpha=0.3, linewidth=0.8)
        axs[0, plot_idx].set_ylabel("X-Position (ft)", fontsize=13, fontweight='bold', labelpad=10)
        axs[0, plot_idx].tick_params(axis='both', which='major', labelsize=11)

    # Plot WB lanes (only populated ones)
    for plot_idx, lane_num in enumerate(wb_populated_lanes):
        axs[1, plot_idx].set_title(f"WB Lane {lane_num} (Raw Fragments)", fontsize=14, fontweight='bold', pad=15)
        for timestamps, x_positions, color, frag_id in raw_trajectories["WB"][f"Lane {lane_num}"]:
            axs[1, plot_idx].plot(timestamps, x_positions, 'o-', color=color, alpha=0.7,
                          linewidth=2.5, markersize=1)
        axs[1, plot_idx].grid(True, alpha=0.3, linewidth=0.8)
        axs[1, plot_idx].set_ylabel("X-Position (ft)", fontsize=13, fontweight='bold', labelpad=10)
        axs[1, plot_idx].tick_params(axis='both', which='major', labelsize=11)

    for ax in axs[1, :]:
        ax.set_xlabel("Time (s)", fontsize=13, fontweight='bold', labelpad=10)

    title = f"{title_prefix}Raw Fragments Time-Space Diagram (Before Reconstruction)" if title_prefix else "Raw Fragments Time-Space Diagram (Before Reconstruction)"
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.998)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.05,
                        wspace=0.25, hspace=0.20)

    print(f"  EB Lanes plotted: {eb_populated_lanes}")
    print(f"  WB Lanes plotted: {wb_populated_lanes}")

    return fig


def plot_predicted_diagram(trajectories, eb_lane_bounds, wb_lane_bounds,
                          num_lanes_eb, num_lanes_wb, max_lanes, min_timestamp, title_prefix=""):
    """Plot time-space diagram showing model's predicted trajectories (only non-empty lanes)."""

    cmap = get_cmap('tab20')

    predicted_trajectories = {
        "EB": {f"Lane {i+1}": [] for i in range(max_lanes)},
        "WB": {f"Lane {i+1}": [] for i in range(max_lanes)}
    }

    for traj_idx, traj_fragments in enumerate(trajectories):
        lane, direction_str = assign_lane_to_trajectory(traj_fragments, eb_lane_bounds, wb_lane_bounds)

        if lane is None or lane > max_lanes:
            continue

        all_x = []
        all_t = []
        for frag in traj_fragments:
            all_x.extend(frag['x_position'])
            all_t.extend([t - min_timestamp for t in frag['timestamp']])

        color = cmap(traj_idx % 20)
        lane_key = f"Lane {lane}"
        predicted_trajectories[direction_str][lane_key].append((all_t, all_x, color, traj_idx))

    # Find which lanes have data
    eb_populated_lanes = [i+1 for i in range(max_lanes) if len(predicted_trajectories["EB"][f"Lane {i + 1}"]) > 0]
    wb_populated_lanes = [i+1 for i in range(max_lanes) if len(predicted_trajectories["WB"][f"Lane {i + 1}"]) > 0]

    # Determine actual number of columns to display
    num_cols = max(len(eb_populated_lanes), len(wb_populated_lanes))
    if num_cols == 0:
        print("  No data to plot!")
        return None

    fig, axs = plt.subplots(2, num_cols, figsize=(8 * num_cols, 16),
                            sharex='all', sharey='row')

    if num_cols == 1:
        axs = np.array([axs]).T

    # Plot EB lanes (only populated ones)
    for plot_idx, lane_num in enumerate(eb_populated_lanes):
        axs[0, plot_idx].set_title(f"EB Lane {lane_num} (Siamese Reconstructed)", fontsize=14, fontweight='bold', pad=15)
        for timestamps, x_positions, color, traj_id in predicted_trajectories["EB"][f"Lane {lane_num}"]:
            axs[0, plot_idx].plot(timestamps, x_positions, color=color, alpha=0.7, linewidth=2.5)
        axs[0, plot_idx].grid(True, alpha=0.3, linewidth=0.8)
        axs[0, plot_idx].set_ylabel("X-Position (ft)", fontsize=13, fontweight='bold', labelpad=10)
        axs[0, plot_idx].tick_params(axis='both', which='major', labelsize=11)

    # Plot WB lanes (only populated ones)
    for plot_idx, lane_num in enumerate(wb_populated_lanes):
        axs[1, plot_idx].set_title(f"WB Lane {lane_num} (Siamese Reconstructed)", fontsize=14, fontweight='bold', pad=15)
        for timestamps, x_positions, color, traj_id in predicted_trajectories["WB"][f"Lane {lane_num}"]:
            axs[1, plot_idx].plot(timestamps, x_positions, color=color, alpha=0.7, linewidth=2.5)
        axs[1, plot_idx].grid(True, alpha=0.3, linewidth=0.8)
        axs[1, plot_idx].set_ylabel("X-Position (ft)", fontsize=13, fontweight='bold', labelpad=10)
        axs[1, plot_idx].tick_params(axis='both', which='major', labelsize=11)

    for ax in axs[1, :]:
        ax.set_xlabel("Time (s)", fontsize=13, fontweight='bold', labelpad=10)

    title = f"{title_prefix}Siamese BiLSTM Reconstructed Trajectories" if title_prefix else "Siamese BiLSTM Reconstructed Trajectories"
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.998)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.05,
                        wspace=0.25, hspace=0.20)

    print(f"  EB Lanes plotted: {eb_populated_lanes}")
    print(f"  WB Lanes plotted: {wb_populated_lanes}")

    return fig


def plot_comparison_diagram(raw_fragments, trajectories, eb_lane_bounds, wb_lane_bounds,
                            num_lanes_eb, num_lanes_wb, max_lanes, min_timestamp,
                            scenario: str, output_path: Path = None):
    """Plot side-by-side comparison: Raw fragments (left) vs Reconstructed (right)."""

    cmap = get_cmap('tab20')

    # Organize raw fragments by lane
    raw_by_lane = {
        "EB": {f"Lane {i+1}": [] for i in range(max_lanes)},
        "WB": {f"Lane {i+1}": [] for i in range(max_lanes)}
    }

    for frag_idx, frag in enumerate(raw_fragments):
        y_avg = np.mean(frag['y_position'])
        direction = frag['direction']

        lane = None
        direction_str = None

        if direction == 1:
            direction_str = 'EB'
            for i in range(len(eb_lane_bounds) - 1):
                if eb_lane_bounds[i] <= y_avg < eb_lane_bounds[i + 1]:
                    lane = i + 1
                    break
            if lane is None:
                lane = len(eb_lane_bounds) - 1
        elif direction == -1:
            direction_str = 'WB'
            for i in range(len(wb_lane_bounds) - 1):
                if wb_lane_bounds[i] <= y_avg < wb_lane_bounds[i + 1]:
                    lane = i + 1
                    break
            if lane is None:
                lane = len(wb_lane_bounds) - 1

        if lane is None or lane > max_lanes or direction_str is None:
            continue

        timestamps = [t - min_timestamp for t in frag['timestamp']]
        x_positions = frag['x_position']
        color = cmap(frag_idx % 20)
        raw_by_lane[direction_str][f"Lane {lane}"].append((timestamps, x_positions, color, frag_idx))

    # Organize reconstructed trajectories by lane
    recon_by_lane = {
        "EB": {f"Lane {i+1}": [] for i in range(max_lanes)},
        "WB": {f"Lane {i+1}": [] for i in range(max_lanes)}
    }

    for traj_idx, traj_fragments in enumerate(trajectories):
        lane, direction_str = assign_lane_to_trajectory(traj_fragments, eb_lane_bounds, wb_lane_bounds)

        if lane is None or lane > max_lanes:
            continue

        all_x = []
        all_t = []
        for frag in traj_fragments:
            all_x.extend(frag['x_position'])
            all_t.extend([t - min_timestamp for t in frag['timestamp']])

        color = cmap(traj_idx % 20)
        recon_by_lane[direction_str][f"Lane {lane}"].append((all_t, all_x, color, traj_idx))

    # Find populated lanes (union of raw and reconstructed)
    eb_populated = sorted(set(
        [i+1 for i in range(max_lanes) if len(raw_by_lane["EB"][f"Lane {i+1}"]) > 0] +
        [i+1 for i in range(max_lanes) if len(recon_by_lane["EB"][f"Lane {i+1}"]) > 0]
    ))
    wb_populated = sorted(set(
        [i+1 for i in range(max_lanes) if len(raw_by_lane["WB"][f"Lane {i+1}"]) > 0] +
        [i+1 for i in range(max_lanes) if len(recon_by_lane["WB"][f"Lane {i+1}"]) > 0]
    ))

    num_cols = max(len(eb_populated), len(wb_populated), 1)

    # Create figure: 4 rows (EB Raw, EB Recon, WB Raw, WB Recon) x num_cols
    fig, axs = plt.subplots(4, num_cols, figsize=(7 * num_cols, 20), sharex='col')

    if num_cols == 1:
        axs = axs.reshape(-1, 1)

    # Row 0: EB Raw
    for plot_idx, lane_num in enumerate(eb_populated):
        ax = axs[0, plot_idx]
        ax.set_title(f"EB Lane {lane_num} - Raw Fragments", fontsize=12, fontweight='bold')
        for timestamps, x_positions, color, _ in raw_by_lane["EB"][f"Lane {lane_num}"]:
            ax.plot(timestamps, x_positions, 'o-', color=color, alpha=0.7, linewidth=2, markersize=1)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("X-Position (ft)", fontsize=11)

    # Row 1: EB Reconstructed
    for plot_idx, lane_num in enumerate(eb_populated):
        ax = axs[1, plot_idx]
        ax.set_title(f"EB Lane {lane_num} - Reconstructed", fontsize=12, fontweight='bold')
        for timestamps, x_positions, color, _ in recon_by_lane["EB"][f"Lane {lane_num}"]:
            ax.plot(timestamps, x_positions, color=color, alpha=0.8, linewidth=2.5)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("X-Position (ft)", fontsize=11)

    # Row 2: WB Raw
    for plot_idx, lane_num in enumerate(wb_populated):
        ax = axs[2, plot_idx]
        ax.set_title(f"WB Lane {lane_num} - Raw Fragments", fontsize=12, fontweight='bold')
        for timestamps, x_positions, color, _ in raw_by_lane["WB"][f"Lane {lane_num}"]:
            ax.plot(timestamps, x_positions, 'o-', color=color, alpha=0.7, linewidth=2, markersize=1)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("X-Position (ft)", fontsize=11)

    # Row 3: WB Reconstructed
    for plot_idx, lane_num in enumerate(wb_populated):
        ax = axs[3, plot_idx]
        ax.set_title(f"WB Lane {lane_num} - Reconstructed", fontsize=12, fontweight='bold')
        for timestamps, x_positions, color, _ in recon_by_lane["WB"][f"Lane {lane_num}"]:
            ax.plot(timestamps, x_positions, color=color, alpha=0.8, linewidth=2.5)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("X-Position (ft)", fontsize=11)
        ax.set_xlabel("Time (s)", fontsize=11)

    plt.suptitle(f"Scenario {scenario}: Raw Fragments vs Siamese BiLSTM Reconstruction (by Lane)",
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    return fig


def plot_training_history(history_path: Path, output_path: Path = None):
    """Plot training curves from saved history."""

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axs[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axs[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axs[0, 0].set_xlabel('Epoch', fontsize=12)
    axs[0, 0].set_ylabel('Total Loss', fontsize=12)
    axs[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axs[0, 0].legend(fontsize=11)
    axs[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axs[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train', linewidth=2)
    axs[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axs[0, 1].set_xlabel('Epoch', fontsize=12)
    axs[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axs[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axs[0, 1].legend(fontsize=11)
    axs[0, 1].grid(True, alpha=0.3)

    # Best accuracy annotation
    best_val_acc = max(history['val_accuracy'])
    best_epoch = history['val_accuracy'].index(best_val_acc) + 1
    axs[0, 1].axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.7)
    axs[0, 1].annotate(f'Best: {best_val_acc:.2f}% (Epoch {best_epoch})',
                       xy=(best_epoch, best_val_acc), fontsize=10,
                       xytext=(best_epoch + 5, best_val_acc + 2))

    # Loss components
    axs[1, 0].plot(epochs, history['train_bce'], 'b-', label='Train BCE', linewidth=2)
    axs[1, 0].plot(epochs, history['val_bce'], 'b--', label='Val BCE', linewidth=2)
    axs[1, 0].plot(epochs, history['train_contrastive'], 'r-', label='Train Contrastive', linewidth=2)
    axs[1, 0].plot(epochs, history['val_contrastive'], 'r--', label='Val Contrastive', linewidth=2)
    axs[1, 0].set_xlabel('Epoch', fontsize=12)
    axs[1, 0].set_ylabel('Loss', fontsize=12)
    axs[1, 0].set_title('Loss Components', fontsize=14, fontweight='bold')
    axs[1, 0].legend(fontsize=10)
    axs[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axs[1, 1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axs[1, 1].set_xlabel('Epoch', fontsize=12)
    axs[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axs[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_yscale('log')

    plt.suptitle('Siamese BiLSTM Training History', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {len(epochs)}")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final train accuracy: {history['train_accuracy'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main visualization pipeline."""

    print("="*70)
    print("SIAMESE BiLSTM TRAJECTORY RECONSTRUCTION VISUALIZATION")
    print("="*70)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Plot training history
    print("\n[1/4] Plotting training history...")
    history_path = OUTPUT_DIR / "training_history.json"
    if history_path.exists():
        fig1 = plot_training_history(history_path, OUTPUT_DIR / "siamese_training_curves.png")
        plt.close(fig1)
    else:
        print(f"  Warning: Training history not found at {history_path}")

    # 2. Load model
    print("\n[2/4] Loading trained Siamese model...")
    if not MODEL_PATH.exists():
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        print("  Please train the model first using train_siamese.py")
        return

    reconstructor = SiameseReconstructor(MODEL_PATH)

    # 3. Process each scenario
    scenarios = ['i', 'ii', 'iii']
    all_results = {}

    for scenario in scenarios:
        print(f"\n[3/4] Processing scenario {scenario}...")

        try:
            gt_data, raw_data = load_scenario_data(scenario)
        except FileNotFoundError as e:
            print(f"  Warning: Could not load scenario {scenario}: {e}")
            continue

        # Calculate lane bounds
        eb_bounds, wb_bounds, num_eb, num_wb, min_t = calculate_lane_bounds(gt_data, raw_data)

        # Reconstruct trajectories
        reconstructed = reconstructor.reconstruct_trajectories(
            raw_data,
            threshold=0.5,
            max_time_gap=5.0,
            max_spatial_gap=200.0,
            max_y_diff=8.0
        )

        all_results[scenario] = {
            'gt': gt_data,
            'raw': raw_data,
            'reconstructed': reconstructed,
            'eb_bounds': eb_bounds,
            'wb_bounds': wb_bounds,
            'min_t': min_t
        }

        max_lanes = max(num_eb, num_wb)

        # Plot raw fragments by lane
        print(f"\n  Plotting raw fragments by lane...")
        fig_raw = plot_raw_fragments_diagram(
            raw_data, eb_bounds, wb_bounds, num_eb, num_wb, max_lanes, min_t,
            title_prefix=f"Scenario {scenario}: "
        )
        if fig_raw:
            fig_raw.savefig(OUTPUT_DIR / f"scenario_{scenario}_raw_fragments.png", dpi=300, bbox_inches='tight')
            print(f"  Saved: scenario_{scenario}_raw_fragments.png")
            plt.close(fig_raw)

        # Plot reconstructed trajectories by lane
        print(f"\n  Plotting reconstructed trajectories by lane...")
        fig_recon = plot_predicted_diagram(
            reconstructed, eb_bounds, wb_bounds, num_eb, num_wb, max_lanes, min_t,
            title_prefix=f"Scenario {scenario}: "
        )
        if fig_recon:
            fig_recon.savefig(OUTPUT_DIR / f"scenario_{scenario}_reconstructed.png", dpi=300, bbox_inches='tight')
            print(f"  Saved: scenario_{scenario}_reconstructed.png")
            plt.close(fig_recon)

        # Plot comparison (raw vs reconstructed side by side)
        print(f"\n  Plotting comparison diagram...")
        fig_comp = plot_comparison_diagram(
            raw_data, reconstructed, eb_bounds, wb_bounds, num_eb, num_wb, max_lanes, min_t,
            scenario, OUTPUT_DIR / f"scenario_{scenario}_comparison.png"
        )
        if fig_comp:
            plt.close(fig_comp)

    # 4. Summary statistics
    print("\n[4/4] Computing summary statistics...")
    print("\n" + "="*70)
    print("RECONSTRUCTION RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Scenario':<12} {'GT Vehicles':<15} {'Raw Frags':<15} {'Reconstructed':<15} {'Frag Ratio':<12}")
    print("-"*70)

    for scenario, results in all_results.items():
        gt_count = len(results['gt'])
        raw_count = len(results['raw'])
        recon_count = len(results['reconstructed'])
        frag_ratio = recon_count / gt_count if gt_count > 0 else 0

        print(f"{scenario:<12} {gt_count:<15} {raw_count:<15} {recon_count:<15} {frag_ratio:<12.2f}")

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in OUTPUT_DIR.glob("siamese_*.png"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
