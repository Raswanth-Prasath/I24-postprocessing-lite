"""
Time-Space Diagram Visualization for Trajectory Reconstruction

Compares three views:
1. Raw Fragments (before reconstruction)
2. Ground Truth (complete trajectories)
3. Predicted Trajectories (model reconstruction)

Supports both Logistic Regression and Siamese Network models.
"""

import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.cm import get_cmap
from pathlib import Path
import pickle
import torch
from typing import List, Dict, Tuple

# For Logistic Regression
from sklearn.preprocessing import StandardScaler

# For Siamese Network (optional)
try:
    from siamese_model import SiameseTrajectoryNetwork
    SIAMESE_AVAILABLE = True
except:
    SIAMESE_AVAILABLE = False


# ============================================================================
# FEATURE EXTRACTION (for Logistic Regression)
# ============================================================================

def extract_lr_features(frag_a: Dict, frag_b: Dict) -> List[float]:
    """
    Extract features for Logistic Regression model
    (matches the features from your LR training)
    """
    features = []

    # Temporal features
    time_gap = frag_b['first_timestamp'] - frag_a['last_timestamp']
    duration_a = frag_a['last_timestamp'] - frag_a['first_timestamp']
    duration_b = frag_b['last_timestamp'] - frag_b['first_timestamp']
    duration_ratio = duration_a / (duration_b + 1e-6)

    features.extend([time_gap, duration_a, duration_b, duration_ratio])

    # Spatial features
    if frag_a['direction'] == 1:  # Eastbound
        spatial_gap = frag_b['starting_x'] - frag_a['ending_x']
    else:  # Westbound
        spatial_gap = frag_a['starting_x'] - frag_b['ending_x']

    features.append(spatial_gap)

    # Lateral features
    y_mean_a = np.mean(frag_a['y_position'])
    y_mean_b = np.mean(frag_b['y_position'])
    y_diff = abs(y_mean_b - y_mean_a)
    y_std_a = np.std(frag_a['y_position'])
    y_std_b = np.std(frag_b['y_position'])

    features.extend([y_mean_a, y_mean_b, y_diff, y_std_a, y_std_b])

    # Kinematic features
    if 'velocity' in frag_a and 'velocity' in frag_b:
        vel_a_end = frag_a['velocity'][-1] if len(frag_a['velocity']) > 0 else 0
        vel_b_start = frag_b['velocity'][0] if len(frag_b['velocity']) > 0 else 0
        vel_a_mean = np.mean(frag_a['velocity'])
        vel_b_mean = np.mean(frag_b['velocity'])
        vel_diff = abs(vel_a_end - vel_b_start)
        vel_ratio = vel_a_end / (vel_b_start + 1e-6)
    else:
        vel_a_mean = vel_b_mean = vel_diff = 0
        vel_ratio = 1.0

    features.extend([vel_a_mean, vel_b_mean, vel_diff, vel_ratio])

    # Vehicle dimension features
    length_a = np.mean(frag_a['length'])
    length_b = np.mean(frag_b['length'])
    length_diff = abs(length_a - length_b)
    width_a = np.mean(frag_a['width'])
    width_b = np.mean(frag_b['width'])
    width_diff = abs(width_a - width_b)

    features.extend([length_a, length_b, length_diff, width_a, width_b, width_diff])

    # Detection confidence
    if 'detection_confidence' in frag_a and 'detection_confidence' in frag_b:
        conf_a_mean = np.mean(frag_a['detection_confidence'])
        conf_b_mean = np.mean(frag_b['detection_confidence'])
        conf_a_min = np.min(frag_a['detection_confidence'])
        conf_b_min = np.min(frag_b['detection_confidence'])
    else:
        conf_a_mean = conf_b_mean = conf_a_min = conf_b_min = 1.0

    features.extend([conf_a_mean, conf_b_mean, conf_a_min, conf_b_min])

    # Direction match
    direction_match = 1 if frag_a['direction'] == frag_b['direction'] else 0
    features.append(direction_match)

    return features


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_dir: Path, scenario: str = 'i'):
    """Load GT and RAW data for a scenario"""
    gt_path = data_dir / f"GT_{scenario}.json"
    raw_path = data_dir / f"RAW_{scenario}.json"

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)

    print(f"Loaded scenario {scenario}:")
    print(f"  Ground truth: {len(gt_data)} vehicles")
    print(f"  Raw fragments: {len(raw_data)} fragments")

    return gt_data, raw_data


def calculate_lane_bounds(gt_data, raw_data):
    """Calculate lane boundaries from data"""
    eb_y_positions = []
    wb_y_positions = []
    all_timestamps = []

    # Collect from GT
    for vehicle in gt_data:
        if 'direction' in vehicle and vehicle['direction'] in [1, -1]:
            y_positions = vehicle['y_position']
            all_timestamps.extend(vehicle['timestamp'])
            if vehicle['direction'] == 1:
                eb_y_positions.extend(y_positions)
            else:
                wb_y_positions.extend(y_positions)

    # Collect from RAW
    for fragment in raw_data:
        if 'direction' in fragment and fragment['direction'] in [1, -1]:
            y_positions = fragment['y_position']
            all_timestamps.extend(fragment['timestamp'])
            if fragment['direction'] == 1:
                eb_y_positions.extend(y_positions)
            else:
                wb_y_positions.extend(y_positions)

    # Find bounds
    min_y_eb = min(eb_y_positions) if eb_y_positions else 0
    max_y_eb = max(eb_y_positions) if eb_y_positions else 12
    min_y_wb = min(wb_y_positions) if wb_y_positions else -12
    max_y_wb = max(wb_y_positions) if wb_y_positions else 0

    # Calculate lanes (12 ft per lane)
    num_lanes_eb = int(np.ceil((max_y_eb - min_y_eb) / 12))
    num_lanes_wb = int(np.ceil((max_y_wb - min_y_wb) / 12))

    eb_lane_bounds = [min_y_eb + i * 12 for i in range(num_lanes_eb + 1)]
    wb_lane_bounds = [max_y_wb - i * 12 for i in range(num_lanes_wb + 1)][::-1]

    min_timestamp = min(all_timestamps) if all_timestamps else 0

    return eb_lane_bounds, wb_lane_bounds, num_lanes_eb, num_lanes_wb, min_timestamp


# ============================================================================
# TRAJECTORY RECONSTRUCTION
# ============================================================================

def reconstruct_trajectories_lr(fragments, model_path: Path, scaler_path: Path, threshold: float = 0.5):
    """
    Reconstruct trajectories using Logistic Regression model
    """
    print("\nReconstrucing trajectories with Logistic Regression...")

    # Load model and scaler
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"Loaded LR model from {model_path}")

    # Sort fragments by time
    sorted_fragments = sorted(fragments, key=lambda x: x['first_timestamp'])

    trajectories = []
    used_fragments = set()

    for i, frag_a in enumerate(sorted_fragments):
        if i in used_fragments:
            continue

        # Start new trajectory
        current_trajectory = [frag_a]
        current_frag = frag_a

        # Try to find next fragment
        while True:
            best_match = None
            best_prob = threshold

            for j, frag_b in enumerate(sorted_fragments):
                if j in used_fragments or j <= i:
                    continue

                # Basic constraints
                if current_frag['last_timestamp'] >= frag_b['first_timestamp']:
                    continue
                if current_frag['direction'] != frag_b['direction']:
                    continue

                time_gap = frag_b['first_timestamp'] - current_frag['last_timestamp']
                if time_gap > 5.0:
                    continue

                spatial_gap = abs(frag_b['starting_x'] - current_frag['ending_x'])
                if spatial_gap > 200:
                    continue

                y_diff = abs(np.mean(frag_b['y_position']) - np.mean(current_frag['y_position']))
                if y_diff > 5.0:
                    continue

                # Extract features and predict
                features = extract_lr_features(current_frag, frag_b)
                features_scaled = scaler.transform([features])
                prob = model.predict_proba(features_scaled)[0, 1]

                if prob > best_prob:
                    best_prob = prob
                    best_match = (j, frag_b)

            if best_match is not None:
                idx, matched_frag = best_match
                current_trajectory.append(matched_frag)
                used_fragments.add(idx)
                current_frag = matched_frag
            else:
                break

        trajectories.append(current_trajectory)
        used_fragments.add(i)

    print(f"Reconstructed {len(trajectories)} trajectories from {len(fragments)} fragments")

    return trajectories


def reconstruct_trajectories_siamese(fragments, model_path: Path, threshold: float = 0.5):
    """
    Reconstruct trajectories using Siamese Network model
    """
    if not SIAMESE_AVAILABLE:
        print("Siamese Network not available. Install PyTorch and siamese_model.py")
        return []

    print("\nReconstrucing trajectories with Siamese Network...")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseTrajectoryNetwork(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        similarity_hidden_dim=64
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded Siamese model from {model_path}")

    # TODO: Implement Siamese reconstruction logic
    # This would use the Siamese network to score pairs
    # For now, return empty list
    print("Siamese reconstruction not yet implemented in this script")

    return []


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def assign_lane(y_positions, direction, eb_bounds, wb_bounds):
    """Assign fragment/trajectory to a lane"""
    y_avg = np.mean(y_positions)

    if direction == 1:  # Eastbound
        for i in range(len(eb_bounds) - 1):
            if eb_bounds[i] <= y_avg < eb_bounds[i + 1]:
                return i + 1, 'EB'
        return len(eb_bounds) - 1, 'EB'
    else:  # Westbound
        for i in range(len(wb_bounds) - 1):
            if wb_bounds[i] <= y_avg < wb_bounds[i + 1]:
                return i + 1, 'WB'
        return len(wb_bounds) - 1, 'WB'


def plot_comparison(raw_data, gt_data, predicted_trajectories,
                   eb_bounds, wb_bounds, num_lanes_eb, num_lanes_wb,
                   min_timestamp, output_path: Path = None):
    """
    Create comprehensive comparison plot showing:
    - Raw fragments (top)
    - Ground truth (middle)
    - Predicted trajectories (bottom)
    """

    max_lanes = max(num_lanes_eb, num_lanes_wb)
    cmap = get_cmap('tab20')

    # Organize data by lane
    def organize_by_lane(data_list, is_fragments=False):
        result = {
            "EB": {f"Lane {i+1}": [] for i in range(max_lanes)},
            "WB": {f"Lane {i+1}": [] for i in range(max_lanes)}
        }

        for idx, item in enumerate(data_list):
            if is_fragments:
                # Item is a list of fragments (trajectory)
                all_y = []
                for frag in item:
                    all_y.extend(frag['y_position'])
                direction = item[0]['direction']
            else:
                # Item is a vehicle/fragment
                all_y = item['y_position']
                direction = item['direction']

            lane, dir_str = assign_lane(all_y, direction, eb_bounds, wb_bounds)

            if lane > max_lanes:
                continue

            # Get positions and timestamps
            if is_fragments:
                # Combine fragments
                all_x = []
                all_t = []
                for frag in item:
                    all_x.extend(frag['x_position'])
                    all_t.extend([t - min_timestamp for t in frag['timestamp']])
            else:
                all_x = item['x_position']
                all_t = [t - min_timestamp for t in item['timestamp']]

            color = cmap(idx % 20)
            result[dir_str][f"Lane {lane}"].append((all_t, all_x, color, idx))

        return result

    # Organize all three datasets
    raw_organized = organize_by_lane(raw_data, is_fragments=False)
    gt_organized = organize_by_lane(gt_data, is_fragments=False)
    pred_organized = organize_by_lane(predicted_trajectories, is_fragments=True)

    # Create figure with 3 rows
    fig, axs = plt.subplots(6, max_lanes, figsize=(8 * max_lanes, 24),
                           sharex='all', sharey='row')

    if max_lanes == 1:
        axs = axs.reshape(-1, 1)

    # Row 0-1: Raw Fragments
    for i in range(max_lanes):
        # EB
        axs[0, i].set_title(f"EB Lane {i+1} - Raw Fragments",
                           fontsize=14, fontweight='bold', pad=15)
        for t, x, color, idx in raw_organized["EB"][f"Lane {i+1}"]:
            axs[0, i].plot(t, x, 'o-', color=color, alpha=0.6, linewidth=2, markersize=1)
        axs[0, i].grid(True, alpha=0.3)
        axs[0, i].set_ylabel("X-Position (ft)", fontsize=12, fontweight='bold')

        # WB
        axs[1, i].set_title(f"WB Lane {i+1} - Raw Fragments",
                           fontsize=14, fontweight='bold', pad=15)
        for t, x, color, idx in raw_organized["WB"][f"Lane {i+1}"]:
            axs[1, i].plot(t, x, 'o-', color=color, alpha=0.6, linewidth=2, markersize=1)
        axs[1, i].grid(True, alpha=0.3)
        axs[1, i].set_ylabel("X-Position (ft)", fontsize=12, fontweight='bold')

    # Row 2-3: Ground Truth
    for i in range(max_lanes):
        # EB
        axs[2, i].set_title(f"EB Lane {i+1} - Ground Truth",
                           fontsize=14, fontweight='bold', pad=15)
        for t, x, color, idx in gt_organized["EB"][f"Lane {i+1}"]:
            axs[2, i].plot(t, x, color=color, alpha=0.7, linewidth=2.5)
        axs[2, i].grid(True, alpha=0.3)
        axs[2, i].set_ylabel("X-Position (ft)", fontsize=12, fontweight='bold')

        # WB
        axs[3, i].set_title(f"WB Lane {i+1} - Ground Truth",
                           fontsize=14, fontweight='bold', pad=15)
        for t, x, color, idx in gt_organized["WB"][f"Lane {i+1}"]:
            axs[3, i].plot(t, x, color=color, alpha=0.7, linewidth=2.5)
        axs[3, i].grid(True, alpha=0.3)
        axs[3, i].set_ylabel("X-Position (ft)", fontsize=12, fontweight='bold')

    # Row 4-5: Predicted
    for i in range(max_lanes):
        # EB
        axs[4, i].set_title(f"EB Lane {i+1} - Model Prediction",
                           fontsize=14, fontweight='bold', pad=15)
        for t, x, color, idx in pred_organized["EB"][f"Lane {i+1}"]:
            axs[4, i].plot(t, x, color=color, alpha=0.7, linewidth=2.5)
        axs[4, i].grid(True, alpha=0.3)
        axs[4, i].set_ylabel("X-Position (ft)", fontsize=12, fontweight='bold')

        # WB
        axs[5, i].set_title(f"WB Lane {i+1} - Model Prediction",
                           fontsize=14, fontweight='bold', pad=15)
        for t, x, color, idx in pred_organized["WB"][f"Lane {i+1}"]:
            axs[5, i].plot(t, x, color=color, alpha=0.7, linewidth=2.5)
        axs[5, i].grid(True, alpha=0.3)
        axs[5, i].set_ylabel("X-Position (ft)", fontsize=12, fontweight='bold')

    # X labels for bottom row
    for ax in axs[5, :]:
        ax.set_xlabel("Time (s)", fontsize=13, fontweight='bold', labelpad=10)

    plt.suptitle("Trajectory Reconstruction Comparison: Raw → Ground Truth → Model Prediction",
                fontsize=20, fontweight='bold', y=0.999)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.975, bottom=0.03,
                       wspace=0.20, hspace=0.30)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot to: {output_path}")

    plt.show()

    # Print statistics
    print("\n" + "="*70)
    print("COMPARISON STATISTICS")
    print("="*70)
    print(f"Raw fragments: {len(raw_data)}")
    print(f"Ground truth vehicles: {len(gt_data)}")
    print(f"Predicted trajectories: {len(predicted_trajectories)}")

    # Count by direction
    raw_eb = sum(1 for f in raw_data if f['direction'] == 1)
    raw_wb = sum(1 for f in raw_data if f['direction'] == -1)
    gt_eb = sum(1 for v in gt_data if v['direction'] == 1)
    gt_wb = sum(1 for v in gt_data if v['direction'] == -1)

    print(f"\nRaw: EB={raw_eb}, WB={raw_wb}")
    print(f"GT:  EB={gt_eb}, WB={gt_wb}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main visualization function"""
    print("="*70)
    print("TRAJECTORY RECONSTRUCTION VISUALIZATION")
    print("="*70)
    print()

    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    output_dir = script_dir / "outputs"

    # Configuration
    scenario = 'i'  # Change to 'ii' or 'iii' for other scenarios
    model_type = 'lr'  # 'lr' or 'siamese'

    # Load data
    print(f"Loading scenario {scenario}...")
    gt_data, raw_data = load_data(data_dir, scenario)

    # Calculate lane bounds
    eb_bounds, wb_bounds, num_eb, num_wb, min_t = calculate_lane_bounds(gt_data, raw_data)

    print(f"\nLane configuration:")
    print(f"  Eastbound: {num_eb} lanes, bounds: {eb_bounds}")
    print(f"  Westbound: {num_wb} lanes, bounds: {wb_bounds}")

    # Reconstruct trajectories
    if model_type == 'lr':
        # For Logistic Regression (you need to save model and scaler first)
        model_path = Path(r"D:\ASU Academics\Thesis & Research\02_Code\Logistic-Regression\outputs\lr_model.pkl")
        scaler_path = Path(r"D:\ASU Academics\Thesis & Research\02_Code\Logistic-Regression\outputs\scaler.pkl")

        if model_path.exists() and scaler_path.exists():
            predicted = reconstruct_trajectories_lr(raw_data, model_path, scaler_path)
        else:
            print(f"\nWarning: LR model not found at {model_path}")
            print("Creating empty predictions for visualization")
            predicted = []
    else:
        # For Siamese Network
        model_path = output_dir / "best_accuracy.pth"
        if model_path.exists():
            predicted = reconstruct_trajectories_siamese(raw_data, model_path)
        else:
            print(f"\nWarning: Siamese model not found at {model_path}")
            print("Train the model first using train_siamese.py")
            predicted = []

    # Create comparison plot
    output_path = output_dir / f"comparison_scenario_{scenario}.png"
    output_dir.mkdir(exist_ok=True)

    plot_comparison(
        raw_data,
        gt_data,
        predicted,
        eb_bounds,
        wb_bounds,
        num_eb,
        num_wb,
        min_t,
        output_path
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
