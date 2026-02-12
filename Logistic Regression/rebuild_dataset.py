"""
Rebuild Training Dataset v3 for Logistic Regression Cost Function

Generates a balanced, augmented dataset by masking GT trajectories:
- Positive pairs: Split a GT trajectory at a mask region → before/after fragments
- Negative pairs: Cross-pair before(V1) + after(V2) where V1≠V2, within ±10s, same lane
- Augmentation: Detection noise (x, y, length, width, height, velocity, confidence)
- Upscaling: Multiple augmented copies of each pair

Mask regions simulate camera blind spots at 4 positions along the corridor.
Mask width is randomized per-copy to avoid constant spatial_gap.

Usage:
    conda activate i24
    python "Logistic Regression/rebuild_dataset.py"
    python "Logistic Regression/rebuild_dataset.py" --upscale 5 --output training_dataset_v3_5x.npz
"""

import argparse
import json
import sys
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LR_DIR = Path(__file__).resolve().parent
CANONICAL_OUTPUT_DIR = LR_DIR / "outputs"
CANONICAL_DATA_DIR = LR_DIR / "data"
DATASETS = ['i', 'ii', 'iii']

# 4 mask center positions along the ~2200 ft corridor
MASK_CENTERS = [225, 625, 1125, 1625]
MASK_WIDTH_MEAN = 250  # ft
MASK_WIDTH_STD = 50    # ft, randomized per copy

MIN_FRAGMENT_POINTS = 10

# Detection noise calibrated from RAW data statistics
NOISE = {
    'x_std': 1.5,           # ft per detection
    'y_std': 0.5,           # ft per detection
    'length_std': 0.8,      # ft per detection (RAW mean=0.86)
    'width_std': 0.15,      # ft per detection (RAW mean=0.16)
    'height_std': 0.3,      # ft per detection
    'velocity_std': 4.0,    # ft/s per detection (RAW mean=5.4)
    'conf_mean': 0.92,      # RAW mean=0.918
    'conf_std': 0.11,       # RAW std=0.112
    'point_dropout': 0.05,  # 5% random point dropout
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

siamese_dir = PROJECT_ROOT / "Siamese-Network"
if str(siamese_dir) not in sys.path:
    sys.path.insert(0, str(siamese_dir))


def load_json(path: Path) -> List[Dict]:
    with open(path, 'r') as f:
        return json.load(f)


def get_gt_id(traj: Dict) -> str:
    tid = traj.get('_id', {})
    if isinstance(tid, dict) and '$oid' in tid:
        return tid['$oid']
    return str(tid)


def compute_velocity(timestamps, x_positions):
    """Compute velocity from finite differences of x_position."""
    t = np.array(timestamps)
    x = np.array(x_positions)
    if len(t) < 2:
        return np.zeros(len(t))
    dt = np.diff(t)
    dx = np.diff(x)
    # Avoid division by zero
    dt = np.where(dt == 0, 1e-6, dt)
    vel = dx / dt
    # Pad to same length (repeat last value)
    vel = np.append(vel, vel[-1])
    return vel


def augment_fragment(frag: Dict, rng: np.random.RandomState) -> Dict:
    """
    Apply realistic detection noise to a GT fragment to simulate RAW data.
    """
    frag = deepcopy(frag)
    n = len(frag['timestamp'])

    # X-position noise
    frag['x_position'] = (np.array(frag['x_position']) +
                          rng.normal(0, NOISE['x_std'], n)).tolist()

    # Y-position noise
    frag['y_position'] = (np.array(frag['y_position']) +
                          rng.normal(0, NOISE['y_std'], n)).tolist()

    # Compute velocity from (noisy) positions + add noise
    vel = compute_velocity(frag['timestamp'], frag['x_position'])
    vel += rng.normal(0, NOISE['velocity_std'], n)
    frag['velocity'] = vel.tolist()

    # Length/width/height: add per-detection noise to break identical values
    for key, std in [('length', NOISE['length_std']),
                     ('width', NOISE['width_std']),
                     ('height', NOISE['height_std'])]:
        if key in frag:
            arr = np.array(frag[key])
            if len(arr) == n:
                arr = arr + rng.normal(0, std, n)
                arr = np.maximum(arr, 1.0)  # physical minimum
                frag[key] = arr.tolist()
            elif np.isscalar(frag[key]) or len(arr) == 1:
                base = float(arr[0]) if len(arr) > 0 else 10.0
                frag[key] = (base + rng.normal(0, std, n)).clip(1.0).tolist()

    # Detection confidence
    conf = rng.normal(NOISE['conf_mean'], NOISE['conf_std'], n).clip(0.1, 1.0)
    frag['detection_confidence'] = conf.tolist()

    # Random point dropout
    if n > MIN_FRAGMENT_POINTS * 2:
        keep = rng.random(n) > NOISE['point_dropout']
        # Always keep first and last
        keep[0] = True
        keep[-1] = True
        if keep.sum() >= MIN_FRAGMENT_POINTS:
            for key in ['timestamp', 'x_position', 'y_position', 'velocity',
                        'length', 'width', 'height', 'detection_confidence']:
                if key in frag and isinstance(frag[key], list) and len(frag[key]) == n:
                    frag[key] = np.array(frag[key])[keep].tolist()

    # Update derived fields
    frag['first_timestamp'] = frag['timestamp'][0]
    frag['last_timestamp'] = frag['timestamp'][-1]
    frag['starting_x'] = frag['x_position'][0]
    frag['ending_x'] = frag['x_position'][-1]

    return frag


def make_fragment(traj: Dict, mask: np.ndarray) -> Optional[Dict]:
    """Extract a raw (unaugmented) fragment from a GT trajectory."""
    if mask.sum() < MIN_FRAGMENT_POINTS:
        return None

    n_total = len(traj.get('timestamp', []))
    frag = {}
    for key in ['timestamp', 'x_position', 'y_position']:
        if key in traj:
            frag[key] = np.array(traj[key])[mask].tolist()

    for key in ['length', 'width', 'height']:
        if key in traj:
            val = traj[key]
            if isinstance(val, list) and len(val) == n_total:
                frag[key] = np.array(val)[mask].tolist()
            else:
                frag[key] = val

    # Velocity: compute from positions (GT has no velocity)
    frag['velocity'] = compute_velocity(frag['timestamp'], frag['x_position']).tolist()

    frag['direction'] = traj.get('direction', 1)
    frag['_id'] = get_gt_id(traj) + '_frag'
    frag['first_timestamp'] = frag['timestamp'][0]
    frag['last_timestamp'] = frag['timestamp'][-1]
    frag['starting_x'] = frag['x_position'][0]
    frag['ending_x'] = frag['x_position'][-1]

    return frag


# ── Pair generation ──────────────────────────────────────────────────

def generate_positive_pairs(
    gt_trajectories: List[Dict],
    mask_centers: List[float],
    mask_width: float,
    max_y_diff: float = 5.0,
) -> List[Tuple[Dict, Dict, str, int]]:
    """Split GT trajectories at mask regions into before/after fragment pairs."""
    pairs = []
    half = mask_width / 2

    for traj in gt_trajectories:
        direction = traj.get('direction', 1)
        x_arr = np.array(traj['x_position'])
        gt_id = get_gt_id(traj)

        for mask_idx, center in enumerate(mask_centers):
            mask_lo = center - half
            mask_hi = center + half

            if direction == 1:  # EB
                before_mask = x_arr < mask_lo
                after_mask = x_arr > mask_hi
            else:  # WB
                before_mask = x_arr > mask_hi
                after_mask = x_arr < mask_lo

            frag_before = make_fragment(traj, before_mask)
            frag_after = make_fragment(traj, after_mask)

            if frag_before is None or frag_after is None:
                continue

            if frag_before['last_timestamp'] >= frag_after['first_timestamp']:
                continue

            # y_diff filter (same constraint as negatives)
            y_diff = abs(np.mean(frag_before['y_position']) -
                         np.mean(frag_after['y_position']))
            if y_diff > max_y_diff:
                continue

            pairs.append((frag_before, frag_after, gt_id, mask_idx))

    return pairs


def generate_negative_pairs(
    positive_pairs: List[Tuple[Dict, Dict, str, int]],
    time_window: float = 10.0,
    max_y_diff: float = 5.0,
) -> List[Tuple[Dict, Dict]]:
    """Cross-pair before(V1) + after(V2), different vehicles, within time window."""
    groups = defaultdict(list)
    for frag_before, frag_after, gt_id, mask_idx in positive_pairs:
        direction = frag_before['direction']
        groups[(mask_idx, direction)].append((frag_before, frag_after, gt_id))

    negatives = []
    for (mask_idx, direction), entries in groups.items():
        for i in range(len(entries)):
            before_i, _, id_i = entries[i]
            for j in range(len(entries)):
                if i == j:
                    continue
                _, after_j, id_j = entries[j]
                if id_i == id_j:
                    continue

                t_end = before_i['last_timestamp']
                t_start = after_j['first_timestamp']
                time_gap = t_start - t_end
                if time_gap <= 0 or time_gap > time_window:
                    continue

                y_diff = abs(np.mean(before_i['y_position']) -
                             np.mean(after_j['y_position']))
                if y_diff > max_y_diff:
                    continue

                negatives.append((before_i, after_j))

    return negatives


# ── Feature extraction ───────────────────────────────────────────────

def extract_features(
    pairs: List[Tuple[Dict, Dict]],
    labels: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from utils.features_stitch import StitchFeatureExtractor

    extractor = StitchFeatureExtractor(mode='advanced')
    feature_names = extractor.get_feature_names()
    X_list, y_list = [], []
    skipped = 0

    for (a, b), label in zip(pairs, labels):
        try:
            vec = extractor.extract_feature_vector(a, b)
            vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
            X_list.append(vec)
            y_list.append(label)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: skipped pair: {e}")

    if skipped > 5:
        print(f"  (total skipped: {skipped})")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), feature_names


# ── Diagnostics ──────────────────────────────────────────────────────

def generate_diagnostics(X, y, feat_names, output_path):
    """Generate diagnostic plots for the new dataset."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Key features to check
    check_features = [
        'y_diff', 'time_gap', 'spatial_gap', 'length_diff', 'width_diff',
        'height_diff', 'vel_diff', 'projection_error_x_max',
        'bhattacharyya_coeff',
    ]

    n_feats = len([f for f in check_features if f in feat_names])
    ncols = 3
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    fig.suptitle('Dataset v3 Feature Diagnostics', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    plot_idx = 0
    for feat_name in check_features:
        if feat_name not in feat_names:
            continue
        idx = feat_names.index(feat_name)
        ax = axes[plot_idx]
        pos_vals = X[y == 1, idx]
        neg_vals = X[y == 0, idx]

        # Clip outliers for visibility
        all_vals = np.concatenate([pos_vals, neg_vals])
        clip_lo = np.percentile(all_vals, 1)
        clip_hi = np.percentile(all_vals, 99)

        ax.hist(np.clip(pos_vals, clip_lo, clip_hi), bins=50, alpha=0.6,
                label=f'Pos (n={len(pos_vals)})', color='green', density=True)
        ax.hist(np.clip(neg_vals, clip_lo, clip_hi), bins=50, alpha=0.6,
                label=f'Neg (n={len(neg_vals)})', color='red', density=True)
        ax.set_title(feat_name, fontsize=11)
        ax.legend(fontsize=8)
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Diagnostics saved to: {output_path}")


# ── Time-space visualization ─────────────────────────────────────────

def generate_timespace(pos_pairs, neg_pairs, scenario, direction, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    LANE_WIDTH = 12.0
    N_LANES = 4
    dir_label = 'EB' if direction == 1 else 'WB'

    def get_lane(frag):
        y_mean = np.mean(frag['y_position'])
        if direction == 1:
            return max(0, min(int(y_mean / LANE_WIDTH), N_LANES - 1))
        else:
            return max(0, min(int((y_mean - 72) / LANE_WIDTH), N_LANES - 1))

    dir_pos = [(a, b) for a, b, _, _ in pos_pairs if a['direction'] == direction]
    dir_neg = [(a, b) for a, b in neg_pairs if a['direction'] == direction]

    if not dir_pos and not dir_neg:
        return

    all_frags = []
    for a, b in dir_pos:
        all_frags.extend([a, b])
    for a, b in dir_neg:
        all_frags.extend([a, b])
    t_min = min(f['first_timestamp'] for f in all_frags)

    pos_by_lane = defaultdict(list)
    neg_by_lane = defaultdict(list)
    for a, b in dir_pos:
        pos_by_lane[get_lane(a)].append((a, b))
    for a, b in dir_neg:
        neg_by_lane[get_lane(a)].append((a, b))

    fig, axes = plt.subplots(2, N_LANES, figsize=(18, 8), sharex=True, sharey=True)
    fig.suptitle(f'Scenario {scenario.upper()} ({dir_label}) — GT-Masked Pairs (v3)',
                 fontsize=14, fontweight='bold')

    for lane in range(N_LANES):
        ax = axes[0, lane]
        for a, b in pos_by_lane[lane]:
            ta = np.array(a['timestamp']) - t_min
            xa = np.array(a['x_position'])
            tb = np.array(b['timestamp']) - t_min
            xb = np.array(b['x_position'])
            ax.plot(ta, xa, color='#6baed6', alpha=0.6, linewidth=0.8)
            ax.plot(tb, xb, color='#6baed6', alpha=0.6, linewidth=0.8)
            ax.plot([ta[-1], tb[0]], [xa[-1], xb[0]], color='#2171b5',
                    ls='--', alpha=0.3, lw=0.5)
        ax.set_ylim(0, 2000)
        ax.set_xlim(left=0)
        ax.set_title(f'Lane {lane + 1}', fontsize=12)
        if lane == 0:
            ax.set_ylabel('Positive\nX-Position (ft)', fontsize=11)

        ax = axes[1, lane]
        for a, b in neg_by_lane[lane]:
            ta = np.array(a['timestamp']) - t_min
            xa = np.array(a['x_position'])
            tb = np.array(b['timestamp']) - t_min
            xb = np.array(b['x_position'])
            ax.plot(ta, xa, color='#fc9272', alpha=0.6, linewidth=0.8)
            ax.plot(tb, xb, color='#fc9272', alpha=0.6, linewidth=0.8)
            ax.plot([ta[-1], tb[0]], [xa[-1], xb[0]], color='#de2d26',
                    ls='--', alpha=0.3, lw=0.5)
        ax.set_ylim(0, 2000)
        ax.set_xlim(left=0)
        if lane == 0:
            ax.set_ylabel('Negative\nX-Position (ft)', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Time-space saved: {output_path}")


# ── Main pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rebuild LR training dataset v3 using GT masking + augmentation"
    )
    parser.add_argument('--output', default='data/training_dataset_v3.npz')
    parser.add_argument('--max-y-diff', type=float, default=5.0,
                        help='Max y_diff for BOTH pos and neg pairs (ft)')
    parser.add_argument('--time-window', type=float, default=10.0,
                        help='Max time gap for negative pairs (s)')
    parser.add_argument('--upscale', type=int, default=3,
                        help='Number of augmented copies per base pair')
    parser.add_argument('--skip-diagnostics', action='store_true')
    parser.add_argument('--skip-timespace', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)

    print("=" * 70)
    print("REBUILDING DATASET v3 — GT Masking + Augmentation")
    print("=" * 70)
    print(f"Mask centers: {MASK_CENTERS}")
    print(f"Mask width: {MASK_WIDTH_MEAN} ± {MASK_WIDTH_STD} ft")
    print(f"Neg time window: ±{args.time_window}s")
    print(f"Neg max y_diff: {args.max_y_diff} ft (also applied to positives)")
    print(f"Upscale: {args.upscale}x augmented copies")
    print(f"Detection noise: {NOISE}")

    # ── Phase 1: Generate base pairs (unaugmented) ──
    print(f"\n{'='*50}")
    print("PHASE 1: Generate base pairs from GT masking")
    print(f"{'='*50}")

    all_base_pos = []
    all_base_neg = []

    for scenario in DATASETS:
        print(f"\nScenario: {scenario}")
        gt_path = PROJECT_ROOT / f"GT_{scenario}.json"
        if not gt_path.exists():
            print(f"  WARNING: {gt_path} not found, skipping")
            continue

        gt_data = load_json(gt_path)
        print(f"  Loaded {len(gt_data)} GT trajectories")

        pos = generate_positive_pairs(gt_data, MASK_CENTERS, MASK_WIDTH_MEAN,
                                      max_y_diff=args.max_y_diff)
        neg = generate_negative_pairs(pos, time_window=args.time_window,
                                      max_y_diff=args.max_y_diff)
        print(f"  Base positive pairs: {len(pos)}")
        print(f"  Base negative pairs: {len(neg)}")

        # Time-space diagram (using base/unaugmented pairs)
        if not args.skip_timespace:
            for direction in [1, -1]:
                dl = 'EB' if direction == 1 else 'WB'
                out = CANONICAL_OUTPUT_DIR / f"timespace_v3_{scenario}_{dl}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                generate_timespace(pos, neg, scenario, direction, out)

        all_base_pos.extend(pos)
        all_base_neg.extend(neg)

    n_base_pos = len(all_base_pos)
    n_base_neg = len(all_base_neg)
    print(f"\nBase totals: {n_base_pos} pos, {n_base_neg} neg")

    if n_base_pos == 0:
        print("ERROR: No positive pairs!")
        return

    # ── Phase 2: Augment + upscale ──
    print(f"\n{'='*50}")
    print(f"PHASE 2: Augment × {args.upscale} with detection noise")
    print(f"{'='*50}")

    # Balance base pairs first
    n_base = min(n_base_pos, n_base_neg)

    if n_base_pos > n_base:
        idx = rng.choice(n_base_pos, n_base, replace=False)
        base_pos = [all_base_pos[i] for i in idx]
    else:
        base_pos = all_base_pos

    if n_base_neg > n_base:
        idx = rng.choice(n_base_neg, n_base, replace=False)
        base_neg = [all_base_neg[i] for i in idx]
    else:
        base_neg = all_base_neg

    print(f"Balanced base: {len(base_pos)} pos, {len(base_neg)} neg")

    all_aug_pairs = []
    all_aug_labels = []

    for copy_idx in range(args.upscale):
        copy_rng = np.random.RandomState(args.seed + copy_idx + 1)

        # Randomize mask width for this copy (affects spatial_gap feature)
        mask_width = max(100, rng.normal(MASK_WIDTH_MEAN, MASK_WIDTH_STD))

        # Re-generate pairs with varied mask width for copies > 0
        if copy_idx > 0:
            copy_pos_all = []
            copy_neg_all = []
            for scenario in DATASETS:
                gt_path = PROJECT_ROOT / f"GT_{scenario}.json"
                if not gt_path.exists():
                    continue
                gt_data = load_json(gt_path)
                pos = generate_positive_pairs(gt_data, MASK_CENTERS, mask_width,
                                              max_y_diff=args.max_y_diff)
                neg = generate_negative_pairs(pos, time_window=args.time_window,
                                              max_y_diff=args.max_y_diff)
                copy_pos_all.extend(pos)
                copy_neg_all.extend(neg)

            # Balance this copy
            n_copy = min(len(copy_pos_all), len(copy_neg_all))
            if n_copy == 0:
                continue
            if len(copy_pos_all) > n_copy:
                idx = copy_rng.choice(len(copy_pos_all), n_copy, replace=False)
                copy_pos = [copy_pos_all[i] for i in idx]
            else:
                copy_pos = copy_pos_all
            if len(copy_neg_all) > n_copy:
                idx = copy_rng.choice(len(copy_neg_all), n_copy, replace=False)
                copy_neg = [copy_neg_all[i] for i in idx]
            else:
                copy_neg = copy_neg_all
        else:
            copy_pos = base_pos
            copy_neg = base_neg

        print(f"  Copy {copy_idx + 1}/{args.upscale}: mask_width={mask_width:.0f}ft, "
              f"{len(copy_pos)} pos, {len(copy_neg)} neg")

        # Augment positive pairs
        for frag_before, frag_after, gt_id, mask_idx in copy_pos:
            aug_a = augment_fragment(frag_before, copy_rng)
            aug_b = augment_fragment(frag_after, copy_rng)
            all_aug_pairs.append((aug_a, aug_b))
            all_aug_labels.append(1)

        # Augment negative pairs
        for frag_a, frag_b in copy_neg:
            aug_a = augment_fragment(frag_a, copy_rng)
            aug_b = augment_fragment(frag_b, copy_rng)
            all_aug_pairs.append((aug_a, aug_b))
            all_aug_labels.append(0)

    print(f"\nTotal augmented: {len(all_aug_pairs)} pairs "
          f"({sum(1 for l in all_aug_labels if l==1)} pos, "
          f"{sum(1 for l in all_aug_labels if l==0)} neg)")

    # ── Phase 3: Extract features ──
    print(f"\n{'='*50}")
    print("PHASE 3: Extract 47 features per pair")
    print(f"{'='*50}")

    X, y, feature_names = extract_features(all_aug_pairs, all_aug_labels)

    print(f"\nDataset shape: {X.shape}")
    print(f"Positives: {(y == 1).sum()}")
    print(f"Negatives: {(y == 0).sum()}")

    # ── Audit key features ──
    print(f"\n--- Feature audit ---")
    for name in ['y_diff', 'time_gap', 'spatial_gap', 'length_diff',
                 'width_diff', 'height_diff', 'vel_diff']:
        if name not in feature_names:
            continue
        idx = feature_names.index(name)
        pos_vals = X[y == 1, idx]
        neg_vals = X[y == 0, idx]
        print(f"  {name}:")
        print(f"    Pos: mean={pos_vals.mean():.2f}, std={pos_vals.std():.2f}, "
              f"range=[{pos_vals.min():.2f}, {pos_vals.max():.2f}]")
        print(f"    Neg: mean={neg_vals.mean():.2f}, std={neg_vals.std():.2f}, "
              f"range=[{neg_vals.min():.2f}, {neg_vals.max():.2f}]")

    # ── Save ──
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = LR_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y,
                        feature_names=np.array(feature_names))
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # ── Diagnostics ──
    if not args.skip_diagnostics:
        diag_path = CANONICAL_OUTPUT_DIR / "dataset_v3_diagnostics.png"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        generate_diagnostics(X, y, feature_names, diag_path)

    print("\n" + "=" * 70)
    print("DATASET v3 REBUILD COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
