#!/usr/bin/env python3
"""
Dataset v4 Diagnostics — Comprehensive feature overlap analysis + time-space diagrams.

Loads training_dataset_v3.npz and produces:
  1. All-47-feature overlap table (printed + CSV)
  2. Full histogram grid (dataset_v4_diagnostics.png)
  3. Per-scenario time-space diagrams from GT masking (timespace_v4_*.png)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Import pair generation from rebuild_dataset (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_dataset import generate_positive_pairs, generate_negative_pairs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LR_DIR = Path(__file__).resolve().parent


# ─── Overlap metric ───────────────────────────────────────────────────
def histogram_overlap(pos_vals, neg_vals, n_bins=200):
    """Compute histogram overlap coefficient (0 = no overlap, 1 = identical)."""
    all_vals = np.concatenate([pos_vals, neg_vals])
    lo, hi = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
    if lo == hi:
        return 1.0  # constant feature
    bins = np.linspace(lo, hi, n_bins + 1)
    h_pos, _ = np.histogram(pos_vals, bins=bins, density=True)
    h_neg, _ = np.histogram(neg_vals, bins=bins, density=True)
    # Normalize to sum to 1
    h_pos = h_pos / (h_pos.sum() + 1e-12)
    h_neg = h_neg / (h_neg.sum() + 1e-12)
    overlap = np.minimum(h_pos, h_neg).sum()
    return overlap


def cohen_d(pos_vals, neg_vals):
    """Effect size: Cohen's d."""
    n1, n2 = len(pos_vals), len(neg_vals)
    m1, m2 = pos_vals.mean(), neg_vals.mean()
    s1, s2 = pos_vals.std(), neg_vals.std()
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return abs(m1 - m2) / pooled_std


# ─── Feature analysis ─────────────────────────────────────────────────
def analyze_features(X, y, feature_names):
    """Compute overlap and effect size for all features."""
    results = []
    for i, name in enumerate(feature_names):
        pos = X[y == 1, i]
        neg = X[y == 0, i]
        ovlp = histogram_overlap(pos, neg)
        d = cohen_d(pos, neg)
        results.append({
            'feature': name,
            'overlap': ovlp,
            'cohen_d': d,
            'pos_mean': pos.mean(),
            'pos_std': pos.std(),
            'neg_mean': neg.mean(),
            'neg_std': neg.std(),
            'pos_zero_frac': (np.abs(pos) < 1e-6).mean(),
            'neg_zero_frac': (np.abs(neg) < 1e-6).mean(),
        })
    return sorted(results, key=lambda r: r['overlap'])


# ─── Full histogram grid ──────────────────────────────────────────────
def plot_all_features(X, y, feature_names, output_path):
    """Plot histograms for ALL features in a grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(feature_names)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    fig.suptitle('Dataset v3 — All 47 Features (Pos=green, Neg=red)', fontsize=14, fontweight='bold', y=1.0)
    axes = axes.flatten()

    for i, name in enumerate(feature_names):
        ax = axes[i]
        pos_vals = X[y == 1, i]
        neg_vals = X[y == 0, i]

        all_vals = np.concatenate([pos_vals, neg_vals])
        lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        if lo == hi:
            lo -= 1
            hi += 1

        ax.hist(np.clip(pos_vals, lo, hi), bins=50, alpha=0.6, color='green', density=True, label='Pos')
        ax.hist(np.clip(neg_vals, lo, hi), bins=50, alpha=0.6, color='red', density=True, label='Neg')

        ovlp = histogram_overlap(pos_vals, neg_vals)
        ax.set_title(f'{name}\novlp={ovlp:.1%}', fontsize=8)
        ax.tick_params(labelsize=6)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Full histogram grid saved: {output_path}")


# ─── Shortcut risk summary plot ───────────────────────────────────────
def plot_overlap_barplot(results, output_path):
    """Horizontal bar plot of feature overlap, color-coded by risk."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [r['feature'] for r in results]
    overlaps = [r['overlap'] for r in results]

    colors = []
    for o in overlaps:
        if o < 0.3:
            colors.append('#d32f2f')   # red — high shortcut risk
        elif o < 0.5:
            colors.append('#ff9800')   # orange — moderate risk
        elif o < 0.7:
            colors.append('#fdd835')   # yellow — mild
        else:
            colors.append('#4caf50')   # green — safe

    fig, ax = plt.subplots(figsize=(10, max(8, len(names) * 0.3)))
    ax.barh(range(len(names)), overlaps, color=colors, edgecolor='none')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Histogram Overlap (0=separable, 1=identical)')
    ax.set_title('Feature Overlap: Shortcut Risk Assessment\n'
                 'Red=high risk (<30%), Orange=moderate (<50%), Yellow=mild (<70%), Green=safe (>70%)')
    ax.axvline(0.3, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(0.7, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overlap barplot saved: {output_path}")


# ─── Time-space diagrams (v3-style: lane-separated, pos/neg rows) ────
MASK_CENTERS = [225, 625, 1125, 1625]
MASK_WIDTH = 250

DATASETS = ['i', 'ii', 'iii']
LANE_WIDTH = 12.0
N_LANES = 4


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _get_lane(frag, direction):
    y_mean = np.mean(frag['y_position'])
    if direction == 1:
        return max(0, min(int(y_mean / LANE_WIDTH), N_LANES - 1))
    else:
        return max(0, min(int((y_mean - 72) / LANE_WIDTH), N_LANES - 1))


def plot_timespace_v4(pos_pairs, neg_pairs, scenario, direction, output_path):
    """Lane-separated time-space diagram with pos (top, blue) and neg (bottom, red) rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    dir_label = 'EB' if direction == 1 else 'WB'

    # Filter by direction — positive tuples are (before, after, gt_id, mask_idx)
    dir_pos = [(a, b) for a, b, _, _ in pos_pairs if a['direction'] == direction]
    dir_neg = [(a, b) for a, b in neg_pairs if a['direction'] == direction]

    if not dir_pos and not dir_neg:
        return

    # Find global t_min for x-axis offset
    all_frags = []
    for a, b in dir_pos:
        all_frags.extend([a, b])
    for a, b in dir_neg:
        all_frags.extend([a, b])
    t_min = min(f['first_timestamp'] for f in all_frags)

    # Bin by lane
    pos_by_lane = defaultdict(list)
    neg_by_lane = defaultdict(list)
    for a, b in dir_pos:
        pos_by_lane[_get_lane(a, direction)].append((a, b))
    for a, b in dir_neg:
        neg_by_lane[_get_lane(a, direction)].append((a, b))

    fig, axes = plt.subplots(2, N_LANES, figsize=(18, 8), sharex=True, sharey=True)
    fig.suptitle(f'Scenario {scenario.upper()} ({dir_label}) — GT-Masked Pairs (v4)\n'
                 f'Pos: {len(dir_pos)} pairs (blue, top)  |  Neg: {len(dir_neg)} pairs (red, bottom)',
                 fontsize=13, fontweight='bold')

    for lane in range(N_LANES):
        # Positive row (top)
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

        # Negative row (bottom)
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


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Dataset v4 Diagnostics')
    parser.add_argument('--dataset', default='training_dataset_v3.npz',
                        help='Dataset file to analyze')
    parser.add_argument('--skip-timespace', action='store_true',
                        help='Skip time-space diagram generation')
    args = parser.parse_args()

    dataset_path = LR_DIR / args.dataset
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    print("=" * 70)
    print("DATASET v4 DIAGNOSTICS")
    print("=" * 70)

    # Load
    data = np.load(dataset_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    feature_names = list(data['feature_names'])

    print(f"Dataset: {dataset_path.name}")
    print(f"Shape: {X.shape}")
    print(f"Positive: {(y == 1).sum()}, Negative: {(y == 0).sum()}")
    print(f"Features: {len(feature_names)}")

    # ── Feature overlap analysis ──
    print(f"\n{'=' * 70}")
    print("FEATURE OVERLAP ANALYSIS (sorted by overlap, low = shortcut risk)")
    print(f"{'=' * 70}")
    print(f"{'Feature':<30} {'Overlap':>8} {'Cohen d':>8} {'Pos Mean':>10} {'Neg Mean':>10} {'Risk':<12}")
    print("-" * 80)

    results = analyze_features(X, y, feature_names)

    n_high_risk = 0
    n_moderate = 0
    for r in results:
        ovlp = r['overlap']
        if ovlp < 0.3:
            risk = "HIGH"
            n_high_risk += 1
        elif ovlp < 0.5:
            risk = "MODERATE"
            n_moderate += 1
        elif ovlp < 0.7:
            risk = "mild"
        else:
            risk = "safe"

        print(f"  {r['feature']:<28} {ovlp:>7.1%} {r['cohen_d']:>8.2f} "
              f"{r['pos_mean']:>10.3f} {r['neg_mean']:>10.3f} {risk:<12}")

    print(f"\nSummary: {n_high_risk} HIGH risk, {n_moderate} MODERATE risk, "
          f"{len(results) - n_high_risk - n_moderate} safe features")

    # ── Plots ──
    print(f"\n{'=' * 70}")
    print("GENERATING PLOTS")
    print(f"{'=' * 70}")

    plot_all_features(X, y, feature_names,
                      LR_DIR / 'dataset_v4_diagnostics.png')
    plot_overlap_barplot(results,
                        LR_DIR / 'dataset_v4_overlap_barplot.png')

    # ── Time-space diagrams (regenerate pairs from GT) ──
    if not args.skip_timespace:
        print(f"\n{'=' * 70}")
        print("TIME-SPACE DIAGRAMS (lane-separated, pos/neg pairs)")
        print(f"{'=' * 70}")

        for scenario in DATASETS:
            gt_path = PROJECT_ROOT / f'GT_{scenario}.json'
            if not gt_path.exists():
                print(f"  SKIP scenario {scenario}: {gt_path} not found")
                continue
            print(f"\nScenario {scenario}:")
            gt_data = load_json(gt_path)
            pos = generate_positive_pairs(gt_data, MASK_CENTERS, MASK_WIDTH,
                                          max_y_diff=5.0)
            neg = generate_negative_pairs(pos, time_window=10.0, max_y_diff=5.0)
            print(f"  Generated {len(pos)} pos, {len(neg)} neg pairs")
            for direction in [1, -1]:
                dl = 'EB' if direction == 1 else 'WB'
                out = LR_DIR / f'timespace_v4_{scenario}_{dl}.png'
                plot_timespace_v4(pos, neg, scenario, direction, out)

    print(f"\n{'=' * 70}")
    print("DIAGNOSTICS v4 COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
