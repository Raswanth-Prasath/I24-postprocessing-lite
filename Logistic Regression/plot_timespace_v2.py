"""
Generate trajectory time-space diagrams for Dataset v2 (positive and negative pairs).

Layout matches the reference style:
- Columns: Lane 1-4 (12 ft bins on y_position)
- Rows: Positive pairs (same vehicle), Negative pairs (different vehicles)
- Each trajectory fragment plotted as a line; paired fragments connected by dashed line

Usage:
    conda activate i24
    python "Logistic Regression/plot_timespace_v2.py"
"""

import json
import sys
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['i', 'ii', 'iii']
LANE_WIDTH = 12.0  # ft per lane
N_LANES = 4

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

siamese_dir = PROJECT_ROOT / "Siamese-Network"
if str(siamese_dir) not in sys.path:
    sys.path.insert(0, str(siamese_dir))


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_gt_id(fragment):
    if 'gt_ids' in fragment and len(fragment['gt_ids']) > 0:
        first = fragment['gt_ids'][0]
        if isinstance(first, list) and len(first) > 0:
            if isinstance(first[0], dict) and '$oid' in first[0]:
                return first[0]['$oid']
    if '_source_gt_id' in fragment:
        return fragment['_source_gt_id']
    return None


def ensure_fields(frag):
    if 'timestamp' in frag and len(frag['timestamp']) > 0:
        frag.setdefault('first_timestamp', frag['timestamp'][0])
        frag.setdefault('last_timestamp', frag['timestamp'][-1])
    if 'x_position' in frag and len(frag['x_position']) > 0:
        frag.setdefault('starting_x', frag['x_position'][0])
        frag.setdefault('ending_x', frag['x_position'][-1])
    return frag


def get_lane(frag, direction=1):
    """Assign lane index (0-based) from mean y_position."""
    y_mean = np.mean(frag['y_position'])
    if direction == 1:  # EB
        lane = int(y_mean / LANE_WIDTH)
    else:  # WB - offset by EB lanes
        lane = int((y_mean - 72) / LANE_WIDTH)
    return max(0, min(lane, N_LANES - 1))


def get_pair_lane(frag_a, frag_b, direction=1):
    """Get lane for a pair (average of both fragments)."""
    y_mean = (np.mean(frag_a['y_position']) + np.mean(frag_b['y_position'])) / 2
    if direction == 1:
        lane = int(y_mean / LANE_WIDTH)
    else:
        lane = int((y_mean - 72) / LANE_WIDTH)
    return max(0, min(lane, N_LANES - 1))


def generate_raw_positives(raw_fragments, max_y_diff=5.0, max_time_gap=5.0):
    vehicle_fragments = defaultdict(list)
    for frag in raw_fragments:
        gt_id = get_gt_id(frag)
        if gt_id is not None:
            vehicle_fragments[gt_id].append(frag)

    pairs = []
    for gt_id, frags in vehicle_fragments.items():
        frags.sort(key=lambda f: f.get('first_timestamp', f['timestamp'][0]))
        for i in range(len(frags)):
            for j in range(i + 1, len(frags)):
                a, b = frags[i], frags[j]
                if (a['last_timestamp'] < b['first_timestamp'] and
                    b['first_timestamp'] - a['last_timestamp'] <= max_time_gap and
                    a.get('direction') == b.get('direction') and
                    abs(np.mean(b['y_position']) - np.mean(a['y_position'])) <= max_y_diff):
                    pairs.append((a, b))
    return pairs


def generate_raw_negatives(raw_fragments, num_negatives, max_y_diff=5.0, max_time_gap=5.0):
    lane_bin_size = 6.0
    lane_groups = defaultdict(list)
    for frag in raw_fragments:
        gt_id = get_gt_id(frag)
        if gt_id is None:
            continue
        direction = frag.get('direction', 1)
        y_mean = np.mean(frag['y_position'])
        lane_bin = int(y_mean / lane_bin_size)
        lane_groups[(direction, lane_bin)].append((gt_id, frag))

    pairs = []
    for (direction, lane_bin), fragments in lane_groups.items():
        if len(fragments) < 2:
            continue
        fragments.sort(key=lambda x: x[1]['first_timestamp'])
        for i in range(len(fragments)):
            if len(pairs) >= num_negatives:
                break
            gt_id_a, frag_a = fragments[i]
            for j in range(i + 1, min(i + 30, len(fragments))):
                gt_id_b, frag_b = fragments[j]
                if gt_id_a == gt_id_b:
                    continue
                if frag_a['last_timestamp'] >= frag_b['first_timestamp']:
                    continue
                gap = frag_b['first_timestamp'] - frag_a['last_timestamp']
                if gap > max_time_gap or gap < 0:
                    continue
                y_diff = abs(np.mean(frag_b['y_position']) - np.mean(frag_a['y_position']))
                if y_diff > max_y_diff:
                    continue
                pairs.append((frag_a, frag_b))
                if len(pairs) >= num_negatives:
                    break

    random.shuffle(pairs)
    return pairs[:num_negatives]


def plot_lane_panel(ax, pairs, t_min, direction=1, color='#6baed6', link_color=None,
                    alpha=0.6, lw=0.8):
    """Plot all trajectory pairs in a single lane panel, matching reference style."""
    for frag_a, frag_b in pairs:
        t_a = np.array(frag_a['timestamp']) - t_min
        x_a = np.array(frag_a['x_position'])
        t_b = np.array(frag_b['timestamp']) - t_min
        x_b = np.array(frag_b['x_position'])

        ax.plot(t_a, x_a, color=color, alpha=alpha, linewidth=lw)
        ax.plot(t_b, x_b, color=color, alpha=alpha, linewidth=lw)

        if link_color:
            ax.plot([t_a[-1], t_b[0]], [x_a[-1], x_b[0]],
                    color=link_color, linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_ylim(0, 2000)
    ax.set_xlim(left=0)


def main():
    random.seed(42)
    np.random.seed(42)

    for scenario in DATASETS:
        print(f"\n{'='*60}")
        print(f"Generating time-space diagram for scenario {scenario}")
        print(f"{'='*60}")

        raw_path = PROJECT_ROOT / f"RAW_{scenario}.json"
        if not raw_path.exists():
            raw_path = PROJECT_ROOT / f"RAW_{scenario}_Bhat.json"
        if not raw_path.exists():
            print(f"  Skipping: no RAW file found")
            continue

        raw_data = load_json(raw_path)
        for frag in raw_data:
            ensure_fields(frag)

        # Separate by direction
        for direction, dir_label in [(1, 'EB'), (-1, 'WB')]:
            dir_frags = [f for f in raw_data if f.get('direction', 1) == direction]
            if not dir_frags:
                continue

            print(f"\n  Direction: {dir_label} ({len(dir_frags)} fragments)")

            # Global t_min for this scenario+direction
            t_min = min(f['first_timestamp'] for f in dir_frags)

            # Generate pairs
            pos_pairs = generate_raw_positives(dir_frags)
            neg_pairs = generate_raw_negatives(dir_frags, num_negatives=len(pos_pairs) * 2)
            print(f"    Positive pairs: {len(pos_pairs)}")
            print(f"    Negative pairs: {len(neg_pairs)}")

            # Bin pairs by lane
            pos_by_lane = defaultdict(list)
            neg_by_lane = defaultdict(list)
            for a, b in pos_pairs:
                lane = get_pair_lane(a, b, direction)
                pos_by_lane[lane].append((a, b))
            for a, b in neg_pairs:
                lane = get_pair_lane(a, b, direction)
                neg_by_lane[lane].append((a, b))

            for lane in range(N_LANES):
                print(f"    Lane {lane+1}: {len(pos_by_lane[lane])} pos, {len(neg_by_lane[lane])} neg")

            # ── Create figure matching reference style ──
            fig, axes = plt.subplots(2, N_LANES, figsize=(18, 8),
                                     sharex=True, sharey=True)
            fig.suptitle(f'Scenario {scenario.upper()} ({dir_label}) — Dataset v2 Pairs',
                         fontsize=14, fontweight='bold')

            row_labels = ['Positive\nX-Position (ft)', 'Negative\nX-Position (ft)']

            for lane in range(N_LANES):
                # Top row: positive pairs
                ax = axes[0, lane]
                plot_lane_panel(ax, pos_by_lane[lane], t_min, direction,
                                color='#6baed6', link_color='#2171b5')
                if lane == 0:
                    ax.set_ylabel(row_labels[0], fontsize=11)
                ax.set_title(f'Lane {lane + 1}', fontsize=12)

                # Bottom row: negative pairs
                ax = axes[1, lane]
                plot_lane_panel(ax, neg_by_lane[lane], t_min, direction,
                                color='#fc9272', link_color='#de2d26')
                if lane == 0:
                    ax.set_ylabel(row_labels[1], fontsize=11)
                ax.set_xlabel('Time (s)', fontsize=11)

            plt.tight_layout()
            out_path = (Path(__file__).resolve().parent /
                        f'timespace_v2_{scenario}_{dir_label}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {out_path}")


if __name__ == '__main__':
    main()
