#!/usr/bin/env python3
"""
Plot IEEE-style time-space diagrams comparing RAW, GT, and REC per direction.

Defaults:
- Data files in repo root: GT_i.json, RAW_i.json, REC_i.json (and ii/iii)
- Saves PNGs under outputs/time_space/
  Example: timespace_i_EB.png, timespace_i_WB.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def configure_ieee_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 0.9,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _iter_tracks(data):
    if not isinstance(data, list):
        return
    for item in data:
        if not isinstance(item, dict):
            continue
        direction = item.get("direction")
        if direction not in (1, -1):
            continue
        yield item


def calculate_lane_bounds(datasets):
    eb_y_positions = []
    wb_y_positions = []
    all_timestamps = []

    for data in datasets:
        for item in _iter_tracks(data):
            y_positions = item.get("y_position", [])
            timestamps = item.get("timestamp", [])
            if not y_positions or not timestamps:
                continue
            all_timestamps.extend(timestamps)
            if item["direction"] == 1:
                eb_y_positions.extend(y_positions)
            else:
                wb_y_positions.extend(y_positions)

    min_y_eb = min(eb_y_positions) if eb_y_positions else 0
    max_y_eb = max(eb_y_positions) if eb_y_positions else 12
    min_y_wb = min(wb_y_positions) if wb_y_positions else -12
    max_y_wb = max(wb_y_positions) if wb_y_positions else 0

    num_lanes_eb = int(np.ceil((max_y_eb - min_y_eb) / 12))
    num_lanes_wb = int(np.ceil((max_y_wb - min_y_wb) / 12))

    eb_lane_bounds = [min_y_eb + i * 12 for i in range(num_lanes_eb + 1)]
    wb_lane_bounds = [max_y_wb - i * 12 for i in range(num_lanes_wb + 1)][::-1]

    min_timestamp = min(all_timestamps) if all_timestamps else 0

    return eb_lane_bounds, wb_lane_bounds, num_lanes_eb, num_lanes_wb, min_timestamp


def collect_trajectories_by_lane(data, direction, lane_bounds, min_timestamp):
    lane_count = max(len(lane_bounds) - 1, 0)
    lanes = {i + 1: [] for i in range(lane_count)}

    for item in _iter_tracks(data):
        if item.get("direction") != direction:
            continue
        y_positions = item.get("y_position", [])
        x_positions = item.get("x_position", [])
        timestamps = item.get("timestamp", [])
        if not y_positions or not x_positions or not timestamps:
            continue

        y_avg = float(np.mean(y_positions))
        lane = None
        for i in range(len(lane_bounds) - 1):
            if lane_bounds[i] <= y_avg < lane_bounds[i + 1]:
                lane = i + 1
                break
        if lane is None:
            lane = len(lane_bounds) - 1

        if lane < 1 or lane > lane_count:
            continue

        normalized_timestamps = [t - min_timestamp for t in timestamps]
        lanes[lane].append((normalized_timestamps, x_positions))

    return lanes


def plot_direction_comparison(
    scenario,
    direction,
    raw_data,
    gt_data,
    rec_data,
    eb_lane_bounds,
    wb_lane_bounds,
    num_lanes_eb,
    num_lanes_wb,
    min_timestamp,
    out_path: Path,
    show: bool = False,
):
    direction_label = "EB" if direction == 1 else "WB"
    lane_bounds = eb_lane_bounds if direction == 1 else wb_lane_bounds
    lane_count = num_lanes_eb if direction == 1 else num_lanes_wb

    if lane_count <= 0:
        print(f"Skipping scenario {scenario} {direction_label}: no lanes detected")
        return

    rows = [
        ("RAW", raw_data),
        ("GT", gt_data),
        ("REC", rec_data),
    ]

    width = max(2.7 * lane_count, 6.0)
    height = 7.2
    # Build per-row lane data first so we can drop empty lanes (e.g., lane 5 in iii)
    row_lanes = []
    non_empty_lanes = set()
    for row_label, data in rows:
        lanes = collect_trajectories_by_lane(data, direction, lane_bounds, min_timestamp)
        row_lanes.append((row_label, lanes))
        for lane_num, trajectories in lanes.items():
            if trajectories:
                non_empty_lanes.add(lane_num)

    lane_indices = sorted(non_empty_lanes)
    if not lane_indices:
        print(f"Skipping scenario {scenario} {direction_label}: no trajectories")
        return

    lane_count = len(lane_indices)
    width = max(2.7 * lane_count, 6.0)
    fig, axs = plt.subplots(3, lane_count, figsize=(width, height), sharex="all", sharey="row")
    axs = np.array(axs).reshape(3, lane_count)

    line_color = "#1f77b4"
    line_alpha = 0.6
    line_style = "-"
    line_width = 0.8

    for row_idx, (row_label, lanes) in enumerate(row_lanes):
        for col_idx, lane_num in enumerate(lane_indices):
            ax = axs[row_idx, col_idx]
            trajectories = lanes.get(lane_num, [])

            for traj_idx, (timestamps, x_positions) in enumerate(trajectories):
                ax.plot(
                    timestamps,
                    x_positions,
                    linestyle=line_style,
                    color=line_color,
                    alpha=line_alpha,
                    linewidth=line_width,
                )

            if row_idx == 0:
                ax.set_title(f"Lane {lane_num}")

            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nX-Position (ft)")

            if row_idx == 2:
                ax.set_xlabel("Time (s)")

            ax.grid(True, linestyle=":", alpha=0.3)
            ax.tick_params(axis="both", which="major", length=3)

    scenario_label = scenario.upper()
    fig.suptitle(f"Scenario {scenario_label} ({direction_label})", y=0.98)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.08, wspace=0.25, hspace=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot RAW/REC/GT time-space diagrams for scenarios.")
    parser.add_argument("--data-dir", default=".", help="Directory containing GT/RAW/REC JSON files")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["i", "ii", "iii"],
        help="Scenarios to plot (e.g., i ii iii)",
    )
    parser.add_argument("--out-dir", default="outputs/time_space", help="Output directory for PNGs")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    configure_ieee_style()

    for scenario in args.scenarios:
        gt_path = data_dir / f"GT_{scenario}.json"
        raw_path = data_dir / f"RAW_{scenario}.json"
        rec_path = data_dir / f"REC_{scenario}.json"

        if not gt_path.exists() or not raw_path.exists() or not rec_path.exists():
            print(f"Skipping scenario {scenario}: missing one of {gt_path.name}, {raw_path.name}, {rec_path.name}")
            continue

        gt_data = load_json(gt_path)
        raw_data = load_json(raw_path)
        rec_data = load_json(rec_path)

        eb_bounds, wb_bounds, n_eb, n_wb, min_ts = calculate_lane_bounds(
            [gt_data, raw_data, rec_data]
        )

        print(f"\nScenario {scenario}:")
        print(f"  GT:  {len(gt_data)}")
        print(f"  RAW: {len(raw_data)}")
        print(f"  REC: {len(rec_data)}")

        for direction in (1, -1):
            direction_label = "EB" if direction == 1 else "WB"
            plot_direction_comparison(
                scenario,
                direction,
                raw_data,
                gt_data,
                rec_data,
                eb_bounds,
                wb_bounds,
                n_eb,
                n_wb,
                min_ts,
                out_dir / f"timespace_{scenario}_{direction_label}.png",
                show=args.show,
            )


if __name__ == "__main__":
    main()
