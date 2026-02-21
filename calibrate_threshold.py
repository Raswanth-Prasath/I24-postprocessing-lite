#!/usr/bin/env python3
"""
Automatic stitch threshold calibration via ROC analysis on GT-labeled fragment pairs.

Strategies:
    fpr_ceiling     – (default) maximize TPR subject to FPR ≤ --max-fpr (0.5%)
    precision_floor – maximize recall subject to precision ≥ --min-precision (0.99)
    f_beta          – maximize F_β score (β < 1 heavily penalises false positives)
    youden          – maximize TPR − FPR (original, often too permissive for MCF)

Usage:
    python calibrate_threshold.py --config parameters_PINN.json
    python calibrate_threshold.py --config parameters_PINN.json --strategy fpr_ceiling --max-fpr 0.01
    python calibrate_threshold.py --config parameters_PINN.json --strategy precision_floor --min-precision 0.995
    python calibrate_threshold.py --config parameters_PINN.json --strategy youden
"""

import argparse
import json
import os
import sys
import time

import ijson
import numpy as np
from sklearn.metrics import roc_curve

import utils.misc as misc
from utils.stitch_cost_interface import CostFunctionFactory


def load_fragments(raw_path: str) -> list:
    """Load and preprocess fragments from a RAW JSON file via ijson streaming."""
    fragments = []
    discarded = 0

    with open(raw_path, "rb") as f:
        for doc in ijson.items(f, "item"):
            if len(doc["timestamp"]) <= 3:
                discarded += 1
                continue

            # Convert decimal types to float
            doc["timestamp"] = list(map(float, doc["timestamp"]))
            doc["x_position"] = list(map(float, doc["x_position"]))
            doc["y_position"] = list(map(float, doc["y_position"]))
            doc["width"] = list(map(float, doc["width"]))
            doc["length"] = list(map(float, doc["length"]))
            doc["height"] = list(map(float, doc["height"]))
            doc["velocity"] = list(map(float, doc["velocity"]))
            doc["detection_confidence"] = list(map(float, doc["detection_confidence"]))

            doc["first_timestamp"] = float(doc["first_timestamp"])
            doc["last_timestamp"] = float(doc["last_timestamp"])
            doc["starting_x"] = float(doc["starting_x"])
            doc["ending_x"] = float(doc["ending_x"])
            doc["_id"] = doc["_id"]["$oid"]
            doc["compute_node_id"] = 1

            doc = misc.interpolate(doc)
            fragments.append(doc)

    print(f"  Loaded {len(fragments)} fragments ({discarded} discarded short)")
    return fragments


def get_gt_id(fragment: dict) -> str:
    """Extract the primary ground truth ID from a fragment, or None."""
    gt_ids = fragment.get("gt_ids")
    if not gt_ids:
        return None
    first = gt_ids[0]
    if isinstance(first, list):
        if not first:
            return None
        first = first[0]
    if isinstance(first, dict) and "$oid" in first:
        return first["$oid"]
    return str(first)


def find_raw_path(scenario: str) -> str:
    """Locate the RAW file for a scenario, handling both naming conventions."""
    candidates = [
        f"RAW_{scenario}.json",
        f"RAW_{scenario}_Bhat.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"No RAW file found for scenario {scenario}. "
        f"Tried: {candidates}"
    )


def generate_pairs(fragments: list, time_win: float, max_x_gap: float = 800.0):
    """
    Generate candidate fragment pairs and label them using gt_ids.

    Yields (frag_a, frag_b, label) where label=1 means same vehicle.
    """
    # Sort by last_timestamp for efficient sequential pairing
    sorted_frags = sorted(fragments, key=lambda f: f["last_timestamp"])

    for i, frag_a in enumerate(sorted_frags):
        gt_a = get_gt_id(frag_a)
        if gt_a is None:
            continue

        dir_a = frag_a.get("direction", 1)
        end_t = frag_a["last_timestamp"]

        for j in range(i + 1, len(sorted_frags)):
            frag_b = sorted_frags[j]

            # frag_b must start after frag_a ends
            start_t = frag_b["first_timestamp"]
            gap = start_t - end_t
            if gap < 0:
                continue
            if gap > time_win:
                break  # all subsequent fragments are further away

            # Same direction
            dir_b = frag_b.get("direction", 1)
            if dir_a != dir_b:
                continue

            # Reasonable spatial proximity
            x_gap = abs(frag_b["starting_x"] - frag_a["ending_x"])
            if x_gap > max_x_gap:
                continue

            gt_b = get_gt_id(frag_b)
            if gt_b is None:
                continue

            label = 1 if gt_a == gt_b else 0
            yield frag_a, frag_b, label


def select_threshold(
    fpr, tpr, actual_thresholds, n_pos, n_neg,
    strategy="fpr_ceiling", max_fpr=0.005, min_precision=0.99, f_beta=0.1,
):
    """Select optimal threshold using the specified strategy.

    Strategies:
        youden       – maximize TPR − FPR (original, permissive)
        fpr_ceiling  – maximize TPR subject to FPR ≤ max_fpr
        precision_floor – maximize recall subject to precision ≥ min_precision
        f_beta       – maximize F_β score (β < 1 penalises FP heavily)

    Returns (threshold, tpr_at_thresh, fpr_at_thresh, strategy_name).
    """
    if strategy == "youden":
        j_scores = tpr - fpr
        idx = np.argmax(j_scores)

    elif strategy == "fpr_ceiling":
        mask = fpr <= max_fpr
        if not np.any(mask):
            # fall back to lowest FPR available
            idx = np.argmin(fpr[fpr > 0]) if np.any(fpr > 0) else 0
        else:
            # among points with FPR ≤ ceiling, pick highest TPR
            candidates = np.where(mask)[0]
            idx = candidates[np.argmax(tpr[candidates])]

    elif strategy == "precision_floor":
        # precision = TP / (TP + FP) ≈ (TPR * n_pos) / (TPR * n_pos + FPR * n_neg)
        with np.errstate(divide='ignore', invalid='ignore'):
            tp_est = tpr * n_pos
            fp_est = fpr * n_neg
            precision = np.where(
                (tp_est + fp_est) > 0,
                tp_est / (tp_est + fp_est),
                0.0,
            )
        mask = precision >= min_precision
        if not np.any(mask):
            idx = np.argmax(precision)
        else:
            candidates = np.where(mask)[0]
            idx = candidates[np.argmax(tpr[candidates])]

    elif strategy == "f_beta":
        beta2 = f_beta ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            tp_est = tpr * n_pos
            fp_est = fpr * n_neg
            precision = np.where(
                (tp_est + fp_est) > 0,
                tp_est / (tp_est + fp_est),
                0.0,
            )
            recall = tpr
            fb = np.where(
                (precision + recall) > 0,
                (1 + beta2) * precision * recall / (beta2 * precision + recall),
                0.0,
            )
        idx = np.argmax(fb)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return (
        float(actual_thresholds[idx]),
        float(tpr[idx]),
        float(fpr[idx]),
        strategy,
    )


def calibrate_scenario(
    scenario: str, cost_fn, time_win: float, param: dict,
    strategy: str = "fpr_ceiling", max_fpr: float = 0.005,
    min_precision: float = 0.99, f_beta: float = 0.1,
) -> dict:
    """Run calibration for a single scenario. Returns results dict or None."""
    print(f"\n--- Scenario {scenario} ---")

    try:
        raw_path = find_raw_path(scenario)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    print(f"  Loading from {raw_path}...")
    fragments = load_fragments(raw_path)

    print("  Generating pairs...")
    costs = []
    labels = []
    errors = 0
    t0 = time.time()

    for frag_a, frag_b, label in generate_pairs(fragments, time_win):
        cost = cost_fn.compute_cost(frag_a, frag_b, time_win, param)
        if cost >= 1e5:  # skip invalid pairs
            errors += 1
            continue
        costs.append(cost)
        labels.append(label)

    elapsed = time.time() - t0
    costs = np.array(costs)
    labels = np.array(labels)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    print(f"  Pairs: {len(labels)} (pos={n_pos}, neg={n_neg}), "
          f"skipped={errors}, time={elapsed:.1f}s")

    if n_pos == 0 or n_neg == 0:
        print("  SKIP: need both positive and negative pairs for ROC")
        return None

    # ROC analysis — note: lower cost = positive (same vehicle),
    # so we negate costs for sklearn's roc_curve which expects higher = positive
    fpr, tpr, thresholds = roc_curve(labels, -costs)
    # Thresholds from roc_curve on negated scores: actual_thresh = -thresholds
    actual_thresholds = -thresholds

    # Select threshold using chosen strategy
    optimal_thresh, best_tpr, best_fpr, strat_used = select_threshold(
        fpr, tpr, actual_thresholds, n_pos, n_neg,
        strategy=strategy, max_fpr=max_fpr,
        min_precision=min_precision, f_beta=f_beta,
    )

    # Also compute Youden for comparison when using a different strategy
    youden_thresh = None
    if strategy != "youden":
        j_scores = tpr - fpr
        youden_idx = np.argmax(j_scores)
        youden_thresh = float(actual_thresholds[youden_idx])

    # Cost distribution stats
    pos_costs = costs[labels == 1]
    neg_costs = costs[labels == 0]

    # Margin: gap between max positive cost and min negative cost at threshold
    margin = float(neg_costs[neg_costs > optimal_thresh].min() - optimal_thresh) \
        if np.any(neg_costs > optimal_thresh) else 0.0
    margin_below = float(optimal_thresh - pos_costs[pos_costs < optimal_thresh].max()) \
        if np.any(pos_costs < optimal_thresh) else 0.0
    margin = min(margin, margin_below)

    # Expected false edges at this threshold
    expected_false_edges = int(round(best_fpr * n_neg))
    neg_pos_ratio = n_neg / n_pos if n_pos > 0 else float('inf')

    result = {
        "scenario": scenario,
        "strategy": strat_used,
        "optimal_thresh": float(optimal_thresh),
        "margin": float(margin),
        "tpr": float(best_tpr),
        "fpr": float(best_fpr),
        "expected_false_edges": expected_false_edges,
        "neg_pos_ratio": float(neg_pos_ratio),
        "n_pairs": len(labels),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_cost_mean": float(pos_costs.mean()),
        "pos_cost_std": float(pos_costs.std()),
        "pos_cost_max": float(pos_costs.max()),
        "neg_cost_mean": float(neg_costs.mean()),
        "neg_cost_std": float(neg_costs.std()),
        "neg_cost_min": float(neg_costs.min()),
    }
    if youden_thresh is not None:
        result["youden_thresh"] = youden_thresh

    print(f"  Strategy: {strat_used}")
    print(f"  Optimal threshold: {optimal_thresh:.3f} "
          f"(TPR={best_tpr:.3f}, FPR={best_fpr:.3f})")
    print(f"  Expected false edges: {expected_false_edges} "
          f"(FPR={best_fpr:.4f} × {n_neg} neg)")
    print(f"  Neg:Pos ratio: {neg_pos_ratio:.1f}:1")
    print(f"  Margin: {margin:.3f}")
    print(f"  Positive costs: mean={pos_costs.mean():.3f}, "
          f"std={pos_costs.std():.3f}, max={pos_costs.max():.3f}")
    print(f"  Negative costs: mean={neg_costs.mean():.3f}, "
          f"std={neg_costs.std():.3f}, min={neg_costs.min():.3f}")
    if youden_thresh is not None:
        print(f"  (Youden J threshold would be: {youden_thresh:.3f})")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate stitch threshold via ROC analysis on GT-labeled pairs"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to parameters JSON config file"
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["i", "ii", "iii"],
        help="Scenarios to calibrate on (default: i ii iii)"
    )
    parser.add_argument(
        "--time-win", type=float, default=None,
        help="Override time window from config (default: use config value)"
    )
    parser.add_argument(
        "--max-x-gap", type=float, default=800.0,
        help="Maximum x-position gap for candidate pairs (default: 800 ft)"
    )
    parser.add_argument(
        "--strategy", default="fpr_ceiling",
        choices=["youden", "fpr_ceiling", "precision_floor", "f_beta"],
        help="Threshold selection strategy (default: fpr_ceiling)"
    )
    parser.add_argument(
        "--max-fpr", type=float, default=0.005,
        help="Max FPR for fpr_ceiling strategy (default: 0.005 = 0.5%%)"
    )
    parser.add_argument(
        "--min-precision", type=float, default=0.99,
        help="Min precision for precision_floor strategy (default: 0.99)"
    )
    parser.add_argument(
        "--f-beta", type=float, default=0.1,
        help="Beta for f_beta strategy; <1 penalises FP (default: 0.1)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        parameters = json.load(f)

    cost_config = parameters.get("cost_function", {"type": "bhattacharyya"})
    cost_type = cost_config.get("type", "bhattacharyya")
    time_win = args.time_win or parameters.get("time_win", 15)
    param = parameters.get("stitcher_args", {})

    print(f"=== Stitch Threshold Calibration ===")
    print(f"Config: {args.config}")
    print(f"Cost function: {cost_type}")
    print(f"Strategy: {args.strategy}")
    print(f"Time window: {time_win}s")
    print(f"Scenarios: {args.scenarios}")

    # Create cost function
    print(f"\nInitializing cost function...")
    cost_fn = CostFunctionFactory.create(cost_config)

    # Run calibration per scenario
    results = []
    for scenario in args.scenarios:
        result = calibrate_scenario(
            scenario, cost_fn, time_win, param,
            strategy=args.strategy, max_fpr=args.max_fpr,
            min_precision=args.min_precision, f_beta=args.f_beta,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("\nNo valid results. Check that RAW files exist and contain GT labels.")
        sys.exit(1)

    # Weighted average across scenarios
    total_pairs = sum(r["n_pairs"] for r in results)
    avg_thresh = sum(r["optimal_thresh"] * r["n_pairs"] for r in results) / total_pairs
    avg_margin = sum(r["margin"] * r["n_pairs"] for r in results) / total_pairs

    # Aggregate cost distributions
    all_pos_means = [r["pos_cost_mean"] for r in results]
    all_neg_means = [r["neg_cost_mean"] for r in results]

    print(f"\n{'='*50}")
    print(f"=== Calibration Results ===")
    print(f"Config: {args.config}")
    print(f"Cost function: {cost_type}")
    print()

    for r in results:
        print(
            f"Scenario {r['scenario']:>3s}:  "
            f"optimal_thresh={r['optimal_thresh']:.3f}, "
            f"margin={r['margin']:.3f}, "
            f"pairs={r['n_pairs']} "
            f"(pos={r['n_pos']}, neg={r['n_neg']}), "
            f"expected_FP_edges={r['expected_false_edges']}"
        )

    print()
    print(f"Average optimal threshold: {avg_thresh:.3f} (weighted by pair count)")
    print(f"Average margin: {avg_margin:.3f}")
    print()
    print(f"Cost distribution (positive pairs): "
          f"mean={np.mean(all_pos_means):.3f}")
    print(f"Cost distribution (negative pairs): "
          f"mean={np.mean(all_neg_means):.3f}")


if __name__ == "__main__":
    main()
