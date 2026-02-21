#!/usr/bin/env python3
"""
Two-phase threshold sweep for stitch_thresh calibration.

Phase 1: Pair-level ROC analysis to select candidate thresholds using multiple
          strategies (fpr_ceiling, precision_floor, f_beta, youden).
Phase 2: End-to-end pipeline runs (pp_lite + hota_trackeval) at each candidate
          threshold to optimise HOTA directly.

Usage:
    python sweep_threshold.py --config parameters_PINN.json
    python sweep_threshold.py --config parameters_PINN.json --scenarios i ii
    python sweep_threshold.py --config parameters_PINN.json --n-thresholds 10
    python sweep_threshold.py --config parameters_PINN.json --thresholds 1.5 2.0 2.5 3.0
    python sweep_threshold.py --config parameters_PINN.json --phase1-only
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from sklearn.metrics import roc_curve

from calibrate_threshold import (
    load_fragments,
    find_raw_path,
    generate_pairs,
    select_threshold,
)
from utils.stitch_cost_interface import CostFunctionFactory


# ---------------------------------------------------------------------------
# Phase 1: pair-level analysis to propose candidate thresholds
# ---------------------------------------------------------------------------

def phase1_candidates(
    config_path: str,
    scenarios: list,
    n_thresholds: int = 7,
    max_fpr_values: tuple = (0.001, 0.005, 0.01),
) -> dict:
    """Compute costs on GT-labelled pairs and propose candidate thresholds.

    Returns dict with keys: candidates (sorted list of floats),
    roc_data (per-scenario raw ROC arrays), pair_stats.
    """
    with open(config_path) as f:
        parameters = json.load(f)

    cost_config = parameters.get("cost_function", {"type": "bhattacharyya"})
    time_win = parameters.get("time_win", 15)
    param = parameters.get("stitcher_args", {})

    cost_fn = CostFunctionFactory.create(cost_config)

    all_costs = []
    all_labels = []
    scenario_data = {}

    for scenario in scenarios:
        try:
            raw_path = find_raw_path(scenario)
        except FileNotFoundError:
            print(f"  [phase1] SKIP scenario {scenario}: RAW file not found")
            continue

        frags = load_fragments(raw_path)
        costs, labels, errors = [], [], 0
        for fa, fb, lab in generate_pairs(frags, time_win):
            c = cost_fn.compute_cost(fa, fb, time_win, param)
            if c >= 1e5:
                errors += 1
                continue
            costs.append(c)
            labels.append(lab)

        costs_arr = np.array(costs)
        labels_arr = np.array(labels)
        n_pos = int(labels_arr.sum())
        n_neg = len(labels_arr) - n_pos
        print(f"  [phase1] scenario {scenario}: {len(labels_arr)} pairs "
              f"(pos={n_pos}, neg={n_neg}, skip={errors})")

        if n_pos == 0 or n_neg == 0:
            continue

        all_costs.append(costs_arr)
        all_labels.append(labels_arr)
        scenario_data[scenario] = {
            "n_pos": n_pos, "n_neg": n_neg, "n_pairs": len(labels_arr),
        }

    if not all_costs:
        print("  [phase1] No valid scenario data")
        return {"candidates": [], "pair_stats": {}}

    costs = np.concatenate(all_costs)
    labels = np.concatenate(all_labels)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    fpr, tpr, thresholds = roc_curve(labels, -costs)
    actual_thresholds = -thresholds

    # Collect candidate thresholds from multiple strategies
    candidate_set = set()

    # fpr_ceiling at several FPR caps
    for mf in max_fpr_values:
        th, _, _, _ = select_threshold(
            fpr, tpr, actual_thresholds, n_pos, n_neg,
            strategy="fpr_ceiling", max_fpr=mf,
        )
        candidate_set.add(round(th, 4))

    # precision_floor at several levels
    for mp in (0.99, 0.995, 0.98):
        th, _, _, _ = select_threshold(
            fpr, tpr, actual_thresholds, n_pos, n_neg,
            strategy="precision_floor", min_precision=mp,
        )
        candidate_set.add(round(th, 4))

    # f_beta at several beta values
    for beta in (0.1, 0.25, 0.5):
        th, _, _, _ = select_threshold(
            fpr, tpr, actual_thresholds, n_pos, n_neg,
            strategy="f_beta", f_beta=beta,
        )
        candidate_set.add(round(th, 4))

    # Youden J for reference
    th, _, _, _ = select_threshold(
        fpr, tpr, actual_thresholds, n_pos, n_neg, strategy="youden",
    )
    candidate_set.add(round(th, 4))

    # Deduplicate, sort, and trim to n_thresholds
    candidates = sorted(candidate_set)
    if len(candidates) > n_thresholds:
        # Keep evenly spaced subset spanning the full range
        indices = np.linspace(0, len(candidates) - 1, n_thresholds, dtype=int)
        candidates = [candidates[i] for i in indices]

    # Print phase-1 summary
    pos_costs = costs[labels == 1]
    neg_costs = costs[labels == 0]
    print(f"\n  [phase1] Aggregated: {len(labels)} pairs "
          f"(pos={n_pos}, neg={n_neg}, ratio={n_neg/n_pos:.1f}:1)")
    print(f"  [phase1] Positive costs: mean={pos_costs.mean():.3f}, "
          f"max={pos_costs.max():.3f}")
    print(f"  [phase1] Negative costs: mean={neg_costs.mean():.3f}, "
          f"min={neg_costs.min():.3f}")
    print(f"  [phase1] Candidate thresholds ({len(candidates)}): "
          f"{[f'{t:.3f}' for t in candidates]}")

    # Annotate each candidate with expected FPR / false edges
    annotated = []
    for th in candidates:
        # Find closest ROC point for this threshold
        idx = np.searchsorted(actual_thresholds, th)
        idx = min(idx, len(fpr) - 1)
        fp_rate = float(fpr[idx])
        tp_rate = float(tpr[idx])
        annotated.append({
            "threshold": th,
            "est_tpr": round(tp_rate, 4),
            "est_fpr": round(fp_rate, 4),
            "est_false_edges": int(round(fp_rate * n_neg)),
        })

    return {
        "candidates": candidates,
        "annotated": annotated,
        "pair_stats": {
            "n_pos": n_pos, "n_neg": n_neg,
            "neg_pos_ratio": round(n_neg / n_pos, 1),
            "pos_cost_mean": round(float(pos_costs.mean()), 3),
            "neg_cost_mean": round(float(neg_costs.mean()), 3),
        },
        "scenarios": scenario_data,
    }


# ---------------------------------------------------------------------------
# Phase 2: end-to-end pipeline sweep
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str, threshold: float, scenario: str,
                 tag: str, timeout_sec: int = 600) -> str:
    """Run pp_lite.py with overridden stitch_thresh. Returns output REC path."""
    with open(config_path) as f:
        params = json.load(f)

    params["stitcher_args"]["stitch_thresh"] = threshold
    params["stitcher_args"]["master_stitch_thresh"] = threshold

    # Ensure timeouts are reasonable
    for key in ("stitcher_timeout", "reconciliation_pool_timeout",
                "reconciliation_writer_timeout", "write_temp_timeout"):
        params.setdefault(key, 300)

    # Write temp config
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="sweep_cfg_")
    with os.fdopen(tmp_fd, "w") as f:
        json.dump(params, f, indent=2)

    rec_path = f"REC_{scenario}_{tag}.json"
    # Remove stale output
    for p in (rec_path, rec_path + ".bak"):
        if os.path.exists(p):
            os.remove(p)

    cmd = [
        sys.executable, "pp_lite.py", scenario,
        "--config", tmp_path, "--tag", tag,
    ]
    print(f"  [phase2] Running: {' '.join(cmd[-5:])}")
    try:
        subprocess.run(
            cmd, timeout=timeout_sec,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        print(f"  [phase2] TIMEOUT after {timeout_sec}s for threshold={threshold:.3f}")
    finally:
        os.unlink(tmp_path)

    # Auto-fix JSON if needed
    if os.path.exists(rec_path):
        subprocess.run(
            [sys.executable, "diagnose_json.py", rec_path, "--fix"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    return rec_path


def evaluate_run(gt_file: str, rec_file: str, name: str) -> dict:
    """Evaluate with hota_trackeval and return metrics dict."""
    if not os.path.exists(rec_file):
        print(f"  [phase2] Missing output: {rec_file}")
        return None

    try:
        from hota_trackeval import evaluate_with_trackeval
        return evaluate_with_trackeval(gt_file, rec_file, name)
    except Exception as e:
        print(f"  [phase2] Evaluation error: {e}")
        return None


def phase2_sweep(
    config_path: str,
    candidates: list,
    scenarios: list,
    timeout_sec: int = 600,
) -> list:
    """Run pipeline + eval at each candidate threshold. Returns list of results."""
    results = []

    for threshold in candidates:
        tag = f"T_sweep_{str(threshold).replace('.', 'p')}"
        run_result = {"threshold": threshold, "tag": tag, "scenarios": {}}

        for scenario in scenarios:
            gt_file = f"GT_{scenario}.json"
            if not os.path.exists(gt_file):
                print(f"  [phase2] SKIP scenario {scenario}: {gt_file} not found")
                continue

            print(f"\n  [phase2] threshold={threshold:.3f}, scenario={scenario}")
            t0 = time.time()
            rec_file = run_pipeline(
                config_path, threshold, scenario, tag, timeout_sec,
            )
            elapsed_pipeline = time.time() - t0

            metrics = evaluate_run(gt_file, rec_file, f"{tag}_{scenario}")
            if metrics is not None:
                metrics["elapsed_sec"] = round(elapsed_pipeline, 1)
                run_result["scenarios"][scenario] = metrics
            else:
                run_result["scenarios"][scenario] = {"status": "FAILED"}

        results.append(run_result)

    return results


# ---------------------------------------------------------------------------
# Summary & output
# ---------------------------------------------------------------------------

def print_summary(results: list, scenarios: list):
    """Print a summary table of the sweep results."""
    print(f"\n{'='*90}")
    print("  THRESHOLD SWEEP SUMMARY")
    print(f"{'='*90}")

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario}")
        header = (
            f"  {'Thresh':>7s} {'HOTA':>7s} {'MOTA':>7s} {'Prec':>7s} "
            f"{'Recall':>7s} {'IDsw':>7s} {'FP':>8s} {'Sw/GT':>7s} "
            f"{'Trajs':>6s} {'Time':>6s}"
        )
        print(header)
        print(f"  {'-'*82}")

        best_hota = -1
        best_thresh = None

        for r in results:
            th = r["threshold"]
            m = r["scenarios"].get(scenario)
            if m is None or "HOTA" not in m:
                print(f"  {th:7.3f} {'FAILED':>7s}")
                continue

            hota = m["HOTA"]
            if hota > best_hota:
                best_hota = hota
                best_thresh = th

            print(
                f"  {th:7.3f} {hota:7.3f} {m['MOTA']:7.3f} "
                f"{m['Precision']:7.3f} {m['Recall']:7.3f} "
                f"{m['IDsw']:7.0f} {m['FP']:8.0f} {m['Sw/GT']:7.2f} "
                f"{m['No. trajs']:6.0f} {m.get('elapsed_sec', 0):5.0f}s"
            )

        if best_thresh is not None:
            print(f"\n  >>> Best HOTA for scenario {scenario}: "
                  f"{best_hota:.3f} at threshold={best_thresh:.3f}")

    print(f"\n{'='*90}")


def save_results(results: list, phase1_data: dict, output_path: str):
    """Save full results to JSON."""
    out = {
        "phase1": phase1_data,
        "phase2": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-phase threshold sweep: pair-level pre-filter → "
                    "end-to-end pipeline optimisation"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to parameters JSON config file"
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["i"],
        help="Scenarios to sweep (default: i)"
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=None,
        help="Explicit thresholds to test (skips phase 1)"
    )
    parser.add_argument(
        "--n-thresholds", type=int, default=7,
        help="Max number of candidate thresholds from phase 1 (default: 7)"
    )
    parser.add_argument(
        "--phase1-only", action="store_true",
        help="Only run phase 1 (pair-level analysis), skip pipeline runs"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Timeout per pipeline run in seconds (default: 600)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: sweep_results_{scenario}.json)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Threshold Sweep")
    print(f"  Config: {args.config}")
    print(f"  Scenarios: {args.scenarios}")
    print("=" * 60)

    # Phase 1
    if args.thresholds:
        candidates = sorted(args.thresholds)
        phase1_data = {"candidates": candidates, "source": "user-specified"}
        print(f"\n[Phase 1] Using user-specified thresholds: "
              f"{[f'{t:.3f}' for t in candidates]}")
    else:
        print(f"\n[Phase 1] Pair-level ROC analysis...")
        phase1_data = phase1_candidates(
            args.config, args.scenarios, n_thresholds=args.n_thresholds,
        )
        candidates = phase1_data["candidates"]

        if not candidates:
            print("No candidate thresholds found. Check RAW files and GT labels.")
            sys.exit(1)

        # Print annotated candidates
        if "annotated" in phase1_data:
            print(f"\n  {'Threshold':>10s} {'Est TPR':>8s} {'Est FPR':>8s} "
                  f"{'Est FP Edges':>13s}")
            print(f"  {'-'*43}")
            for a in phase1_data["annotated"]:
                print(f"  {a['threshold']:10.3f} {a['est_tpr']:8.4f} "
                      f"{a['est_fpr']:8.4f} {a['est_false_edges']:13d}")

    if args.phase1_only:
        output = args.output or "sweep_phase1_results.json"
        save_results([], phase1_data, output)
        return

    # Phase 2
    print(f"\n[Phase 2] End-to-end pipeline sweep "
          f"({len(candidates)} thresholds × {len(args.scenarios)} scenarios)...")
    results = phase2_sweep(
        args.config, candidates, args.scenarios, timeout_sec=args.timeout,
    )

    # Summary
    print_summary(results, args.scenarios)

    # Save
    suffix = "_".join(args.scenarios)
    output = args.output or f"sweep_results_{suffix}.json"
    save_results(results, phase1_data, output)


if __name__ == "__main__":
    main()
