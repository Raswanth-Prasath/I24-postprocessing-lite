"""
Comprehensive grid search for logistic-regression cost function tuning.

Two-phase search:
  Phase 1 – Retrain LR models with different sklearn hyperparameters
            (C, penalty, solver) using the top-10 consensus features.
  Phase 2 – For each trained model, sweep pipeline parameters
            (scale_factor, stitch_thresh, time_penalty) and evaluate
            end-to-end with HOTA / TrackEval metrics.

Usage:
    python "Logistic Regression/grid_search_lr.py" --scenario iii
    python "Logistic Regression/grid_search_lr.py" --scenario iii --keep-best
    python "Logistic Regression/grid_search_lr.py" --scenario iii --C-values 0.1,1,10 --scale-factors 5,7
"""

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PARAM_PATH = ROOT / "parameters.json"
DATASET_PATH = ROOT / "Logistic Regression" / "training_dataset_advanced.npz"
MODEL_DIR = ROOT / "Logistic Regression" / "model_artifacts" / "grid_search"
OUTPUT_DIR = ROOT / "Logistic Regression" / "feature_selection_outputs"

TOP10_FEATURES = [
    "y_diff",
    "time_gap",
    "projection_error_x_max",
    "length_diff",
    "width_diff",
    "projection_error_y_max",
    "bhattacharyya_coeff",
    "projection_error_x_mean",
    "curvature_diff",
    "projection_error_x_std",
]

HOTA_METRIC_KEYS = [
    "HOTA", "DetA", "AssA", "LocA",
    "MOTA", "MOTP", "IDF1", "IDR", "IDP",
    "Precision", "Recall",
    "IDsw", "FP", "FN", "Frag",
    "No. trajs", "Fgmt/GT", "Sw/GT",
]

CSV_HEADER = [
    "C", "penalty", "solver", "model_path",
    "cv_roc_auc", "test_roc_auc",
    "scale_factor", "stitch_thresh", "master_stitch_thresh", "time_penalty",
] + HOTA_METRIC_KEYS + ["status"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_float_list(value):
    """Parse comma-separated string into list of floats."""
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value):
    """Parse comma-separated string into list of strings."""
    return [x.strip() for x in value.split(",") if x.strip()]


def solver_for_penalty(penalty):
    """Pick a compatible solver for the given penalty."""
    if penalty in ("l1", "elasticnet"):
        return "saga"
    return "lbfgs"


def update_thresholds(cfg, stitch, master):
    """Set stitch/master thresholds at top-level and inside stitcher_args."""
    cfg["stitch_thresh"] = stitch
    cfg["master_stitch_thresh"] = master
    stitcher_args = cfg.get("stitcher_args")
    if isinstance(stitcher_args, dict):
        stitcher_args["stitch_thresh"] = stitch
        stitcher_args["master_stitch_thresh"] = master


def run_cmd(args, cwd):
    """Run a subprocess, raising on failure."""
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


def load_completed_combos(csv_path):
    """Load already-completed (C, penalty, scale, stitch, time_pen) from CSV for resume."""
    done = set()
    if not csv_path.exists():
        return done
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                key = (
                    row["C"], row["penalty"],
                    row["scale_factor"], row["stitch_thresh"], row["time_penalty"],
                )
                done.add(key)
    return done


# ---------------------------------------------------------------------------
# Phase 1 – Train LR models
# ---------------------------------------------------------------------------

def train_lr_models(C_values, penalties):
    """
    Train LR models for each (C, penalty) combo.
    Returns list of dicts with model info.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING LR MODELS")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = np.load(str(DATASET_PATH), allow_pickle=True)
    X_all = data["X"]
    y = data["y"]
    all_features = list(data["feature_names"])

    indices = [all_features.index(f) for f in TOP10_FEATURES]
    X = X_all[:, indices]

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive class: {np.sum(y)} ({100 * np.mean(y):.1f}%)")

    # Scale once for evaluation split
    scaler_eval = StandardScaler()
    X_scaled = scaler_eval.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y,
    )

    models = []
    for C, penalty in product(C_values, penalties):
        solver = solver_for_penalty(penalty)
        tag = f"C{C}_{penalty}_{solver}"
        pkl_name = f"lr_{tag}.pkl"
        pkl_path = MODEL_DIR / pkl_name

        print(f"\n  Training: C={C}, penalty={penalty}, solver={solver} ... ", end="", flush=True)

        extra = {}
        if penalty == "elasticnet":
            extra["l1_ratio"] = 0.5

        try:
            # Eval model (on train/test split)
            lr_eval = LogisticRegression(
                C=C, penalty=penalty, solver=solver, max_iter=2000, **extra,
            )
            lr_eval.fit(X_train, y_train)
            y_proba_test = lr_eval.predict_proba(X_test)[:, 1]
            test_roc = roc_auc_score(y_test, y_proba_test)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(lr_eval, X_scaled, y, cv=cv, scoring="roc_auc")
            cv_roc = cv_scores.mean()

            # Deploy model (retrained on full data)
            scaler_full = StandardScaler()
            X_full_scaled = scaler_full.fit_transform(X)
            lr_full = LogisticRegression(
                C=C, penalty=penalty, solver=solver, max_iter=2000, **extra,
            )
            lr_full.fit(X_full_scaled, y)

            artifacts = {
                "model": lr_full,
                "scaler": scaler_full,
                "feature_names": TOP10_FEATURES,
                "n_features": len(TOP10_FEATURES),
                "hyperparams": {"C": C, "penalty": penalty, "solver": solver},
                "metrics": {
                    "test_roc_auc": test_roc,
                    "cv_roc_auc_mean": cv_roc,
                    "cv_roc_auc_std": cv_scores.std(),
                },
                "created": datetime.now().isoformat(),
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(artifacts, f)

            rel_path = str(pkl_path.relative_to(ROOT))
            print(f"test_ROC={test_roc:.4f}  cv_ROC={cv_roc:.4f}")

            models.append({
                "C": C, "penalty": penalty, "solver": solver,
                "model_path": rel_path,
                "cv_roc_auc": cv_roc, "test_roc_auc": test_roc,
            })

        except Exception as exc:
            print(f"FAILED ({exc})")

    print(f"\n  Trained {len(models)} models successfully.")
    return models


# ---------------------------------------------------------------------------
# Phase 2 – Pipeline sweep
# ---------------------------------------------------------------------------

def pipeline_sweep(models, scale_factors, stitch_thresholds, time_penalties,
                   master_offset, scenario, output_csv, resume):
    """
    For each model x pipeline-param combo, run the full pipeline and
    evaluate with HOTA/TrackEval.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: PIPELINE PARAMETER SWEEP")
    print("=" * 70)

    base_cfg = json.loads(PARAM_PATH.read_text())

    # Resume support
    done = set()
    if resume:
        done = load_completed_combos(output_csv)
        if done:
            print(f"  Resuming: {len(done)} combinations already completed.")

    total = len(models) * len(scale_factors) * len(stitch_thresholds) * len(time_penalties)
    print(f"  Total combinations: {total}  (skipping {len(done)} done)")

    best = {"hota": float("-inf"), "cfg": None, "row": None}

    # Open CSV (append if resuming, else write fresh)
    mode = "a" if resume and done else "w"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_csv.open(mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(CSV_HEADER)

            count = 0
            for model_info in models:
                for scale, stitch, time_pen in product(
                    scale_factors, stitch_thresholds, time_penalties,
                ):
                    count += 1
                    combo_key = (
                        str(model_info["C"]), model_info["penalty"],
                        str(scale), str(stitch), str(time_pen),
                    )
                    if combo_key in done:
                        continue

                    master = stitch + master_offset

                    print(
                        f"\n[{count}/{total}] C={model_info['C']}, "
                        f"penalty={model_info['penalty']}, "
                        f"scale={scale}, stitch={stitch}, time_pen={time_pen}",
                        flush=True,
                    )

                    # Update parameters.json
                    cfg = deepcopy(base_cfg)
                    cfg["cost_function"]["type"] = "logistic_regression"
                    cfg["cost_function"]["model_path"] = model_info["model_path"]
                    cfg["cost_function"]["scale_factor"] = scale
                    cfg["cost_function"]["time_penalty"] = time_pen
                    update_thresholds(cfg, stitch, master)
                    PARAM_PATH.write_text(json.dumps(cfg, indent=4))

                    # Run pipeline
                    metrics = {}
                    status = "ok"
                    try:
                        run_cmd([sys.executable, "pp_lite.py", scenario], cwd=ROOT)
                        run_cmd(
                            [sys.executable, "diagnose_json.py",
                             f"REC_{scenario}.json", "--fix"],
                            cwd=ROOT,
                        )

                        # Evaluate with HOTA/TrackEval (import here to avoid
                        # import-time side effects at module level)
                        sys.path.insert(0, str(ROOT))
                        from hota_trackeval import evaluate_with_trackeval

                        gt_file = str(ROOT / f"GT_{scenario}.json")
                        rec_file = str(ROOT / f"REC_{scenario}.json")
                        metrics = evaluate_with_trackeval(gt_file, rec_file,
                                                          f"GridSearch_{count}")
                        if metrics is None:
                            metrics = {}
                            status = "eval_failed"

                    except subprocess.CalledProcessError as exc:
                        print(f"  Pipeline failed: {exc}", file=sys.stderr)
                        if exc.stderr:
                            print(f"  stderr: {exc.stderr[:500]}", file=sys.stderr)
                        status = f"pipeline_error"
                    except Exception as exc:
                        print(f"  Error: {exc}", file=sys.stderr)
                        status = f"error"

                    # Track best HOTA
                    hota_val = metrics.get("HOTA", float("-inf"))
                    if hota_val > best["hota"]:
                        best["hota"] = hota_val
                        best["cfg"] = deepcopy(cfg)
                        best["row"] = {**model_info, "scale": scale,
                                       "stitch": stitch, "time_pen": time_pen}

                    # Write CSV row
                    metric_vals = [metrics.get(k, "") for k in HOTA_METRIC_KEYS]
                    row = [
                        model_info["C"], model_info["penalty"], model_info["solver"],
                        model_info["model_path"],
                        f"{model_info['cv_roc_auc']:.4f}",
                        f"{model_info['test_roc_auc']:.4f}",
                        scale, stitch, master, time_pen,
                    ] + metric_vals + [status]
                    writer.writerow(row)
                    f.flush()

                    if status == "ok" and metrics:
                        print(f"  -> HOTA={metrics.get('HOTA', '?'):.4f}  "
                              f"MOTA={metrics.get('MOTA', '?'):.4f}  "
                              f"IDF1={metrics.get('IDF1', '?'):.4f}  "
                              f"IDsw={metrics.get('IDsw', '?'):.0f}")

    finally:
        # Restore original parameters.json
        PARAM_PATH.write_text(json.dumps(base_cfg, indent=4))
        print("\nRestored original parameters.json.")

    return best


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(output_csv, best):
    """Print top-5 results by HOTA."""
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)

    if best["cfg"] is None:
        print("  No successful runs.")
        return

    print(f"\n  Best HOTA: {best['hota']:.4f}")
    if best["row"]:
        r = best["row"]
        print(f"  Config: C={r['C']}, penalty={r['penalty']}, "
              f"scale={r['scale']}, stitch={r['stitch']}, time_pen={r['time_pen']}")
        print(f"  Model:  {r['model_path']}")

    # Read CSV and sort by HOTA
    if output_csv.exists():
        rows = []
        with output_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ok" and row.get("HOTA"):
                    try:
                        row["HOTA_val"] = float(row["HOTA"])
                    except ValueError:
                        continue
                    rows.append(row)

        rows.sort(key=lambda r: r["HOTA_val"], reverse=True)

        print(f"\n  Top 5 configurations by HOTA:")
        print(f"  {'Rank':<5} {'C':<8} {'Penalty':<12} {'Scale':<7} "
              f"{'Stitch':<8} {'TimePen':<8} {'HOTA':>8} {'MOTA':>8} {'IDF1':>8} {'IDsw':>8}")
        print("  " + "-" * 90)
        for i, row in enumerate(rows[:5], 1):
            print(f"  {i:<5} {row['C']:<8} {row['penalty']:<12} "
                  f"{row['scale_factor']:<7} {row['stitch_thresh']:<8} "
                  f"{row['time_penalty']:<8} "
                  f"{float(row.get('HOTA', 0)):>8.4f} "
                  f"{float(row.get('MOTA', 0)):>8.4f} "
                  f"{float(row.get('IDF1', 0)):>8.4f} "
                  f"{float(row.get('IDsw', 0)):>8.0f}")

    print(f"\n  Full results: {output_csv}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive grid search: LR hyperparameters + pipeline tuning",
    )
    parser.add_argument("--scenario", default="iii",
                        help="Scenario suffix (default: iii)")

    # Phase 1: LR hyperparameters
    parser.add_argument("--C-values", default="0.01,0.1,0.5,1.0,5.0,10.0",
                        help="Comma-separated C (regularization) values")
    parser.add_argument("--penalties", default="l1,l2,elasticnet",
                        help="Comma-separated penalty types")

    # Phase 2: Pipeline parameters
    parser.add_argument("--scale-factors", default="3,5,7,10",
                        help="Comma-separated scale_factor values")
    parser.add_argument("--stitch-thresholds", default="0.5,0.7,0.8,1.0",
                        help="Comma-separated stitch_thresh values")
    parser.add_argument("--time-penalties", default="0.05,0.1,0.2",
                        help="Comma-separated time_penalty values")
    parser.add_argument("--master-offset", type=float, default=0.2,
                        help="master_stitch_thresh = stitch + offset (default: 0.2)")

    # Output / behavior
    parser.add_argument("--output",
                        default="Logistic Regression/feature_selection_outputs/grid_search_full.csv",
                        help="Output CSV path (relative to repo root)")
    parser.add_argument("--keep-best", action="store_true",
                        help="Write best config to parameters.json at the end")
    parser.add_argument("--resume", action="store_true",
                        help="Skip combos already in the output CSV")

    args = parser.parse_args()

    output_csv = ROOT / args.output

    # Parse parameter grids
    C_values = parse_float_list(args.C_values)
    penalties = parse_str_list(args.penalties)
    scale_factors = parse_float_list(args.scale_factors)
    stitch_thresholds = parse_float_list(args.stitch_thresholds)
    time_penalties = parse_float_list(args.time_penalties)

    n_models = len(C_values) * len(penalties)
    n_pipeline = len(scale_factors) * len(stitch_thresholds) * len(time_penalties)
    total = n_models * n_pipeline

    print("=" * 70)
    print("COMPREHENSIVE LR GRID SEARCH")
    print("=" * 70)
    print(f"  Scenario:           {args.scenario}")
    print(f"  C values:           {C_values}")
    print(f"  Penalties:          {penalties}")
    print(f"  Scale factors:      {scale_factors}")
    print(f"  Stitch thresholds:  {stitch_thresholds}")
    print(f"  Time penalties:     {time_penalties}")
    print(f"  Master offset:      {args.master_offset}")
    print(f"  LR models to train: {n_models}")
    print(f"  Pipeline combos:    {n_pipeline}")
    print(f"  Total evaluations:  {total}")
    print(f"  Output CSV:         {output_csv}")
    print(f"  Resume:             {args.resume}")
    print("=" * 70)

    # Phase 1
    models = train_lr_models(C_values, penalties)
    if not models:
        print("No models trained successfully. Exiting.")
        return

    # Phase 2
    best = pipeline_sweep(
        models, scale_factors, stitch_thresholds, time_penalties,
        args.master_offset, args.scenario, output_csv, args.resume,
    )

    # Keep best config if requested
    if args.keep_best and best["cfg"] is not None:
        PARAM_PATH.write_text(json.dumps(best["cfg"], indent=4))
        print(f"\nBest config written to parameters.json (HOTA={best['hota']:.4f})")

    # Summary
    print_summary(output_csv, best)


if __name__ == "__main__":
    main()
