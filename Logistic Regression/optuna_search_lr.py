"""
Optuna-based Bayesian optimization for logistic-regression cost function tuning.

Two-phase nested studies:
  Phase 1 - Train LR models with Optuna-suggested sklearn hyperparameters
            (C, penalty) using top-10 consensus features. Objective: CV ROC-AUC.
  Phase 2 - For top models, tune pipeline parameters (scale_factor, stitch_thresh,
            time_penalty) across multiple scenarios. Objective: composite MOT score.

Usage:
    python "Logistic Regression/optuna_search_lr.py" --scenarios i,ii,iii
    python "Logistic Regression/optuna_search_lr.py" --scenarios iii --n-pipeline-trials 40
    python "Logistic Regression/optuna_search_lr.py" --keep-best
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
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.trial import TrialState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PARAM_PATH = ROOT / "parameters.json"
DATASET_PATH = ROOT / "Logistic Regression" / "training_dataset_advanced.npz"
MODEL_DIR = ROOT / "Logistic Regression" / "model_artifacts" / "grid_search"
OUTPUT_DIR = ROOT / "Logistic Regression" / "feature_selection_outputs"
DB_PATH = OUTPUT_DIR / "optuna_study.db"

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

# Historical max values per scenario for normalization
# (from CLAUDE.md results + some headroom)
HIST_MAX_SW_GT = {"i": 2.0, "ii": 20.0, "iii": 5.0}
HIST_MAX_FGMT_GT = {"i": 6.0, "ii": 6.0, "iii": 6.0}

CSV_HEADER = [
    "trial", "C", "penalty", "solver", "model_path",
    "cv_roc_auc", "test_roc_auc",
    "scale_factor", "stitch_thresh", "master_stitch_thresh", "time_penalty",
    "scenario", "composite_score",
] + HOTA_METRIC_KEYS + ["status"]


# ---------------------------------------------------------------------------
# Helpers (mirrored from grid_search_lr.py)
# ---------------------------------------------------------------------------

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


def compute_composite_score(metrics, scenario):
    """
    Compute composite MOT score for a single scenario.

    Weights: 0.40*HOTA + 0.25*MOTA + 0.20*IDF1
             + 0.10*(1 - norm_SwGT) + 0.05*(1 - norm_FgmtGT)
    """
    hota = metrics.get("HOTA", 0.0)
    mota = metrics.get("MOTA", 0.0)
    idf1 = metrics.get("IDF1", 0.0)

    sw_gt = metrics.get("Sw/GT", 0.0)
    fgmt_gt = metrics.get("Fgmt/GT", 0.0)

    max_sw = HIST_MAX_SW_GT.get(scenario, 5.0)
    max_fgmt = HIST_MAX_FGMT_GT.get(scenario, 6.0)

    norm_sw = min(sw_gt / max_sw, 1.0)
    norm_fgmt = min(fgmt_gt / max_fgmt, 1.0)

    score = (
        0.40 * hota
        + 0.25 * mota
        + 0.20 * idf1
        + 0.10 * (1.0 - norm_sw)
        + 0.05 * (1.0 - norm_fgmt)
    )
    return score


# ---------------------------------------------------------------------------
# Phase 1 - Model Training with Optuna
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Trains and caches LR models. Used as Optuna Phase 1 objective."""

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Load dataset once
        data = np.load(str(DATASET_PATH), allow_pickle=True)
        X_all = data["X"]
        self.y = data["y"]
        all_features = list(data["feature_names"])

        indices = [all_features.index(f) for f in TOP10_FEATURES]
        self.X = X_all[:, indices]

        # Pre-compute eval split
        scaler_eval = StandardScaler()
        X_scaled = scaler_eval.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y,
        )
        self.X_scaled = X_scaled

        # Cache: tag -> model_info dict
        self._cache = {}
        self._load_existing_models()

        print(f"  Dataset: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"  Positive: {np.sum(self.y)} ({100 * np.mean(self.y):.1f}%)")

    def _load_existing_models(self):
        """Load already-trained models from disk to avoid retraining."""
        if not MODEL_DIR.exists():
            return
        for pkl_path in MODEL_DIR.glob("lr_*.pkl"):
            try:
                with open(pkl_path, "rb") as f:
                    artifacts = pickle.load(f)
                hp = artifacts.get("hyperparams", {})
                tag = f"C{hp['C']}_{hp['penalty']}_{hp['solver']}"
                m = artifacts.get("metrics", {})
                rel_path = str(pkl_path.relative_to(ROOT))
                self._cache[tag] = {
                    "C": hp["C"], "penalty": hp["penalty"], "solver": hp["solver"],
                    "model_path": rel_path,
                    "cv_roc_auc": m.get("cv_roc_auc_mean", 0),
                    "test_roc_auc": m.get("test_roc_auc", 0),
                }
            except Exception:
                continue
        if self._cache:
            print(f"  Loaded {len(self._cache)} cached models from disk")

    def __call__(self, trial):
        """Optuna objective for Phase 1."""
        C = trial.suggest_float("C", 0.001, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        solver = solver_for_penalty(penalty)

        tag = f"C{C}_{penalty}_{solver}"

        # Check cache
        if tag in self._cache:
            cv_roc = self._cache[tag]["cv_roc_auc"]
            trial.set_user_attr("model_info", self._cache[tag])
            return cv_roc

        extra = {}
        if penalty == "elasticnet":
            extra["l1_ratio"] = 0.5

        try:
            # Eval model
            lr_eval = LogisticRegression(
                C=C, penalty=penalty, solver=solver, max_iter=2000, **extra,
            )
            lr_eval.fit(self.X_train, self.y_train)
            y_proba = lr_eval.predict_proba(self.X_test)[:, 1]
            test_roc = roc_auc_score(self.y_test, y_proba)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(lr_eval, self.X_scaled, self.y, cv=cv, scoring="roc_auc")
            cv_roc = cv_scores.mean()

            # Early pruning
            if cv_roc < 0.95:
                raise optuna.TrialPruned(f"cv_roc_auc={cv_roc:.4f} < 0.95")

            # Deploy model (retrain on full data)
            scaler_full = StandardScaler()
            X_full_scaled = scaler_full.fit_transform(self.X)
            lr_full = LogisticRegression(
                C=C, penalty=penalty, solver=solver, max_iter=2000, **extra,
            )
            lr_full.fit(X_full_scaled, self.y)

            pkl_name = f"lr_{tag}.pkl"
            pkl_path = MODEL_DIR / pkl_name
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
            model_info = {
                "C": C, "penalty": penalty, "solver": solver,
                "model_path": rel_path,
                "cv_roc_auc": cv_roc, "test_roc_auc": test_roc,
            }
            self._cache[tag] = model_info
            trial.set_user_attr("model_info", model_info)

            print(f"    C={C:.4f} {penalty:>11s}  cv_ROC={cv_roc:.4f}  test_ROC={test_roc:.4f}")
            return cv_roc

        except optuna.TrialPruned:
            raise
        except Exception as exc:
            print(f"    FAILED: {exc}")
            raise optuna.TrialPruned(str(exc))

    def get_top_models(self, n=5):
        """Return top-n models by CV ROC-AUC from cache."""
        models = sorted(self._cache.values(), key=lambda m: m["cv_roc_auc"], reverse=True)
        return models[:n]


# ---------------------------------------------------------------------------
# Phase 2 - Pipeline Tuning with Optuna
# ---------------------------------------------------------------------------

class PipelineTuner:
    """Tunes pipeline parameters across scenarios. Optuna Phase 2 objective."""

    def __init__(self, model_choices, scenarios, csv_writer, csv_file):
        self.model_choices = model_choices
        self.model_paths = [m["model_path"] for m in model_choices]
        self.model_lookup = {m["model_path"]: m for m in model_choices}
        self.scenarios = scenarios
        self.csv_writer = csv_writer
        self.csv_file = csv_file
        self.base_cfg = json.loads(PARAM_PATH.read_text())
        self.trial_count = 0

    def __call__(self, trial):
        """Optuna objective for Phase 2."""
        self.trial_count += 1

        # Suggest parameters
        model_path = trial.suggest_categorical("model_path", self.model_paths)
        scale_factor = trial.suggest_float("scale_factor", 2.0, 12.0)
        stitch_thresh = trial.suggest_float("stitch_thresh", 0.3, 1.5)
        time_penalty = trial.suggest_float("time_penalty", 0.01, 0.5)
        master_offset = trial.suggest_float("master_offset", 0.1, 0.5)

        master_stitch_thresh = stitch_thresh + master_offset

        model_info = self.model_lookup[model_path]
        print(
            f"\n  [Trial {self.trial_count}] C={model_info['C']:.4f} "
            f"{model_info['penalty']}, scale={scale_factor:.2f}, "
            f"stitch={stitch_thresh:.3f}, time_pen={time_penalty:.3f}, "
            f"master_off={master_offset:.3f}",
            flush=True,
        )

        # Evaluate on each scenario
        scenario_scores = []
        all_scenario_metrics = {}

        for idx, scenario in enumerate(self.scenarios):
            # Update parameters.json
            cfg = deepcopy(self.base_cfg)
            cfg["cost_function"]["type"] = "logistic_regression"
            cfg["cost_function"]["model_path"] = model_path
            cfg["cost_function"]["scale_factor"] = scale_factor
            cfg["cost_function"]["time_penalty"] = time_penalty
            update_thresholds(cfg, stitch_thresh, master_stitch_thresh)

            try:
                PARAM_PATH.write_text(json.dumps(cfg, indent=4))

                # Run pipeline
                run_cmd([sys.executable, "pp_lite.py", scenario], cwd=ROOT)
                run_cmd(
                    [sys.executable, "diagnose_json.py",
                     f"REC_{scenario}.json", "--fix"],
                    cwd=ROOT,
                )

                # Evaluate
                sys.path.insert(0, str(ROOT))
                from hota_trackeval import evaluate_with_trackeval

                gt_file = str(ROOT / f"GT_{scenario}.json")
                rec_file = str(ROOT / f"REC_{scenario}.json")
                metrics = evaluate_with_trackeval(
                    gt_file, rec_file, f"Optuna_T{self.trial_count}_{scenario}"
                )

                if metrics is None:
                    metrics = {}
                    print(f"    Scenario {scenario}: eval returned None")

            except subprocess.CalledProcessError as exc:
                print(f"    Scenario {scenario}: pipeline failed - {exc}")
                if exc.stderr:
                    print(f"    stderr: {exc.stderr[:300]}")
                metrics = {}
            except Exception as exc:
                print(f"    Scenario {scenario}: error - {exc}")
                metrics = {}

            # Compute per-scenario composite score
            score = compute_composite_score(metrics, scenario) if metrics else 0.0
            scenario_scores.append(score)
            all_scenario_metrics[scenario] = (metrics, score)

            # Report intermediate for pruning
            trial.report(np.mean(scenario_scores), idx)
            if trial.should_prune():
                # Still write partial results to CSV
                self._write_csv_rows(trial, model_info, scale_factor, stitch_thresh,
                                     master_stitch_thresh, time_penalty,
                                     all_scenario_metrics, "pruned")
                raise optuna.TrialPruned()

            if metrics:
                print(
                    f"    {scenario}: HOTA={metrics.get('HOTA', 0):.4f} "
                    f"MOTA={metrics.get('MOTA', 0):.4f} "
                    f"IDF1={metrics.get('IDF1', 0):.4f} "
                    f"Sw/GT={metrics.get('Sw/GT', 0):.2f} "
                    f"-> score={score:.4f}"
                )

        # Final composite: mean across scenarios
        final_score = np.mean(scenario_scores) if scenario_scores else 0.0

        # Store user attrs
        trial.set_user_attr("per_scenario_scores", {
            s: sc for s, sc in zip(self.scenarios, scenario_scores)
        })
        trial.set_user_attr("model_info", model_info)

        print(f"    => Composite score: {final_score:.4f}")

        # Write CSV rows
        self._write_csv_rows(trial, model_info, scale_factor, stitch_thresh,
                             master_stitch_thresh, time_penalty,
                             all_scenario_metrics, "ok")

        return final_score

    def _write_csv_rows(self, trial, model_info, scale_factor, stitch_thresh,
                        master_stitch_thresh, time_penalty,
                        all_scenario_metrics, status):
        """Write one row per scenario to CSV."""
        for scenario, (metrics, score) in all_scenario_metrics.items():
            metric_vals = [metrics.get(k, "") for k in HOTA_METRIC_KEYS]
            row = [
                self.trial_count,
                model_info["C"], model_info["penalty"], model_info["solver"],
                model_info["model_path"],
                f"{model_info['cv_roc_auc']:.4f}",
                f"{model_info['test_roc_auc']:.4f}",
                scale_factor, stitch_thresh, master_stitch_thresh, time_penalty,
                scenario, f"{score:.4f}",
            ] + metric_vals + [status]
            self.csv_writer.writerow(row)
            self.csv_file.flush()

    def restore_config(self):
        """Restore original parameters.json."""
        PARAM_PATH.write_text(json.dumps(self.base_cfg, indent=4))
        print("\nRestored original parameters.json.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(study, output_csv, scenarios):
    """Print top results from the Optuna study."""
    print("\n" + "=" * 70)
    print("OPTUNA SEARCH SUMMARY")
    print("=" * 70)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]

    print(f"  Total trials:     {len(study.trials)}")
    print(f"  Completed:        {len(completed)}")
    print(f"  Pruned:           {len(pruned)}")
    print(f"  Failed:           {len(failed)}")
    print(f"  Scenarios:        {', '.join(scenarios)}")

    if not completed:
        print("  No completed trials.")
        return

    best = study.best_trial
    print(f"\n  Best composite score: {best.value:.4f}")
    print(f"  Best params:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    per_scenario = best.user_attrs.get("per_scenario_scores", {})
    if per_scenario:
        print(f"  Per-scenario scores:")
        for s, sc in per_scenario.items():
            print(f"    {s}: {sc:.4f}")

    # Top 5 from CSV
    if output_csv.exists():
        rows = []
        with output_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ok" and row.get("composite_score"):
                    try:
                        row["_score"] = float(row["composite_score"])
                    except ValueError:
                        continue
                    rows.append(row)

        # Group by trial and average composite score across scenarios
        from collections import defaultdict
        trial_scores = defaultdict(list)
        trial_rows = {}
        for row in rows:
            tid = row["trial"]
            trial_scores[tid].append(row["_score"])
            trial_rows[tid] = row  # keep last row for display

        ranked = sorted(trial_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)

        print(f"\n  Top 5 trial configurations:")
        print(f"  {'Rank':<5} {'Trial':<7} {'C':<8} {'Penalty':<12} "
              f"{'Scale':<7} {'Stitch':<8} {'TimePen':<8} {'Score':>8}")
        print("  " + "-" * 70)
        for i, (tid, scores) in enumerate(ranked[:5], 1):
            r = trial_rows[tid]
            print(f"  {i:<5} {tid:<7} {r['C']:<8} {r['penalty']:<12} "
                  f"{float(r['scale_factor']):<7.2f} "
                  f"{float(r['stitch_thresh']):<8.3f} "
                  f"{float(r['time_penalty']):<8.3f} "
                  f"{np.mean(scores):>8.4f}")

    print(f"\n  Full results: {output_csv}")
    print(f"  Optuna DB:    {DB_PATH}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optuna Bayesian optimization: LR hyperparameters + pipeline tuning",
    )
    parser.add_argument("--scenarios", default="i,ii,iii",
                        help="Comma-separated scenario suffixes (default: i,ii,iii)")

    # Phase 1
    parser.add_argument("--n-model-trials", type=int, default=20,
                        help="Number of Optuna trials for LR model search (default: 20)")

    # Phase 2
    parser.add_argument("--n-pipeline-trials", type=int, default=80,
                        help="Number of Optuna trials for pipeline tuning (default: 80)")
    parser.add_argument("--n-top-models", type=int, default=5,
                        help="Number of top models from Phase 1 to use in Phase 2 (default: 5)")

    # Output / behavior
    parser.add_argument("--study-name", default="lr_optuna",
                        help="Optuna study name (default: lr_optuna)")
    parser.add_argument("--output-csv",
                        default="Logistic Regression/feature_selection_outputs/optuna_search_results.csv",
                        help="Output CSV path relative to repo root")
    parser.add_argument("--keep-best", action="store_true",
                        help="Write best config to parameters.json at the end")

    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    output_csv = ROOT / args.output_csv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OPTUNA BAYESIAN OPTIMIZATION FOR LR PIPELINE")
    print("=" * 70)
    print(f"  Scenarios:          {scenarios}")
    print(f"  Model trials:       {args.n_model_trials}")
    print(f"  Pipeline trials:    {args.n_pipeline_trials}")
    print(f"  Top models (Ph2):   {args.n_top_models}")
    print(f"  Study name:         {args.study_name}")
    print(f"  Output CSV:         {output_csv}")
    print(f"  Optuna DB:          {DB_PATH}")
    print("=" * 70)

    # ---- Phase 1: Model Training ----
    print("\n" + "=" * 70)
    print("PHASE 1: BAYESIAN MODEL SEARCH")
    print("=" * 70)

    trainer = ModelTrainer()

    model_study = optuna.create_study(
        study_name=f"{args.study_name}_models",
        storage=f"sqlite:///{DB_PATH}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=5, seed=42,
        ),
    )

    # Only run new trials if needed
    existing_complete = len([
        t for t in model_study.trials if t.state == TrialState.COMPLETE
    ])
    n_new = max(0, args.n_model_trials - existing_complete)
    if n_new > 0:
        print(f"  Running {n_new} new model trials ({existing_complete} already done)")
        model_study.optimize(trainer, n_trials=n_new, show_progress_bar=False)
    else:
        print(f"  {existing_complete} model trials already completed, skipping Phase 1")

    top_models = trainer.get_top_models(n=args.n_top_models)
    if not top_models:
        print("No models trained successfully. Exiting.")
        return

    print(f"\n  Top {len(top_models)} models for Phase 2:")
    for i, m in enumerate(top_models, 1):
        print(f"    {i}. C={m['C']:.4f} {m['penalty']:>11s}  "
              f"cv_ROC={m['cv_roc_auc']:.4f}  path={m['model_path']}")

    # ---- Phase 2: Pipeline Tuning ----
    print("\n" + "=" * 70)
    print("PHASE 2: BAYESIAN PIPELINE TUNING")
    print("=" * 70)
    print(f"  Evaluating on scenarios: {scenarios}")
    print(f"  ~{args.n_pipeline_trials * len(scenarios)} pipeline runs total")

    pipeline_study = optuna.create_study(
        study_name=f"{args.study_name}_pipeline",
        storage=f"sqlite:///{DB_PATH}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=10, seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=0,
        ),
    )

    # Check existing trials
    existing_pipeline = len([
        t for t in pipeline_study.trials if t.state == TrialState.COMPLETE
    ])
    n_new_pipeline = max(0, args.n_pipeline_trials - existing_pipeline)

    # Open CSV
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    csv_file = output_csv.open("a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(CSV_HEADER)
        csv_file.flush()

    tuner = PipelineTuner(top_models, scenarios, csv_writer, csv_file)

    try:
        if n_new_pipeline > 0:
            print(f"  Running {n_new_pipeline} new pipeline trials "
                  f"({existing_pipeline} already done)")
            pipeline_study.optimize(
                tuner, n_trials=n_new_pipeline, show_progress_bar=False,
            )
        else:
            print(f"  {existing_pipeline} pipeline trials already completed")
    except KeyboardInterrupt:
        print("\n  Interrupted! Saving progress...")
    finally:
        tuner.restore_config()
        csv_file.close()

    # ---- Keep best ----
    if args.keep_best:
        completed = [t for t in pipeline_study.trials if t.state == TrialState.COMPLETE]
        if completed:
            best = pipeline_study.best_trial
            cfg = deepcopy(tuner.base_cfg)
            cfg["cost_function"]["type"] = "logistic_regression"
            cfg["cost_function"]["model_path"] = best.params["model_path"]
            cfg["cost_function"]["scale_factor"] = best.params["scale_factor"]
            cfg["cost_function"]["time_penalty"] = best.params["time_penalty"]
            stitch = best.params["stitch_thresh"]
            master = stitch + best.params["master_offset"]
            update_thresholds(cfg, stitch, master)
            PARAM_PATH.write_text(json.dumps(cfg, indent=4))
            print(f"\nBest config written to parameters.json (score={best.value:.4f})")

    # ---- Summary ----
    print_summary(pipeline_study, output_csv, scenarios)


if __name__ == "__main__":
    main()
