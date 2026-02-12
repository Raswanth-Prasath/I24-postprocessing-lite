#!/usr/bin/env python3
"""Logistic regression diagnostics for I24 stitching models."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "Logistic Regression/training_dataset_advanced.npz"
DEFAULT_MODEL = "Logistic Regression/model_artifacts/consensus_top10_full47.pkl"
DEFAULT_OUTPUT_DIR = "Logistic Regression/feature_selection_outputs"


def resolve_existing_path(path_str, fallbacks=None):
    """Resolve an existing file path with backward-compatible fallbacks."""
    candidates = []
    p = Path(path_str)
    candidates.append(p)
    if not p.is_absolute():
        candidates.append(ROOT / p)

    for fb in fallbacks or []:
        fbp = Path(fb)
        candidates.append(fbp)
        if not fbp.is_absolute():
            candidates.append(ROOT / fbp)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Could not resolve existing path for: {path_str}")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    rows = []
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin": b,
                "count": 0,
                "pred_mean": None,
                "obs_rate": None,
                "abs_gap": None,
            })
            continue

        pred_mean = float(np.mean(y_prob[mask]))
        obs_rate = float(np.mean(y_true[mask]))
        gap = abs(pred_mean - obs_rate)
        ece += gap * (count / n)

        rows.append({
            "bin": b,
            "count": count,
            "pred_mean": pred_mean,
            "obs_rate": obs_rate,
            "abs_gap": gap,
        })

    return float(ece), rows


def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """Compute classification metrics at a threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": brier,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _try_statsmodels_influence(X_scaled, y_true):
    """Try influence computation using statsmodels GLM."""
    try:
        import statsmodels.api as sm

        X_design = sm.add_constant(X_scaled, has_constant="add")
        glm = sm.GLM(y_true, X_design, family=sm.families.Binomial())
        glm_res = glm.fit(disp=0)
        infl = glm_res.get_influence(observed=True)

        leverage = np.asarray(infl.hat_matrix_diag, dtype=float)
        cook = np.asarray(infl.cooks_distance[0], dtype=float)

        std_res = getattr(infl, "resid_studentized", None)
        if std_res is None:
            pearson = np.asarray(glm_res.resid_pearson, dtype=float)
            std_res = pearson / np.sqrt(np.clip(1.0 - leverage, 1e-8, None))
        else:
            std_res = np.asarray(std_res, dtype=float)

        return {
            "leverage": leverage,
            "std_residual": std_res,
            "cook_distance": cook,
            "method": "statsmodels",
        }
    except Exception:
        return None


def _manual_influence(X_scaled, y_true, y_prob):
    """Numerically stable manual fallback for leverage/residual/Cook's D."""
    X_design = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    w = p * (1.0 - p)

    xtwx = X_design.T @ (w[:, None] * X_design)
    xtwx_inv = np.linalg.pinv(xtwx)

    # leverage h_i = w_i * x_i^T (X'WX)^-1 x_i
    leverage = w * np.einsum("ij,jk,ik->i", X_design, xtwx_inv, X_design)

    pearson = (y_true - p) / np.sqrt(np.clip(w, 1e-8, None))
    std_res = pearson / np.sqrt(np.clip(1.0 - leverage, 1e-8, None))

    k = X_design.shape[1]
    cook = (std_res ** 2 * leverage) / (k * np.clip(1.0 - leverage, 1e-8, None))

    return {
        "leverage": leverage,
        "std_residual": std_res,
        "cook_distance": cook,
        "method": "manual",
    }


def compute_influence(X_scaled, y_true, y_prob, use_statsmodels=True):
    """Compute influence metrics with optional statsmodels first, then fallback."""
    if use_statsmodels:
        statsmodels_result = _try_statsmodels_influence(X_scaled, y_true)
        if statsmodels_result is not None:
            return statsmodels_result
    return _manual_influence(X_scaled, y_true, y_prob)


def ks_statistic(a, b):
    """Compute two-sample KS statistic without scipy dependency."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    vals = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, vals, side="right") / max(len(a), 1)
    cdf_b = np.searchsorted(b, vals, side="right") / max(len(b), 1)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def compute_shortcut_risk(X, y, feature_names, topn=10):
    """Rank features by class-separation risk signals."""
    rows = []
    y = np.asarray(y)

    for i, name in enumerate(feature_names):
        col = X[:, i]
        pos = col[y == 1]
        neg = col[y == 0]

        if np.std(col) < 1e-12:
            abs_corr = 0.0
        else:
            corr = np.corrcoef(col, y)[0, 1]
            abs_corr = 0.0 if np.isnan(corr) else abs(float(corr))

        ks = ks_statistic(pos, neg)

        rows.append({
            "feature": str(name),
            "abs_corr": abs_corr,
            "ks": ks,
            "risk_score": float(0.5 * abs_corr + 0.5 * ks),
        })

    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    return rows[:topn]


def build_oof_probabilities(X_selected, y_true, model):
    """Build OOF probabilities using model hyperparameters with fold-safe scaling."""
    lr_params = model.get_params()
    cv_model = model.__class__(**lr_params)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", cv_model),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(pipe, X_selected, y_true, cv=cv, method="predict_proba")[:, 1]


def run_diagnostics(
    dataset_path,
    model_path,
    output_dir,
    threshold=0.5,
    topk_influential=250,
    use_statsmodels=True,
):
    """Run LR diagnostics and write summary artifacts."""
    dataset_path = resolve_existing_path(dataset_path, [DEFAULT_DATASET])
    model_path = resolve_existing_path(model_path, [DEFAULT_MODEL])

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    scaler = artifact["scaler"]
    selected_features = artifact.get("feature_names") or artifact.get("features")

    data = np.load(str(dataset_path), allow_pickle=True)
    X_full = data["X"]
    y = data["y"].astype(int)
    all_feature_names = [str(f) for f in data["feature_names"]]

    indices = [all_feature_names.index(f) for f in selected_features]
    X_selected = X_full[:, indices]

    X_scaled = scaler.transform(X_selected)
    y_prob_in_sample = model.predict_proba(X_scaled)[:, 1]

    # OOF metrics are more realistic for reporting.
    y_prob_oof = build_oof_probabilities(X_selected, y, model)

    metrics = compute_classification_metrics(y, y_prob_oof, threshold=threshold)
    ece, calib_rows = expected_calibration_error(y, y_prob_oof, n_bins=10)
    metrics["ece"] = ece

    infl = compute_influence(X_scaled, y, y_prob_in_sample, use_statsmodels=use_statsmodels)
    leverage = infl["leverage"]
    std_res = infl["std_residual"]
    cook = infl["cook_distance"]

    n = len(y)
    k = X_scaled.shape[1] + 1
    cook_thr = 4.0 / n
    lev_thr = 2.0 * k / n
    res_thr = 3.0

    influence_rows = []
    for i in range(n):
        cook_flag = bool(cook[i] > cook_thr)
        lev_flag = bool(leverage[i] > lev_thr)
        res_flag = bool(abs(std_res[i]) > res_thr)
        influence_rows.append({
            "sample_index": i,
            "y_true": int(y[i]),
            "probability": float(y_prob_in_sample[i]),
            "cook_distance": float(cook[i]),
            "leverage": float(leverage[i]),
            "std_residual": float(std_res[i]),
            "cook_flag": cook_flag,
            "leverage_flag": lev_flag,
            "residual_flag": res_flag,
            "influential": bool(cook_flag or lev_flag or res_flag),
        })

    influence_rows.sort(key=lambda r: r["cook_distance"], reverse=True)
    influence_rows = influence_rows[:topk_influential]

    shortcut_rows = compute_shortcut_risk(X_full, y, all_feature_names, topn=10)

    stem = dataset_path.stem
    metrics_path = output_dir / f"lr_metrics_{stem}.json"
    influence_path = output_dir / f"lr_influence_points_{stem}.csv"
    summary_path = output_dir / f"lr_diagnostics_summary_{stem}.md"

    metrics_payload = {
        "dataset": str(dataset_path),
        "model_path": str(model_path),
        "n_samples": int(n),
        "n_features_full": int(X_full.shape[1]),
        "n_features_model": int(X_selected.shape[1]),
        "influence_method": infl["method"],
        "thresholds": {
            "classification_threshold": float(threshold),
            "cook_threshold": float(cook_thr),
            "leverage_threshold": float(lev_thr),
            "std_residual_threshold": float(res_thr),
        },
        "metrics_oof": metrics,
        "calibration_bins": calib_rows,
        "influence_summary": {
            "cook_flag_count": int(sum(r["cook_flag"] for r in influence_rows)),
            "leverage_flag_count": int(sum(r["leverage_flag"] for r in influence_rows)),
            "residual_flag_count": int(sum(r["residual_flag"] for r in influence_rows)),
            "influential_count": int(sum(r["influential"] for r in influence_rows)),
        },
        "shortcut_risk_top_features": shortcut_rows,
    }

    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    influence_fieldnames = [
        "sample_index",
        "y_true",
        "probability",
        "cook_distance",
        "leverage",
        "std_residual",
        "cook_flag",
        "leverage_flag",
        "residual_flag",
        "influential",
    ]
    with influence_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=influence_fieldnames)
        writer.writeheader()
        writer.writerows(influence_rows)

    summary_lines = [
        "# Logistic Regression Diagnostics Summary",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Model: `{model_path}`",
        f"- Samples: {n}",
        f"- Full feature count: {X_full.shape[1]}",
        f"- Model feature count: {X_selected.shape[1]}",
        f"- Influence method: `{infl['method']}`",
        "",
        "## OOF Metrics",
        "",
        f"- ROC-AUC: {metrics['roc_auc']:.6f}",
        f"- PR-AUC: {metrics['pr_auc']:.6f}",
        f"- Log Loss: {metrics['log_loss']:.6f}",
        f"- Brier Score: {metrics['brier_score']:.6f}",
        f"- Precision: {metrics['precision']:.6f}",
        f"- Recall: {metrics['recall']:.6f}",
        f"- Specificity: {metrics['specificity']:.6f}",
        f"- F1: {metrics['f1']:.6f}",
        f"- ECE: {metrics['ece']:.6f}",
        "",
        "## Influence Summary",
        "",
        f"- Cook's D threshold (4/n): {cook_thr:.6f}",
        f"- Leverage threshold (2k/n): {lev_thr:.6f}",
        f"- |Standardized residual| threshold: {res_thr:.1f}",
        f"- Flagged influential points (top-k window): {metrics_payload['influence_summary']['influential_count']}",
        "",
        "## Top Shortcut-Risk Features (full dataset)",
        "",
        "| Feature | abs(corr) | KS | Risk score |",
        "|---|---:|---:|---:|",
    ]
    for row in shortcut_rows:
        summary_lines.append(
            f"| {row['feature']} | {row['abs_corr']:.4f} | {row['ks']:.4f} | {row['risk_score']:.4f} |"
        )

    summary_lines.extend([
        "",
        f"- Metrics JSON: `{metrics_path}`",
        f"- Influence CSV: `{influence_path}`",
    ])

    summary_path.write_text("\n".join(summary_lines))

    return {
        "metrics_path": metrics_path,
        "influence_path": influence_path,
        "summary_path": summary_path,
        "metrics": metrics_payload,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run LR diagnostics and influence analysis")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Dataset NPZ path (default: training_dataset_advanced)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL,
                        help="Model artifact path (default: production LR model)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for diagnostic outputs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for confusion metrics")
    parser.add_argument("--topk-influential", type=int, default=250,
                        help="Number of top Cook's distance rows to export")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_diagnostics(
        dataset_path=args.dataset,
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        topk_influential=args.topk_influential,
        use_statsmodels=True,
    )

    print("=" * 70)
    print("LR DIAGNOSTICS COMPLETE")
    print("=" * 70)
    print(f"Metrics JSON:   {result['metrics_path']}")
    print(f"Influence CSV:  {result['influence_path']}")
    print(f"Summary report: {result['summary_path']}")


if __name__ == "__main__":
    main()
