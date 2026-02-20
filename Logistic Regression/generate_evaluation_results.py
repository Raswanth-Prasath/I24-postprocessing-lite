#!/usr/bin/env python3
"""Generate leakage-safe logistic regression evaluation plots and metrics."""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = "Logistic Regression/model_artifacts/consensus_top10_full47.pkl"
DEFAULT_DATASET = "Logistic Regression/training_dataset_advanced.npz"
DEFAULT_OUTPUT_DIR = "Logistic Regression/lr_analysis_plots"

plt.style.use("seaborn-v0_8-whitegrid")


def resolve_path(path_str, default_rel):
    """Resolve path with fallback to default relative path."""
    p = Path(path_str) if path_str else ROOT / default_rel
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        return p

    fallback = ROOT / default_rel
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve path: {path_str}")


def resolve_source_holdout_tags(source_split_tag):
    """Pick deterministic train/test tags for source-holdout evaluation."""
    tags = np.asarray(source_split_tag).astype(str)
    unique_tags, counts = np.unique(tags, return_counts=True)
    if len(unique_tags) < 2:
        return None, None

    if {"advanced_keepall", "v4_diverse_curated"}.issubset(set(unique_tags)):
        return "advanced_keepall", "v4_diverse_curated"

    order = np.argsort(counts)[::-1]
    return str(unique_tags[order[0]]), str(unique_tags[order[1]])


def load_model_and_data(model_path, dataset_path):
    """Load model artifact and select model features from dataset."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    selected_features = model_data.get("feature_names") or model_data.get("features")
    if selected_features is None:
        raise KeyError("Model artifact must contain 'feature_names' or 'features'.")

    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    dataset_features = [str(f) for f in data["feature_names"]]

    feature_indices = [dataset_features.index(str(f)) for f in selected_features]
    X_selected = X[:, feature_indices]

    source_split_tag = data["source_split_tag"] if "source_split_tag" in data else None

    return {
        "model": model,
        "selected_features": [str(f) for f in selected_features],
        "X": X_selected,
        "y": y,
        "source_split_tag": source_split_tag,
    }


def compute_metrics(y_true, y_prob, threshold):
    """Compute summary metrics for one evaluation protocol."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "labels": y_true,
        "predictions": y_pred,
        "probabilities": y_prob,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "counts": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_random_split(model, X, y, threshold, test_size, random_state):
    """Leakage-safe random split: split first, then fit scaler/model on train only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", clone(model)),
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_prob, threshold)
    return {
        "protocol": "random",
        "metrics": metrics,
        "metadata": {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "test_size": float(test_size),
            "random_state": int(random_state),
        },
    }


def evaluate_source_holdout(model, X, y, source_split_tag, threshold):
    """Source-holdout split based on dataset provenance tags."""
    if source_split_tag is None:
        return None, "source_split_tag missing in dataset"

    train_tag, test_tag = resolve_source_holdout_tags(source_split_tag)
    if train_tag is None or test_tag is None:
        return None, "fewer than 2 source tags available"

    tags = np.asarray(source_split_tag).astype(str)
    train_mask = tags == train_tag
    test_mask = tags == test_tag

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None, "empty source-holdout train/test partition"

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None, "single-class train/test split from source tags"

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", clone(model)),
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_prob, threshold)
    return {
        "protocol": "source_holdout",
        "metrics": metrics,
        "metadata": {
            "train_tag": train_tag,
            "test_tag": test_tag,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        },
    }, None


def plot_evaluation_results(metrics, output_path, title_label):
    """Plot evaluation results in 2x2 grid format."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    y_true = metrics["labels"]
    y_prob = metrics["probabilities"]

    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, linewidth=2, color="#2171b5", label=f"LR (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve ({title_label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    ax = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ax.plot(
        recall,
        precision,
        linewidth=2,
        color="#2171b5",
        label=f"LR (AP={metrics['avg_precision']:.3f})",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve ({title_label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0.5, 1.02])

    ax = axes[1, 0]
    cm = metrics["confusion_matrix"]
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix ({title_label})", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Different", "Same"], fontsize=11)
    ax.set_yticklabels(["Different", "Same"], fontsize=11)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2%})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )

    ax = axes[1, 1]
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    ax.hist(
        y_prob[pos_mask],
        bins=30,
        alpha=0.7,
        label="Same Vehicle",
        color="#2ca02c",
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        y_prob[neg_mask],
        bins=30,
        alpha=0.7,
        label="Different Vehicle",
        color="#d62728",
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Threshold 0.5")
    ax.set_xlabel("Probability Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Probability Distribution ({title_label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper center")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def print_metrics_summary(protocol, payload):
    """Print concise metrics summary."""
    m = payload["metrics"]
    meta = payload["metadata"]
    print("\n" + "=" * 64)
    print(f"LOGISTIC REGRESSION EVALUATION ({protocol.upper()})")
    print("=" * 64)
    print(f"ROC-AUC: {m['roc_auc']:.4f}")
    print(f"AP:      {m['avg_precision']:.4f}")
    print(f"Acc:     {m['accuracy']:.4f}")
    print(f"Prec:    {m['precision']:.4f}")
    print(f"Recall:  {m['recall']:.4f}")
    print(f"F1:      {m['f1']:.4f}")
    print(f"Counts:  TN={m['counts']['tn']} FP={m['counts']['fp']} FN={m['counts']['fn']} TP={m['counts']['tp']}")
    print(f"Split:   {meta}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe LR evaluation plots/metrics."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to NPZ dataset.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="Path to model artifact PKL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--eval-protocol",
        choices=["random", "source_holdout", "both"],
        default="random",
        help="Evaluation protocol (default: random).",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Random split test size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random split seed.")
    args = parser.parse_args()

    model_path = resolve_path(args.model_path, DEFAULT_MODEL)
    dataset_path = resolve_path(args.dataset, DEFAULT_DATASET)
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR) if Path(args.output_dir).exists() else (
        ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model path: {model_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output dir: {output_dir}")

    loaded = load_model_and_data(model_path, dataset_path)
    model = loaded["model"]
    X = loaded["X"]
    y = loaded["y"]
    source_split_tag = loaded["source_split_tag"]

    print(f"Dataset size: {len(y)} pairs ({int(np.sum(y))} positive, {int(len(y) - np.sum(y))} negative)")
    print(f"Selected features: {len(loaded['selected_features'])}")

    results = {}
    skipped = {}

    if args.eval_protocol in ("random", "both"):
        results["random"] = evaluate_random_split(
            model=model,
            X=X,
            y=y,
            threshold=args.threshold,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    if args.eval_protocol in ("source_holdout", "both"):
        source_result, reason = evaluate_source_holdout(
            model=model,
            X=X,
            y=y,
            source_split_tag=source_split_tag,
            threshold=args.threshold,
        )
        if source_result is None:
            skipped["source_holdout"] = reason
        else:
            results["source_holdout"] = source_result

    if not results:
        raise RuntimeError(
            f"No evaluation protocol executed successfully; skipped reasons: {skipped}"
        )

    if skipped:
        for protocol, reason in skipped.items():
            print(f"Skipped {protocol}: {reason}")

    all_metrics = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "eval_protocol": args.eval_protocol,
        "threshold": float(args.threshold),
        "results": {},
        "skipped": skipped,
    }

    for protocol, payload in results.items():
        print_metrics_summary(protocol, payload)
        plot_name = "evaluation_results.png" if len(results) == 1 else f"evaluation_results_{protocol}.png"
        plot_path = output_dir / plot_name
        title = "random split" if protocol == "random" else "source holdout"
        plot_evaluation_results(payload["metrics"], plot_path, title)
        print(f"Saved plot: {plot_path}")

        m = payload["metrics"]
        all_metrics["results"][protocol] = {
            "roc_auc": m["roc_auc"],
            "avg_precision": m["avg_precision"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "counts": m["counts"],
            "metadata": payload["metadata"],
        }

    metrics_path = output_dir / "evaluation_results_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
