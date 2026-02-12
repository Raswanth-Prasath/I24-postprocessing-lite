#!/usr/bin/env python3
"""
Build a more diverse LR training dataset by combining advanced + curated v4.

Design goals:
- Keep compatibility with existing loaders (X, y, feature_names).
- Reduce shortcut bias from overly easy v4 negatives.
- Preserve class balance while increasing sample diversity.

Outputs:
- NPZ dataset with backward-compatible keys and extra metadata arrays.
- JSON summary with hardness/diversity diagnostics.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADVANCED = "Logistic Regression/training_dataset_advanced.npz"
DEFAULT_V4 = "Logistic Regression/training_dataset_v4.npz"
DEFAULT_MODEL = "Logistic Regression/model_artifacts/consensus_top10_full47.pkl"
DEFAULT_OUTPUT = "Logistic Regression/data/training_dataset_v5_diverse.npz"
DEFAULT_REPORT = "Logistic Regression/reports/training_dataset_v5_diverse_report.json"


def resolve_existing_path(path_str, fallbacks=None):
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


def load_npz(path):
    data = np.load(str(path), allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    feature_names = [str(x) for x in data["feature_names"]]
    return X, y, feature_names


def score_v4_samples(v4_X, feature_names, model_path):
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    scaler = artifact["scaler"]
    selected_features = artifact.get("feature_names") or artifact.get("features")
    if selected_features is None:
        raise ValueError("Model artifact is missing selected feature names")

    indices = [feature_names.index(str(f)) for f in selected_features]
    X_sel = v4_X[:, indices]
    X_scaled = scaler.transform(X_sel)
    prob = model.predict_proba(X_scaled)[:, 1]
    return prob


def stratification_keys(X, feature_names, indices, n_bins=8):
    """
    Build stable stratification keys from a small, high-signal feature set.
    This approximates scenario/lane/time diversity when explicit metadata is unavailable.
    """
    preferred = [
        "time_gap",
        "y_diff",
        "projection_error_x_max",
        "length_diff",
        "width_diff",
        "height_diff",
        "bhattacharyya_coeff",
    ]
    chosen = [f for f in preferred if f in feature_names]
    if not chosen:
        return np.array(["all"] * len(indices), dtype=object)

    local_X = X[indices]
    parts = []
    for f in chosen:
        col = local_X[:, feature_names.index(f)].astype(float)
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(col, q)
        edges = np.unique(edges)
        if len(edges) <= 2:
            # Constant-ish feature in this slice.
            b = np.zeros(len(col), dtype=int)
        else:
            b = np.digitize(col, edges[1:-1], right=True)
        parts.append(b)

    key_cols = np.vstack(parts).T
    keys = np.array(["|".join(map(str, row.tolist())) for row in key_cols], dtype=object)
    return keys


def diverse_sample(indices, X, feature_names, k, rng, n_bins=8):
    if k <= 0 or len(indices) == 0:
        return np.array([], dtype=int)
    if len(indices) <= k:
        return np.array(indices, dtype=int)

    indices = np.asarray(indices, dtype=int)
    keys = stratification_keys(X, feature_names, indices, n_bins=n_bins)

    # Build key -> member positions
    key_to_pos = {}
    for pos, key in enumerate(keys):
        key_to_pos.setdefault(key, []).append(pos)

    # Coverage pass: sample at least one from as many bins as possible (rarest first)
    selected_pos = []
    ordered_bins = sorted(key_to_pos.items(), key=lambda kv: len(kv[1]))
    for _, positions in ordered_bins:
        if len(selected_pos) >= k:
            break
        pick = int(rng.choice(positions))
        selected_pos.append(pick)

    selected_pos = np.array(sorted(set(selected_pos)), dtype=int)
    if len(selected_pos) >= k:
        selected_pos = selected_pos[:k]
        return indices[selected_pos]

    # Weighted fill: prefer sparse bins to improve tail coverage
    remaining_mask = np.ones(len(indices), dtype=bool)
    remaining_mask[selected_pos] = False
    remaining_pos = np.where(remaining_mask)[0]
    remaining_keys = keys[remaining_pos]

    counts = {k_: len(v) for k_, v in key_to_pos.items()}
    weights = np.array([1.0 / np.sqrt(float(counts[k_])) for k_ in remaining_keys], dtype=float)
    weights = weights / weights.sum()

    need = k - len(selected_pos)
    chosen = rng.choice(remaining_pos, size=need, replace=False, p=weights)
    final_pos = np.concatenate([selected_pos, np.asarray(chosen, dtype=int)])
    return indices[final_pos]


def bucket_by_hardness(y, prob, hard_quantile=0.25, medium_quantile=0.60):
    """
    Class-aware hardness buckets based on model difficulty.

    For class 1 (positive): difficulty = 1 - p(match)
    For class 0 (negative): difficulty = p(match)

    Within each class:
    - hard   = top `hard_quantile` hardest examples
    - medium = next bucket up to `medium_quantile`
    - easy   = remainder

    Misclassified samples are always forced into `hard`.
    """
    pred = (prob >= 0.5).astype(int)
    margin = np.abs(prob - 0.5)
    mis = pred != y

    # Difficulty score: higher means harder.
    difficulty = np.where(y == 1, 1.0 - prob, prob).astype(float)

    hard = np.zeros(len(y), dtype=bool)
    medium = np.zeros(len(y), dtype=bool)
    easy = np.zeros(len(y), dtype=bool)

    for klass in [0, 1]:
        idx = np.where(y == klass)[0]
        if len(idx) == 0:
            continue

        d = difficulty[idx]
        hard_thr = np.quantile(d, 1.0 - hard_quantile)
        med_thr = np.quantile(d, 1.0 - medium_quantile)

        cls_hard = d >= hard_thr
        cls_medium = (d >= med_thr) & (~cls_hard)
        cls_easy = (~cls_hard) & (~cls_medium)

        hard[idx] = cls_hard
        medium[idx] = cls_medium
        easy[idx] = cls_easy

    # Always include outright errors in hard set.
    hard = hard | mis
    medium = medium & (~mis)
    easy = easy & (~mis)

    return hard, medium, easy, margin


def sample_v4_class(indices, bucket_masks_local, target_count, X, feature_names, rng, n_bins=8,
                    hard_ratio=0.6, medium_ratio=0.3, easy_ratio=0.1):
    hard_m, medium_m, easy_m = bucket_masks_local
    idx_hard = indices[hard_m]
    idx_medium = indices[medium_m]
    idx_easy = indices[easy_m]

    target_h = int(round(target_count * hard_ratio))
    target_m = int(round(target_count * medium_ratio))
    target_e = max(0, target_count - target_h - target_m)

    take_h = diverse_sample(idx_hard, X, feature_names, min(target_h, len(idx_hard)), rng, n_bins=n_bins)
    take_m = diverse_sample(idx_medium, X, feature_names, min(target_m, len(idx_medium)), rng, n_bins=n_bins)
    take_e = diverse_sample(idx_easy, X, feature_names, min(target_e, len(idx_easy)), rng, n_bins=n_bins)

    selected = np.concatenate([take_h, take_m, take_e])
    selected = np.array(sorted(set(selected.tolist())), dtype=int)

    if len(selected) < target_count:
        deficit = target_count - len(selected)
        rem = np.setdiff1d(indices, selected, assume_unique=False)
        fill = diverse_sample(rem, X, feature_names, min(deficit, len(rem)), rng, n_bins=n_bins)
        selected = np.concatenate([selected, fill])
        selected = np.array(sorted(set(selected.tolist())), dtype=int)

    if len(selected) > target_count:
        selected = diverse_sample(selected, X, feature_names, target_count, rng, n_bins=n_bins)

    return selected


def single_feature_auc_report(X, y, feature_names, topk=12):
    rows = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        try:
            auc = float(roc_auc_score(y, col))
            auc = max(auc, 1.0 - auc)
        except Exception:
            auc = 0.5
        rows.append({"feature": name, "single_feature_auc": auc})
    rows.sort(key=lambda r: r["single_feature_auc"], reverse=True)
    return rows[:topk]


def build_dataset(args):
    rng = np.random.default_rng(args.seed)

    adv_path = resolve_existing_path(args.advanced_dataset, [DEFAULT_ADVANCED])
    v4_path = resolve_existing_path(args.v4_dataset, [DEFAULT_V4])
    model_path = resolve_existing_path(args.model_path, [DEFAULT_MODEL])

    adv_X, adv_y, adv_features = load_npz(adv_path)
    v4_X, v4_y, v4_features = load_npz(v4_path)

    if adv_features != v4_features:
        raise ValueError("Feature schema mismatch between advanced and v4 datasets")

    feature_names = adv_features

    # Score v4 with current production LR to identify hard/medium/easy examples.
    v4_prob = score_v4_samples(v4_X, feature_names, model_path)
    hard_m, med_m, easy_m, margin = bucket_by_hardness(
        v4_y,
        v4_prob,
        hard_quantile=args.hard_quantile,
        medium_quantile=args.medium_quantile,
    )

    # Balanced target from v4 (per class), defaults to 2x advanced-per-class.
    adv_pos = int((adv_y == 1).sum())
    adv_neg = int((adv_y == 0).sum())
    if adv_pos != adv_neg:
        raise ValueError("Advanced dataset must be balanced for this builder")

    if args.target_v4_per_class is None:
        target_v4_per_class = adv_pos * 2
    else:
        target_v4_per_class = int(args.target_v4_per_class)

    selected_v4 = []

    for klass in [0, 1]:
        cls_idx = np.where(v4_y == klass)[0]
        cls_h = hard_m[cls_idx]
        cls_m = med_m[cls_idx]
        cls_e = easy_m[cls_idx]

        chosen = sample_v4_class(
            cls_idx,
            (cls_h, cls_m, cls_e),
            target_count=target_v4_per_class,
            X=v4_X,
            feature_names=feature_names,
            rng=rng,
            n_bins=args.n_bins,
            hard_ratio=args.hard_ratio,
            medium_ratio=args.medium_ratio,
            easy_ratio=args.easy_ratio,
        )
        selected_v4.extend(chosen.tolist())

    selected_v4 = np.array(sorted(set(selected_v4)), dtype=int)

    # Label bucket for selected rows from global masks.
    selected_bucket = np.empty(len(selected_v4), dtype=object)
    selected_bucket[:] = "easy"
    selected_bucket[hard_m[selected_v4]] = "hard"
    selected_bucket[med_m[selected_v4]] = "medium"

    sel_X = v4_X[selected_v4]
    sel_y = v4_y[selected_v4]
    sel_prob = v4_prob[selected_v4]
    sel_margin = margin[selected_v4]

    # Combine with full advanced dataset (keep all advanced for stability).
    X = np.vstack([adv_X.astype(np.float32), sel_X.astype(np.float32)])
    y = np.hstack([adv_y.astype(int), sel_y.astype(int)])

    n_adv = len(adv_y)
    n_sel = len(sel_y)

    # Extra metadata arrays (optional; backward compatible).
    source_dataset = np.array(["advanced"] * n_adv + ["v4"] * n_sel, dtype=object)
    source_split_tag = np.array(["advanced_keepall"] * n_adv + ["v4_diverse_curated"] * n_sel, dtype=object)
    scenario = np.array(["unknown"] * (n_adv + n_sel), dtype=object)
    mask_idx = np.array([-1] * (n_adv + n_sel), dtype=np.int32)
    hardness_bucket = np.array(["advanced"] * n_adv + selected_bucket.tolist(), dtype=object)
    v4_probability = np.hstack([np.full(n_adv, np.nan, dtype=float), sel_prob.astype(float)])
    v4_margin = np.hstack([np.full(n_adv, np.nan, dtype=float), sel_margin.astype(float)])

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        feature_names=np.array(feature_names),
        source_dataset=source_dataset,
        source_split_tag=source_split_tag,
        scenario=scenario,
        mask_idx=mask_idx,
        hardness_bucket=hardness_bucket,
        v4_probability=v4_probability,
        v4_margin=v4_margin,
    )

    # Build report.
    out_pos = int((y == 1).sum())
    out_neg = int((y == 0).sum())
    report = {
        "paths": {
            "advanced_dataset": str(adv_path),
            "v4_dataset": str(v4_path),
            "model_path": str(model_path),
            "output_dataset": str(output_path),
        },
        "params": {
            "seed": args.seed,
            "target_v4_per_class": target_v4_per_class,
            "hard_ratio": args.hard_ratio,
            "medium_ratio": args.medium_ratio,
            "easy_ratio": args.easy_ratio,
            "hard_quantile": args.hard_quantile,
            "medium_quantile": args.medium_quantile,
            "n_bins": args.n_bins,
        },
        "input_stats": {
            "advanced": {
                "n": int(len(adv_y)),
                "pos": int((adv_y == 1).sum()),
                "neg": int((adv_y == 0).sum()),
            },
            "v4": {
                "n": int(len(v4_y)),
                "pos": int((v4_y == 1).sum()),
                "neg": int((v4_y == 0).sum()),
                "hard_count": int(hard_m.sum()),
                "medium_count": int(med_m.sum()),
                "easy_count": int(easy_m.sum()),
            },
        },
        "selected_v4_stats": {
            "n": int(n_sel),
            "pos": int((sel_y == 1).sum()),
            "neg": int((sel_y == 0).sum()),
            "hard": int((selected_bucket == "hard").sum()),
            "medium": int((selected_bucket == "medium").sum()),
            "easy": int((selected_bucket == "easy").sum()),
            "misclassified_fraction": float(np.mean((sel_prob >= 0.5).astype(int) != sel_y)),
            "margin_mean": float(np.mean(sel_margin)),
            "margin_std": float(np.std(sel_margin)),
        },
        "output_stats": {
            "n": int(len(y)),
            "pos": out_pos,
            "neg": out_neg,
            "feature_count": int(X.shape[1]),
            "is_balanced": bool(out_pos == out_neg),
        },
        "top_single_feature_auc": {
            "advanced": single_feature_auc_report(adv_X, adv_y, feature_names),
            "v4": single_feature_auc_report(v4_X, v4_y, feature_names),
            "v5_diverse": single_feature_auc_report(X, y, feature_names),
        },
    }

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("DIVERSE DATASET BUILD COMPLETE")
    print("=" * 72)
    print(f"Output dataset : {output_path}")
    print(f"Output report  : {report_path}")
    print(f"Output shape   : {X.shape}")
    print(f"Class balance  : pos={out_pos}, neg={out_neg}")
    print(f"Selected v4    : {n_sel} rows")
    print(
        "Selected v4 buckets: "
        f"hard={(selected_bucket == 'hard').sum()}, "
        f"medium={(selected_bucket == 'medium').sum()}, "
        f"easy={(selected_bucket == 'easy').sum()}"
    )

    return output_path, report_path


def main():
    parser = argparse.ArgumentParser(description="Build a diverse v5 LR dataset from advanced + v4")
    parser.add_argument("--advanced-dataset", default=DEFAULT_ADVANCED,
                        help="Path to training_dataset_advanced.npz")
    parser.add_argument("--v4-dataset", default=DEFAULT_V4,
                        help="Path to training_dataset_v4.npz")
    parser.add_argument("--model-path", default=DEFAULT_MODEL,
                        help="Model used for hardness scoring")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output NPZ path")
    parser.add_argument("--report", default=DEFAULT_REPORT,
                        help="Output JSON report path")
    parser.add_argument("--target-v4-per-class", type=int, default=None,
                        help="Target curated v4 samples per class (default: 2x advanced per class)")
    parser.add_argument("--hard-ratio", type=float, default=0.60,
                        help="Fraction of selected v4 samples from hard bucket")
    parser.add_argument("--medium-ratio", type=float, default=0.30,
                        help="Fraction of selected v4 samples from medium bucket")
    parser.add_argument("--easy-ratio", type=float, default=0.10,
                        help="Fraction of selected v4 samples from easy bucket")
    parser.add_argument("--hard-quantile", type=float, default=0.25,
                        help="Per-class hardest fraction assigned to hard bucket")
    parser.add_argument("--medium-quantile", type=float, default=0.60,
                        help="Per-class cumulative hardest fraction assigned to medium+hard buckets")
    parser.add_argument("--n-bins", type=int, default=8,
                        help="Quantile bins per stratification feature")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    total_ratio = args.hard_ratio + args.medium_ratio + args.easy_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError("hard_ratio + medium_ratio + easy_ratio must sum to 1.0")
    if not (0.0 < args.hard_quantile < args.medium_quantile < 1.0):
        raise ValueError("Require 0 < hard_quantile < medium_quantile < 1")

    build_dataset(args)


if __name__ == "__main__":
    main()
