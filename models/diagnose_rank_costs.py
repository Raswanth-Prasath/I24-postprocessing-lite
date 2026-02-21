#!/usr/bin/env python
"""Diagnostic: measure output distributions of ranking checkpoints (Transformer/PINN).

Loads the ranking dataset (selected split), runs the model on all pairs,
and prints score/cost distributions + suggested thresholds.

Usage:
    conda activate i24
    python models/diagnose_rank_costs.py
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from transformer_model import SiameseTransformerNetwork
from rich_sequence_dataset import extract_rich_sequence, extract_endpoint_features


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return p


def load_isotonic_calibration(path: Optional[str]) -> Optional[Dict[str, Any]]:
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        return None
    with open(resolved, "r") as f:
        artifact = json.load(f)

    x_knots = np.asarray(artifact.get("x_knots", []), dtype=np.float64)
    y_knots = np.asarray(artifact.get("y_knots", []), dtype=np.float64)
    if len(x_knots) < 2 or len(y_knots) != len(x_knots):
        raise ValueError("Calibration artifact must provide matching x_knots/y_knots with >=2 points.")

    order = np.argsort(x_knots)
    x_knots = x_knots[order]
    y_knots = y_knots[order]
    x_knots, unique_idx = np.unique(x_knots, return_index=True)
    y_knots = y_knots[unique_idx]
    if len(x_knots) < 2:
        raise ValueError("Calibration x_knots are not sufficiently distinct.")

    domain = artifact.get("domain", [float(x_knots[0]), float(x_knots[-1])])
    if not isinstance(domain, (list, tuple)) or len(domain) != 2:
        domain = [float(x_knots[0]), float(x_knots[-1])]
    domain = [float(domain[0]), float(domain[1])]
    if domain[0] >= domain[1]:
        domain = [float(x_knots[0]), float(x_knots[-1])]

    artifact["x_knots"] = x_knots
    artifact["y_knots"] = y_knots
    artifact["domain"] = domain
    artifact["_resolved_path"] = str(resolved)
    return artifact


def apply_isotonic_calibration(raw_cost: float, calibration: Optional[Dict[str, Any]]) -> float:
    if calibration is None:
        return float(raw_cost)
    x_knots = calibration["x_knots"]
    y_knots = calibration["y_knots"]
    dmin, dmax = calibration["domain"]
    clipped = float(np.clip(raw_cost, dmin, dmax))
    return float(np.interp(clipped, x_knots, y_knots))


def build_threshold_sweep(costs_pos: np.ndarray) -> np.ndarray:
    if len(costs_pos) == 0:
        return np.asarray([], dtype=np.float64)
    p95 = float(np.percentile(costs_pos, 95))
    cmin = float(np.min(costs_pos))
    if cmin >= 0.0 and p95 <= 10.0:
        return np.asarray([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0], dtype=np.float64)
    candidates = np.asarray(
        [
            np.percentile(costs_pos, 50),
            np.percentile(costs_pos, 75),
            np.percentile(costs_pos, 90),
            np.percentile(costs_pos, 95),
            np.percentile(costs_pos, 99),
        ],
        dtype=np.float64,
    )
    return np.unique(candidates)


def load_model_and_stats(checkpoint_path, device, model_type: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = str(model_type).lower()

    if model_type == "auto":
        if ckpt.get("model_type") == "pinn":
            model_type = "pinn"
        else:
            model_type = "transformer"

    if model_type == "pinn":
        from pinn_model import PhysicsInformedCostNetwork

        model_config = ckpt.get("model_config", {})
        model = PhysicsInformedCostNetwork(**model_config).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
    else:
        model_config = ckpt.get("model_config", {})
        model_config["training_objective"] = "ranking"
        model = SiameseTransformerNetwork(**model_config).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

    seq_mean = torch.as_tensor(ckpt.get("seq_mean", np.zeros(8)), device=device, dtype=torch.float32)
    seq_std = torch.as_tensor(ckpt.get("seq_std", np.ones(8)), device=device, dtype=torch.float32).clamp_min(1e-6)
    ep_mean = torch.as_tensor(ckpt.get("ep_mean", np.zeros(4)), device=device, dtype=torch.float32)
    ep_std = torch.as_tensor(ckpt.get("ep_std", np.ones(4)), device=device, dtype=torch.float32).clamp_min(1e-6)

    train_cfg = ckpt.get("train_config", {})
    runtime_cfg = {
        "dt_floor": float(train_cfg.get("dt_floor", 0.04)),
        "accel_limit": float(train_cfg.get("accel_limit", 15.0)),
        "lane_tolerance": float(train_cfg.get("lane_tolerance", 6.0)),
        "min_points_for_fit": int(train_cfg.get("min_points_for_fit", 3)),
        "time_win": float(train_cfg.get("time_win", 15.0)),
    }

    return model, seq_mean, seq_std, ep_mean, ep_std, model_type, runtime_cfg


def load_dataset_pairs(dataset_path, fragments_path, split="val"):
    """Load all pairs from the ranking dataset JSONL."""
    # Load fragment store
    fragment_store = {}
    if fragments_path.exists():
        with open(fragments_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                fragment_store[str(row["fragment_ref"])] = row["fragment"]

    pairs = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("split", "train")) != split:
                continue

            # Resolve fragments
            if "track_candidate" in row and "track_anchor" in row:
                fa, fb = row["track_candidate"], row["track_anchor"]
            else:
                cand_ref = row.get("candidate_ref")
                anchor_ref = row.get("anchor_ref")
                fa = fragment_store[str(cand_ref)]
                fb = fragment_store[str(anchor_ref)]

            pairs.append({
                "fa": fa,
                "fb": fb,
                "gt_label": row.get("gt_label", -1),
                "bhat_cost": row.get("bhat_cost", float("nan")),
                "anchor_key": row.get("anchor_key", ""),
            })

    return pairs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose ranking-model cost distributions")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_ranking_model_full80.pth"),
                        help="Path to checkpoint")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_ranking_dataset.jsonl"),
                        help="Path to ranking dataset JSONL")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "transformer", "pinn"],
        help="Model type for checkpoint.",
    )
    parser.add_argument(
        "--score-mapping",
        default="physics_total",
        choices=["physics_total", "neg_logit", "softplus_neg_logit"],
        help="PINN-only mapping from model output to base cost.",
    )
    parser.add_argument(
        "--logit-temperature",
        type=float,
        default=1.0,
        help="PINN-only divisor for centered gate logits before mapping.",
    )
    parser.add_argument(
        "--logit-bias",
        type=float,
        default=0.0,
        help="PINN-only bias subtracted from gate logits before mapping.",
    )
    parser.add_argument(
        "--time-penalty",
        type=float,
        default=0.0,
        help="PINN-only explicit additive penalty: base_cost + time_penalty * gap.",
    )
    parser.add_argument(
        "--calibration-mode",
        default="off",
        choices=["off", "linear", "isotonic"],
        help="PINN-only calibration mode for final cost.",
    )
    parser.add_argument(
        "--calibration-path",
        default="",
        help="PINN-only isotonic calibration artifact JSON path.",
    )
    args = parser.parse_args()

    if args.logit_temperature <= 0:
        print(f"Warning: --logit-temperature={args.logit_temperature} must be > 0. Using 1e-6.")
        args.logit_temperature = 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    fragments_path = dataset_path.parent / f"{dataset_path.stem}.fragments.jsonl"

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print()

    model, seq_mean, seq_std, ep_mean, ep_std, resolved_model_type, runtime_cfg = load_model_and_stats(
        checkpoint_path, device, model_type=args.model_type
    )
    pairs = load_dataset_pairs(dataset_path, fragments_path, split=args.split)
    print(f"Loaded {len(pairs)} {args.split} pairs")
    print(f"Resolved model type: {resolved_model_type}")
    if resolved_model_type == "pinn":
        print(
            "PINN runtime: "
            f"score_mapping={args.score_mapping}, "
            f"logit_temperature={args.logit_temperature}, "
            f"logit_bias={args.logit_bias}, "
            f"time_penalty={args.time_penalty}, "
            f"calibration_mode={args.calibration_mode}"
        )
    print()

    raw_scores = []
    sp_costs = []
    gt_labels = []
    bhat_costs = []

    if resolved_model_type == "pinn":
        from physics_residuals import (
            apply_robust_transforms,
            compute_fragment_stats,
            compute_physics_residuals,
            get_fragment_cache_key,
        )

        stats_cache: Dict[str, Any] = {}

        def get_stats(track: dict):
            key = get_fragment_cache_key(track)
            stats = stats_cache.get(key)
            if stats is None:
                stats = compute_fragment_stats(
                    track,
                    dt_floor=runtime_cfg["dt_floor"],
                    min_points_for_fit=runtime_cfg["min_points_for_fit"],
                )
                stats_cache[key] = stats
            return stats

        pinn_calibration = None
        if args.calibration_mode == "isotonic":
            try:
                pinn_calibration = load_isotonic_calibration(args.calibration_path or None)
                if pinn_calibration is None:
                    print(
                        "Warning: isotonic calibration requested but artifact not found. "
                        "Falling back to linear/off behavior."
                    )
                else:
                    print(f"Loaded PINN calibration from {pinn_calibration['_resolved_path']}")
            except Exception as exc:
                print(f"Warning: failed to load PINN calibration artifact: {exc}. Falling back to linear/off.")
                pinn_calibration = None

    with torch.no_grad():
        for i, p in enumerate(pairs):
            try:
                seq_a = extract_rich_sequence(p["fa"])
                seq_b = extract_rich_sequence(p["fb"])
                ep = extract_endpoint_features(p["fa"], p["fb"])

                seq_a_t = (torch.as_tensor(seq_a, dtype=torch.float32, device=device) - seq_mean) / seq_std
                seq_b_t = (torch.as_tensor(seq_b, dtype=torch.float32, device=device) - seq_mean) / seq_std
                ep_t = (torch.as_tensor(ep, dtype=torch.float32, device=device) - ep_mean) / ep_std

                seq_a_t = seq_a_t.unsqueeze(0)
                seq_b_t = seq_b_t.unsqueeze(0)
                ep_t = ep_t.unsqueeze(0)
                if resolved_model_type == "pinn":
                    gap = float(p["fb"]["timestamp"][0] - p["fa"]["timestamp"][-1])
                    sa = get_stats(p["fa"])
                    sb = get_stats(p["fb"])
                    residuals_raw = compute_physics_residuals(
                        sa,
                        sb,
                        time_win=runtime_cfg["time_win"],
                        dt_floor=runtime_cfg["dt_floor"],
                        accel_limit=runtime_cfg["accel_limit"],
                        lane_tolerance=runtime_cfg["lane_tolerance"],
                    )
                    residuals = apply_robust_transforms(residuals_raw)
                    residuals_t = torch.as_tensor(
                        residuals, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    mask_a = torch.zeros(1, seq_a_t.size(1), dtype=torch.bool, device=device)
                    mask_b = torch.zeros(1, seq_b_t.size(1), dtype=torch.bool, device=device)
                    if args.score_mapping == "physics_total":
                        base_cost = float(
                            model.inference_cost(seq_a_t, mask_a, seq_b_t, mask_b, ep_t, residuals_t).item()
                        )
                    else:
                        _total_cost, _aux, _weights, gate_logits = model(
                            seq_a_t,
                            mask_a,
                            seq_b_t,
                            mask_b,
                            ep_t,
                            residuals_t,
                            return_gate_logits=True,
                        )
                        gate_logit = float(gate_logits.item())
                        centered = (gate_logit - float(args.logit_bias)) / max(float(args.logit_temperature), 1e-6)
                        neg_logit = -centered
                        if args.score_mapping == "neg_logit":
                            base_cost = float(neg_logit)
                        else:
                            base_cost = float(softplus(neg_logit))

                    raw_total_cost = float(base_cost + float(args.time_penalty) * gap)
                    if args.calibration_mode == "isotonic" and pinn_calibration is not None:
                        final_cost = apply_isotonic_calibration(raw_total_cost, pinn_calibration)
                    else:
                        final_cost = raw_total_cost
                    raw_scores.append(float(base_cost))
                    sp_costs.append(float(final_cost))
                else:
                    score = model(seq_a_t, None, seq_b_t, None, ep_t).item()
                    raw_scores.append(score)
                    sp_costs.append(float(softplus(score)))
                gl = p["gt_label"]
                gt_labels.append(int(gl) if gl is not None else -1)
                bc = p["bhat_cost"]
                bhat_costs.append(float(bc) if bc is not None else float("nan"))
            except Exception as e:
                if i < 5:
                    print(f"  Pair {i} error: {e}")
                continue

    raw_scores = np.array(raw_scores)
    sp_costs = np.array(sp_costs)
    gt_labels = np.array(gt_labels)
    bhat_costs = np.array(bhat_costs)

    pos_mask = gt_labels == 1
    neg_mask = gt_labels == 0
    unk_mask = ~(pos_mask | neg_mask)

    print(f"\nArray sizes: raw_scores={len(raw_scores)}, gt_labels={len(gt_labels)}, sp_costs={len(sp_costs)}, bhat_costs={len(bhat_costs)}")
    print(f"Processed {len(raw_scores)} pairs ({pos_mask.sum()} positive, {neg_mask.sum()} negative, {unk_mask.sum()} unknown)")

    if resolved_model_type == "transformer":
        # --- Raw scores ---
        print("\n" + "=" * 60)
        print("RAW MODEL SCORES (before softplus)")
        print("=" * 60)
        for label, mask, name in [(1, pos_mask, "POSITIVE"), (0, neg_mask, "NEGATIVE")]:
            s = raw_scores[mask]
            if len(s) == 0:
                continue
            print(f"\n  {name} (n={len(s)}):")
            print(f"    min={s.min():.4f}  p5={np.percentile(s,5):.4f}  p25={np.percentile(s,25):.4f}  "
                  f"median={np.median(s):.4f}  p75={np.percentile(s,75):.4f}  p95={np.percentile(s,95):.4f}  max={s.max():.4f}")

    # --- Costs seen by MCF ---
    print("\n" + "=" * 60)
    if resolved_model_type == "transformer":
        print("SOFTPLUS COSTS (what MCF sees as rank_cost)")
    else:
        print("PINN COSTS (final MCF costs after selected mapping/calibration)")
    print("=" * 60)
    for label, mask, name in [(1, pos_mask, "POSITIVE"), (0, neg_mask, "NEGATIVE")]:
        c = sp_costs[mask]
        if len(c) == 0:
            continue
        print(f"\n  {name} (n={len(c)}):")
        print(f"    min={c.min():.4f}  p5={np.percentile(c,5):.4f}  p25={np.percentile(c,25):.4f}  "
              f"median={np.median(c):.4f}  p75={np.percentile(c,75):.4f}  p95={np.percentile(c,95):.4f}  max={c.max():.4f}")

    # --- Bhattacharyya costs for comparison ---
    valid_bhat = np.isfinite(bhat_costs)
    print("\n" + "=" * 60)
    print("BHATTACHARYYA COSTS (for comparison)")
    print("=" * 60)
    for label, mask, name in [(1, pos_mask & valid_bhat, "POSITIVE"), (0, neg_mask & valid_bhat, "NEGATIVE")]:
        b = bhat_costs[mask]
        if len(b) == 0:
            continue
        print(f"\n  {name} (n={len(b)}):")
        print(f"    min={b.min():.4f}  p5={np.percentile(b,5):.4f}  p25={np.percentile(b,25):.4f}  "
              f"median={np.median(b):.4f}  p75={np.percentile(b,75):.4f}  p95={np.percentile(b,95):.4f}  max={b.max():.4f}")

    # --- Overlap analysis ---
    print("\n" + "=" * 60)
    print("SEPARATION ANALYSIS")
    print("=" * 60)
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        pos_sp = sp_costs[pos_mask]
        neg_sp = sp_costs[neg_mask]
        overlap_lo = max(pos_sp.min(), neg_sp.min())
        overlap_hi = min(pos_sp.max(), neg_sp.max())
        print(f"\n  MCF cost ranges:")
        print(f"    Positive: [{pos_sp.min():.4f}, {pos_sp.max():.4f}]")
        print(f"    Negative: [{neg_sp.min():.4f}, {neg_sp.max():.4f}]")
        if overlap_lo < overlap_hi:
            print(f"    Overlap zone: [{overlap_lo:.4f}, {overlap_hi:.4f}]")
        else:
            print(f"    No overlap — clean separation!")

        # Suggested thresholds from positive percentiles
        print(f"\n  Suggested stitch_thresh (from positive percentiles):")
        for pct in [90, 95, 99]:
            thresh = np.percentile(pos_sp, pct)
            tp_rate = (pos_sp <= thresh).mean()
            fp_rate = (neg_sp <= thresh).mean()
            print(f"    p{pct}(positive)={thresh:.4f}  →  TP={tp_rate:.1%}, FP={fp_rate:.1%}")

        # Fixed threshold sweep
        sweep_thresholds = build_threshold_sweep(pos_sp)
        print(f"\n  Threshold sweep:")
        print(f"    {'thresh':>12s}  {'TP%':>6s}  {'FP%':>6s}  {'precision':>9s}  {'avg_pos_margin':>14s}")
        for thresh in sweep_thresholds:
            tp_rate = (pos_sp <= thresh).mean()
            fp_rate = (neg_sp <= thresh).mean()
            tp_count = (pos_sp <= thresh).sum()
            fp_count = (neg_sp <= thresh).sum()
            precision = tp_count / max(tp_count + fp_count, 1)
            avg_pos_margin = np.mean(np.clip(thresh - pos_sp[pos_sp <= thresh], 0, None)) if tp_count > 0 else 0
            print(f"    {thresh:>12.4f}  {tp_rate:>5.1%}  {fp_rate:>5.1%}  {precision:>9.3f}  {avg_pos_margin:>14.3f}")


if __name__ == "__main__":
    main()
