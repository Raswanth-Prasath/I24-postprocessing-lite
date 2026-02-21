#!/usr/bin/env python
"""
Train Physics-Informed Transformer Cost Network for trajectory stitching.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.physics_residuals import (  # noqa: E402
    RESIDUAL_DIM,
    apply_robust_transforms,
    compute_fragment_stats,
    compute_physics_residuals,
    get_fragment_cache_key,
)
from models.pinn_model import PhysicsInformedCostNetwork  # noqa: E402
from models.rich_sequence_dataset import extract_endpoint_features, extract_rich_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PINN stitching model.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_ranking_dataset.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "pinn_model.pth"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--anchors-per-batch", type=int, default=12)
    parser.add_argument("--cands-per-anchor", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--norm-sample-pairs", type=int, default=2000)

    parser.add_argument("--alpha", type=float, default=0.01, help="Physics auxiliary loss weight.")
    parser.add_argument("--beta", type=float, default=1.0, help="Pairwise ranking loss weight.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gate BCE loss weight.")
    parser.add_argument(
        "--gate-pos-weight",
        type=float,
        default=1.0,
        help="Positive-class weight for gate BCEWithLogits.",
    )
    parser.add_argument("--margin", type=float, default=1.0, help="Pairwise ranking margin.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Gate sigmoid temperature.")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="composite",
        choices=["top1", "composite"],
        help="Validation metric used for best-checkpoint selection and early stopping.",
    )
    parser.add_argument("--selection-weight-top1", type=float, default=0.55)
    parser.add_argument("--selection-weight-pairwise", type=float, default=0.25)
    parser.add_argument("--selection-weight-spearman", type=float, default=0.15)
    parser.add_argument("--selection-weight-gate", type=float, default=0.05)

    parser.add_argument("--time-win", type=float, default=15.0)
    parser.add_argument("--dt-floor", type=float, default=0.04)
    parser.add_argument("--min-points-for-fit", type=int, default=3)
    parser.add_argument("--accel-limit", type=float, default=15.0)
    parser.add_argument("--lane-tolerance", type=float, default=6.0)

    parser.add_argument("--disable-correction", action="store_true", help="Disable correction head for ablation C.")

    # Encoder config
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--endpoint-dim", type=int, default=4)
    parser.add_argument("--pool-weight-first", type=float, default=0.2)
    parser.add_argument("--pool-weight-last", type=float, default=0.5)
    parser.add_argument("--pool-weight-mean", type=float, default=0.3)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rank_simple(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ra = _rank_simple(a)
    rb = _rank_simple(b)
    corr = np.corrcoef(ra, rb)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _metric_with_fallback(metrics: Mapping[str, Any], primary: str, fallback: str) -> float:
    return float(metrics.get(primary, metrics.get(fallback, 0.0)))


def _compute_model_selection_score(
    metrics: Mapping[str, Any],
    *,
    strategy: str = "composite",
    weight_top1: float = 0.55,
    weight_pairwise: float = 0.25,
    weight_spearman: float = 0.15,
    weight_gate: float = 0.05,
) -> float:
    top1 = _metric_with_fallback(metrics, "val_anchor_top1_acc", "anchor_top1_acc")
    if strategy == "top1":
        return top1
    if strategy != "composite":
        raise ValueError(f"Unsupported selection strategy: {strategy}")

    pairwise = _metric_with_fallback(
        metrics, "val_pairwise_acc_gt_crossclass", "pairwise_acc_gt_crossclass"
    )
    spearman = _metric_with_fallback(metrics, "val_mean_anchor_spearman", "mean_anchor_spearman")
    gate = _metric_with_fallback(metrics, "val_gate_acc_known", "gate_acc_known")
    denom = max(float(weight_top1 + weight_pairwise + weight_spearman + weight_gate), 1e-8)
    return float(
        (
            float(weight_top1) * top1
            + float(weight_pairwise) * pairwise
            + float(weight_spearman) * spearman
            + float(weight_gate) * gate
        )
        / denom
    )


@dataclass
class BatchStats:
    loss: float
    aux_term: float
    rank_term: float
    gate_term: float
    num_groups: int


class PINNAnchorDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        split: str,
        max_candidates: int,
        *,
        seed: int = 42,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        dt_floor: float = 0.04,
        min_points_for_fit: int = 3,
        accel_limit: float = 15.0,
        lane_tolerance: float = 6.0,
        time_win: float = 15.0,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = str(split)
        self.max_candidates = int(max_candidates)
        self.rng = np.random.default_rng(int(seed))
        self.norm_stats = norm_stats
        self.dt_floor = float(dt_floor)
        self.min_points_for_fit = int(min_points_for_fit)
        self.accel_limit = float(accel_limit)
        self.lane_tolerance = float(lane_tolerance)
        self.time_win = float(time_win)

        self.fragments_path = self.dataset_path.parent / f"{self.dataset_path.stem}.fragments.jsonl"
        self.fragment_store: Dict[str, Dict[str, Any]] = {}
        self._stats_cache: Dict[str, Any] = {}

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        with open(self.dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if str(row.get("split", "train")) != self.split:
                    continue
                grouped.setdefault(str(row["anchor_key"]), []).append(row)

        self.groups: List[List[Dict[str, Any]]] = [
            sorted(rows, key=lambda r: float(r.get("bhat_cost", 1e6)))
            for _, rows in sorted(grouped.items(), key=lambda kv: kv[0])
            if rows
        ]
        self._load_fragment_store()

    def _load_fragment_store(self) -> None:
        if not self.fragments_path.exists():
            return
        with open(self.fragments_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.fragment_store[str(row["fragment_ref"])] = row["fragment"]

    def __len__(self) -> int:
        return len(self.groups)

    def _resolve_pair_fragments(self, row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if "track_candidate" in row and "track_anchor" in row:
            return row["track_candidate"], row["track_anchor"]
        cand_ref = row.get("candidate_ref")
        anchor_ref = row.get("anchor_ref")
        if cand_ref is None or anchor_ref is None:
            raise KeyError("Row missing tracks and refs (candidate_ref/anchor_ref).")
        return self.fragment_store[str(cand_ref)], self.fragment_store[str(anchor_ref)]

    def _get_stats(self, track: Dict[str, Any]):
        key = get_fragment_cache_key(track)
        stats = self._stats_cache.get(key)
        if stats is None:
            stats = compute_fragment_stats(
                track,
                dt_floor=self.dt_floor,
                min_points_for_fit=self.min_points_for_fit,
            )
            self._stats_cache[key] = stats
        return stats

    def set_norm_stats(self, norm_stats: Dict[str, np.ndarray]) -> None:
        self.norm_stats = {
            "mean": np.asarray(norm_stats["mean"], dtype=np.float32),
            "std": np.asarray(norm_stats["std"], dtype=np.float32),
            "ep_mean": np.asarray(norm_stats["ep_mean"], dtype=np.float32),
            "ep_std": np.asarray(norm_stats["ep_std"], dtype=np.float32),
        }

    def _normalize_seq(self, seq: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return seq
        return (seq - self.norm_stats["mean"]) / self.norm_stats["std"]

    def _normalize_ep(self, ep: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return ep
        return (ep - self.norm_stats["ep_mean"]) / self.norm_stats["ep_std"]

    def compute_normalization_stats(self, sample_pairs: int = 2000) -> Dict[str, np.ndarray]:
        rows: List[Dict[str, Any]] = []
        for group in self.groups:
            rows.extend(group)
        if not rows:
            raise RuntimeError("Cannot compute normalization stats on empty split.")

        n = min(int(sample_pairs), len(rows))
        idx = self.rng.choice(len(rows), size=n, replace=False)

        seq_rows = []
        ep_rows = []
        for i in idx:
            row = rows[int(i)]
            fa, fb = self._resolve_pair_fragments(row)
            seq_rows.append(extract_rich_sequence(fa))
            seq_rows.append(extract_rich_sequence(fb))
            ep_rows.append(extract_endpoint_features(fa, fb))

        seq_data = np.vstack(seq_rows)
        ep_data = np.vstack(ep_rows)
        return {
            "mean": seq_data.mean(axis=0).astype(np.float32),
            "std": (seq_data.std(axis=0) + 1e-6).astype(np.float32),
            "ep_mean": ep_data.mean(axis=0).astype(np.float32),
            "ep_std": (ep_data.std(axis=0) + 1e-6).astype(np.float32),
        }

    def _select_candidates(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(rows) <= self.max_candidates:
            return rows
        if self.split != "train":
            return rows[: self.max_candidates]

        pos = [r for r in rows if r.get("gt_label") == 1]
        neg = [r for r in rows if r.get("gt_label") == 0]

        selected: List[Dict[str, Any]] = []
        if pos and neg:
            selected.append(pos[int(self.rng.integers(len(pos)))])
            selected.append(neg[int(self.rng.integers(len(neg)))])

        seen = {id(r) for r in selected}
        remaining = [r for r in rows if id(r) not in seen]
        self.rng.shuffle(remaining)
        selected.extend(remaining[: max(0, self.max_candidates - len(selected))])
        return sorted(selected, key=lambda r: float(r.get("bhat_cost", 1e6)))

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        rows = self._select_candidates(self.groups[idx])
        items: List[Dict[str, Any]] = []
        for row in rows:
            fa, fb = self._resolve_pair_fragments(row)
            seq_a = self._normalize_seq(extract_rich_sequence(fa)).astype(np.float32)
            seq_b = self._normalize_seq(extract_rich_sequence(fb)).astype(np.float32)
            ep = self._normalize_ep(extract_endpoint_features(fa, fb)).astype(np.float32)

            sa = self._get_stats(fa)
            sb = self._get_stats(fb)
            residuals_raw = compute_physics_residuals(
                sa,
                sb,
                time_win=self.time_win,
                dt_floor=self.dt_floor,
                accel_limit=self.accel_limit,
                lane_tolerance=self.lane_tolerance,
            )
            residuals = apply_robust_transforms(residuals_raw).astype(np.float32)

            gt_label = row.get("gt_label", None)
            items.append(
                {
                    "anchor_id": str(row["anchor_id"]),
                    "candidate_id": str(row["candidate_id"]),
                    "seq_a": seq_a,
                    "seq_b": seq_b,
                    "endpoint": ep,
                    "residuals": residuals,
                    "gt_label": -1 if gt_label is None else int(gt_label),
                    "bhat_cost": float(row.get("bhat_cost", 1e6)),
                }
            )
        return items


def pinn_collate_fn(batch: Sequence[List[Dict[str, Any]]]):
    seqs_a: List[torch.Tensor] = []
    seqs_b: List[torch.Tensor] = []
    eps: List[torch.Tensor] = []
    residuals: List[torch.Tensor] = []
    gt_labels: List[int] = []
    bhat_costs: List[float] = []
    anchor_ids: List[str] = []
    group_slices: List[Tuple[int, int]] = []

    start = 0
    for group in batch:
        if not group:
            continue
        anchor_ids.append(str(group[0]["anchor_id"]))
        end = start + len(group)
        group_slices.append((start, end))
        for item in group:
            seqs_a.append(torch.from_numpy(item["seq_a"]))
            seqs_b.append(torch.from_numpy(item["seq_b"]))
            eps.append(torch.from_numpy(item["endpoint"]))
            residuals.append(torch.from_numpy(item["residuals"]))
            gt_labels.append(int(item["gt_label"]))
            bhat_costs.append(float(item["bhat_cost"]))
        start = end

    if not seqs_a:
        raise RuntimeError("Empty PINN batch after collation.")

    max_a = max(t.size(0) for t in seqs_a)
    max_b = max(t.size(0) for t in seqs_b)
    nf = seqs_a[0].size(1)
    pa = torch.zeros(len(seqs_a), max_a, nf, dtype=torch.float32)
    pb = torch.zeros(len(seqs_b), max_b, nf, dtype=torch.float32)
    ma = torch.ones(len(seqs_a), max_a, dtype=torch.bool)
    mb = torch.ones(len(seqs_b), max_b, dtype=torch.bool)
    for i, (sa, sb) in enumerate(zip(seqs_a, seqs_b)):
        pa[i, : sa.size(0)] = sa
        pb[i, : sb.size(0)] = sb
        ma[i, : sa.size(0)] = False
        mb[i, : sb.size(0)] = False

    ep = torch.stack(eps, dim=0)
    residuals_t = torch.stack(residuals, dim=0)
    gt = torch.tensor(gt_labels, dtype=torch.int64)
    bhat = torch.tensor(bhat_costs, dtype=torch.float32)
    return pa, ma, pb, mb, ep, residuals_t, gt, bhat, group_slices, anchor_ids


def compute_group_pairwise_margin_loss(
    costs: torch.Tensor,
    gt_labels: torch.Tensor,
    group_slices: Sequence[Tuple[int, int]],
    margin: float,
) -> torch.Tensor:
    device = costs.device
    group_losses: List[torch.Tensor] = []
    for start, end in group_slices:
        c = costs[start:end]
        y = gt_labels[start:end]
        pos_idx = torch.nonzero(y == 1, as_tuple=False).squeeze(1)
        neg_idx = torch.nonzero(y == 0, as_tuple=False).squeeze(1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        pos = c[pos_idx].unsqueeze(1)
        neg = c[neg_idx].unsqueeze(0)
        losses = F.relu(float(margin) + pos - neg)
        group_losses.append(losses.reshape(-1).mean())

    if not group_losses:
        return torch.zeros((), device=device)
    return torch.stack(group_losses).mean()


def compute_batch_loss(
    total_costs: torch.Tensor,
    aux_pred: torch.Tensor,
    residuals: torch.Tensor,
    gt_labels: torch.Tensor,
    group_slices: Sequence[Tuple[int, int]],
    *,
    gate_logits: Optional[torch.Tensor] = None,
    alpha: float,
    beta: float,
    gamma: float,
    margin: float,
    temperature: float,
    gate_pos_weight: float = 1.0,
) -> Tuple[torch.Tensor, BatchStats]:
    aux_term = F.mse_loss(aux_pred, residuals)
    rank_term = compute_group_pairwise_margin_loss(total_costs.squeeze(1), gt_labels, group_slices, margin=margin)

    known_mask = gt_labels >= 0
    if bool(known_mask.any().item()):
        y = (gt_labels[known_mask] == 1).float()
        if gate_logits is None:
            logits = -total_costs[known_mask].squeeze(1)
        else:
            logits = gate_logits[known_mask].squeeze(1)
        logits = logits / max(float(temperature), 1e-6)
        pos_weight_t = torch.as_tensor(
            float(gate_pos_weight),
            dtype=logits.dtype,
            device=logits.device,
        )
        gate_term = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_t)
    else:
        gate_term = total_costs.sum() * 0.0

    total = float(alpha) * aux_term + float(beta) * rank_term + float(gamma) * gate_term
    stats = BatchStats(
        loss=float(total.detach().item()),
        aux_term=float(aux_term.detach().item()),
        rank_term=float(rank_term.detach().item()),
        gate_term=float(gate_term.detach().item()),
        num_groups=int(len(group_slices)),
    )
    return total, stats


@torch.no_grad()
def evaluate_anchor_metrics(
    model: PhysicsInformedCostNetwork,
    loader: DataLoader,
    device: torch.device,
    *,
    temperature: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    pairwise_correct = 0
    pairwise_total = 0
    top1_correct = 0
    top1_total = 0
    gate_correct = 0
    gate_total = 0
    spearman_vals: List[float] = []

    for pa, ma, pb, mb, ep, residuals, gt, bhat, group_slices, _anchor_ids in loader:
        pa = pa.to(device)
        ma = ma.to(device)
        pb = pb.to(device)
        mb = mb.to(device)
        ep = ep.to(device)
        residuals = residuals.to(device)
        gt = gt.to(device)
        bhat = bhat.to(device)

        try:
            model_out = model(pa, ma, pb, mb, ep, residuals, return_gate_logits=True)
        except TypeError:
            model_out = model(pa, ma, pb, mb, ep, residuals)

        if len(model_out) == 4:
            total_cost, _aux, _weights, gate_logits = model_out
        else:
            total_cost, _aux, _weights = model_out
            gate_logits = -total_cost
        c = total_cost.squeeze(1)

        known_mask = gt >= 0
        if bool(known_mask.any().item()):
            logits = gate_logits[known_mask].squeeze(1) / max(float(temperature), 1e-6)
            pred_same = (torch.sigmoid(logits) >= 0.5).long()
            gt_same = (gt[known_mask] == 1).long()
            gate_correct += int((pred_same == gt_same).sum().item())
            gate_total += int(gt_same.numel())

        for start, end in group_slices:
            cg = c[start:end]
            yg = gt[start:end]
            bg = bhat[start:end]
            pos_idx = torch.nonzero(yg == 1, as_tuple=False).squeeze(1)
            neg_idx = torch.nonzero(yg == 0, as_tuple=False).squeeze(1)
            if pos_idx.numel() > 0 and neg_idx.numel() > 0:
                pos = cg[pos_idx].unsqueeze(1)
                neg = cg[neg_idx].unsqueeze(0)
                pairwise_correct += int((pos < neg).sum().item())
                pairwise_total += int(pos.numel() * neg.numel())
                top1_idx = int(torch.argmin(cg).item())
                top1_correct += int(int(yg[top1_idx].item()) == 1)
                top1_total += 1

            if cg.numel() >= 2:
                c_np = cg.detach().cpu().numpy().astype(np.float64)
                b_np = bg.detach().cpu().numpy().astype(np.float64)
                if np.all(np.isfinite(c_np)) and np.all(np.isfinite(b_np)):
                    spearman_vals.append(_spearman_corr(c_np, b_np))

    return {
        "pairwise_acc_gt_crossclass": float(pairwise_correct / pairwise_total) if pairwise_total > 0 else 0.0,
        "anchor_top1_acc": float(top1_correct / top1_total) if top1_total > 0 else 0.0,
        "gate_acc_known": float(gate_correct / gate_total) if gate_total > 0 else 0.0,
        "mean_anchor_spearman": float(np.mean(spearman_vals)) if spearman_vals else 0.0,
        "pairwise_total": int(pairwise_total),
        "top1_total": int(top1_total),
        "spearman_anchor_count": int(len(spearman_vals)),
    }


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    device = torch.device(args.device)
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = PINNAnchorDataset(
        dataset_path=dataset_path,
        split="train",
        max_candidates=args.cands_per_anchor,
        seed=args.seed,
        dt_floor=args.dt_floor,
        min_points_for_fit=args.min_points_for_fit,
        accel_limit=args.accel_limit,
        lane_tolerance=args.lane_tolerance,
        time_win=args.time_win,
    )
    val_ds = PINNAnchorDataset(
        dataset_path=dataset_path,
        split="val",
        max_candidates=args.cands_per_anchor,
        seed=args.seed,
        dt_floor=args.dt_floor,
        min_points_for_fit=args.min_points_for_fit,
        accel_limit=args.accel_limit,
        lane_tolerance=args.lane_tolerance,
        time_win=args.time_win,
    )

    if len(train_ds) == 0:
        raise RuntimeError("No train anchor groups in dataset.")
    if len(val_ds) == 0:
        raise RuntimeError("No val anchor groups in dataset.")

    norm_stats = train_ds.compute_normalization_stats(sample_pairs=args.norm_sample_pairs)
    train_ds.set_norm_stats(norm_stats)
    val_ds.set_norm_stats(norm_stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.anchors_per_batch,
        shuffle=True,
        collate_fn=pinn_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.anchors_per_batch,
        shuffle=False,
        collate_fn=pinn_collate_fn,
    )

    model_config = {
        "input_size": 8,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "endpoint_dim": args.endpoint_dim,
        "residual_dim": RESIDUAL_DIM,
        "use_correction": not bool(args.disable_correction),
        "pool_weight_first": args.pool_weight_first,
        "pool_weight_last": args.pool_weight_last,
        "pool_weight_mean": args.pool_weight_mean,
    }
    model = PhysicsInformedCostNetwork(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Train groups: {len(train_ds)}, Val groups: {len(val_ds)}")
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_score = -math.inf
    best_state = None
    best_metrics: Dict[str, Any] = {}
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        run_aux = 0.0
        run_rank = 0.0
        run_gate = 0.0
        run_groups = 0

        for pa, ma, pb, mb, ep, residuals, gt, _bhat, group_slices, _anchor_ids in train_loader:
            pa = pa.to(device)
            ma = ma.to(device)
            pb = pb.to(device)
            mb = mb.to(device)
            ep = ep.to(device)
            residuals = residuals.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            total_cost, aux_pred, _weights, gate_logits = model(
                pa,
                ma,
                pb,
                mb,
                ep,
                residuals,
                return_gate_logits=True,
            )
            loss, stats = compute_batch_loss(
                total_costs=total_cost,
                aux_pred=aux_pred,
                residuals=residuals,
                gt_labels=gt,
                group_slices=group_slices,
                gate_logits=gate_logits,
                alpha=float(args.alpha),
                beta=float(args.beta),
                gamma=float(args.gamma),
                margin=float(args.margin),
                temperature=float(args.temperature),
                gate_pos_weight=float(args.gate_pos_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            scale = max(stats.num_groups, 1)
            run_loss += stats.loss * scale
            run_aux += stats.aux_term * scale
            run_rank += stats.rank_term * scale
            run_gate += stats.gate_term * scale
            run_groups += stats.num_groups

        scheduler.step()

        val_metrics = evaluate_anchor_metrics(
            model,
            val_loader,
            device,
            temperature=float(args.temperature),
        )
        val_top1 = float(val_metrics["anchor_top1_acc"])

        denom = max(run_groups, 1)
        train_loss = run_loss / denom
        train_aux = run_aux / denom
        train_rank = run_rank / denom
        train_gate = run_gate / denom

        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_aux_term": float(train_aux),
            "train_rank_term": float(train_rank),
            "train_gate_term": float(train_gate),
            "val_pairwise_acc_gt_crossclass": float(val_metrics["pairwise_acc_gt_crossclass"]),
            "val_anchor_top1_acc": float(val_metrics["anchor_top1_acc"]),
            "val_gate_acc_known": float(val_metrics["gate_acc_known"]),
            "val_mean_anchor_spearman": float(val_metrics["mean_anchor_spearman"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        record["val_selection_score"] = _compute_model_selection_score(
            record,
            strategy=str(args.selection_metric),
            weight_top1=float(args.selection_weight_top1),
            weight_pairwise=float(args.selection_weight_pairwise),
            weight_spearman=float(args.selection_weight_spearman),
            weight_gate=float(args.selection_weight_gate),
        )
        history.append(record)
        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_loss:.4f} aux={train_aux:.4f} rank={train_rank:.4f} gate={train_gate:.4f} | "
            f"val_top1={val_top1:.4f} val_pair={val_metrics['pairwise_acc_gt_crossclass']:.4f} "
            f"val_gate={val_metrics['gate_acc_known']:.4f} val_spear={val_metrics['mean_anchor_spearman']:.4f} "
            f"val_sel={record['val_selection_score']:.4f}"
        )

        if record["val_selection_score"] > best_val_score + 1e-6:
            best_val_score = float(record["val_selection_score"])
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = dict(record)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_metrics = history[-1] if history else {}

    checkpoint = {
        "model_type": "pinn",
        "model_state_dict": best_state,
        "model_config": model_config,
        "seq_mean": norm_stats["mean"],
        "seq_std": norm_stats["std"],
        "ep_mean": norm_stats["ep_mean"],
        "ep_std": norm_stats["ep_std"],
        "train_config": {
            "dataset_path": str(dataset_path),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "anchors_per_batch": int(args.anchors_per_batch),
            "cands_per_anchor": int(args.cands_per_anchor),
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "gamma": float(args.gamma),
            "gate_pos_weight": float(args.gate_pos_weight),
            "margin": float(args.margin),
            "temperature": float(args.temperature),
            "selection_metric": str(args.selection_metric),
            "selection_weight_top1": float(args.selection_weight_top1),
            "selection_weight_pairwise": float(args.selection_weight_pairwise),
            "selection_weight_spearman": float(args.selection_weight_spearman),
            "selection_weight_gate": float(args.selection_weight_gate),
            "time_win": float(args.time_win),
            "dt_floor": float(args.dt_floor),
            "min_points_for_fit": int(args.min_points_for_fit),
            "accel_limit": float(args.accel_limit),
            "lane_tolerance": float(args.lane_tolerance),
            "disable_correction": bool(args.disable_correction),
            "seed": int(args.seed),
        },
        "best_metrics": best_metrics,
    }
    torch.save(checkpoint, output_path)

    history_path = output_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved checkpoint to {output_path}")
    print(f"Saved history to {history_path}")
    if best_metrics:
        print(
            f"Best epoch metrics: val_top1={best_metrics.get('val_anchor_top1_acc', float('nan')):.4f}, "
            f"val_pair={best_metrics.get('val_pairwise_acc_gt_crossclass', float('nan')):.4f}, "
            f"val_gate={best_metrics.get('val_gate_acc_known', float('nan')):.4f}, "
            f"val_sel={best_metrics.get('val_selection_score', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
