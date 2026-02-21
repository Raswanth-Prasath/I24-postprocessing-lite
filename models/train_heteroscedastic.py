#!/usr/bin/env python
"""
Train heteroscedastic transformer for stitching cost.

Objective:
  loss = positive Gaussian NLL + neg_loss_weight * group margin separation

Data source:
  models/outputs/transformer_ranking_dataset.jsonl (+ fragments store)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.heteroscedastic_model import HeteroscedasticTrajectoryModel  # noqa: E402
from models.rich_sequence_dataset import extract_rich_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heteroscedastic transformer model.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_ranking_dataset.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "heteroscedastic_model.pth"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--anchors-per-batch", type=int, default=12)
    parser.add_argument("--cands-per-anchor", type=int, default=6)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--norm-sample-pairs", type=int, default=2000)

    parser.add_argument("--max-query-points", type=int, default=25)
    parser.add_argument("--time-win", type=float, default=15.0)
    parser.add_argument("--neg-margin", type=float, default=2.0)
    parser.add_argument("--neg-loss-weight", type=float, default=0.1)

    # Optional scenario-based source-holdout filtering.
    parser.add_argument(
        "--train-scenarios",
        nargs="+",
        default=None,
        help="Optional scenario names to keep in train split (e.g. i ii).",
    )
    parser.add_argument(
        "--val-scenarios",
        nargs="+",
        default=None,
        help="Optional scenario names to keep in val split (e.g. iii).",
    )

    # Encoder/model config
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pred-hidden", type=int, default=128)
    parser.add_argument("--cov-eps", type=float, default=1e-3)
    parser.add_argument("--pool-weight-first", type=float, default=0.2)
    parser.add_argument("--pool-weight-last", type=float, default=0.5)
    parser.add_argument("--pool-weight-mean", type=float, default=0.3)

    return parser.parse_args()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class BatchStats:
    loss: float
    nll_pos: float
    sep_loss: float
    pos_maha_mean: float
    neg_maha_mean: float
    num_groups: int


class HeteroscedasticAnchorDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        split: str,
        max_candidates: int,
        *,
        max_query_points: int = 25,
        time_win: float = 15.0,
        seed: int = 42,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        allowed_scenarios: Optional[Sequence[str]] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = str(split)
        self.max_candidates = int(max_candidates)
        self.max_query_points = int(max_query_points)
        self.time_win = float(time_win)
        self.rng = np.random.default_rng(int(seed))
        self.norm_stats = norm_stats
        self.allowed_scenarios = set(str(s) for s in (allowed_scenarios or [])) or None

        self.fragments_path = self.dataset_path.parent / f"{self.dataset_path.stem}.fragments.jsonl"
        self.fragment_store: Dict[str, Dict[str, Any]] = {}

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
                if self.allowed_scenarios is not None and str(row.get("scenario", "")) not in self.allowed_scenarios:
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
        # Ranking dataset convention: candidate_ref is track1 (earlier), anchor_ref is track2 (later).
        cand_ref = row.get("candidate_ref")
        anchor_ref = row.get("anchor_ref")
        if cand_ref is None or anchor_ref is None:
            raise KeyError("Row missing candidate_ref/anchor_ref.")
        return self.fragment_store[str(cand_ref)], self.fragment_store[str(anchor_ref)]

    def _normalize_seq(self, seq: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return seq
        return (seq - self.norm_stats["mean"]) / self.norm_stats["std"]

    def set_norm_stats(self, norm_stats: Dict[str, np.ndarray]) -> None:
        self.norm_stats = {
            "mean": np.asarray(norm_stats["mean"], dtype=np.float32),
            "std": np.asarray(norm_stats["std"], dtype=np.float32),
        }

    def compute_normalization_stats(self, sample_pairs: int = 2000) -> Dict[str, np.ndarray]:
        rows: List[Dict[str, Any]] = []
        for group in self.groups:
            rows.extend(group)
        if not rows:
            raise RuntimeError("Cannot compute normalization stats on empty split.")

        n = min(int(sample_pairs), len(rows))
        idx = self.rng.choice(len(rows), size=n, replace=False)

        seq_rows: List[np.ndarray] = []
        for i in idx:
            row = rows[int(i)]
            track1, _track2 = self._resolve_pair_fragments(row)
            seq_rows.append(extract_rich_sequence(track1))

        seq_data = np.vstack(seq_rows)
        return {
            "mean": seq_data.mean(axis=0).astype(np.float32),
            "std": (seq_data.std(axis=0) + 1e-6).astype(np.float32),
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

    @staticmethod
    def _end_timestamp(track: Dict[str, Any]) -> float:
        if "last_timestamp" in track:
            return float(track["last_timestamp"])
        return float(track["timestamp"][-1])

    @staticmethod
    def _end_position(track: Dict[str, Any]) -> Tuple[float, float]:
        x_end = float(track.get("ending_x", track["x_position"][-1]))
        y_end = float(track["y_position"][-1])
        return x_end, y_end

    def _build_query_and_targets(
        self,
        track1: Dict[str, Any],
        track2: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        t1_end = self._end_timestamp(track1)
        x1_end, y1_end = self._end_position(track1)

        t2 = np.asarray(track2["timestamp"], dtype=np.float64)
        x2 = np.asarray(track2["x_position"], dtype=np.float32)
        y2 = np.asarray(track2["y_position"], dtype=np.float32)

        dt_all = t2 - t1_end
        valid_idx = np.where((dt_all >= 0.0) & (dt_all <= self.time_win))[0]

        if len(valid_idx) == 0:
            # Fallback keeps training robust on edge cases while preserving causal semantics.
            i0 = 0
            dt = np.asarray([max(0.0, float(dt_all[i0]))], dtype=np.float32)
            targets = np.asarray([[x2[i0] - x1_end, y2[i0] - y1_end]], dtype=np.float32)
            return dt, targets

        valid_idx = valid_idx[: self.max_query_points]
        dt = dt_all[valid_idx].astype(np.float32)
        targets = np.stack([x2[valid_idx] - x1_end, y2[valid_idx] - y1_end], axis=-1).astype(np.float32)
        return dt, targets

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        rows = self._select_candidates(self.groups[idx])
        items: List[Dict[str, Any]] = []

        for row in rows:
            track1, track2 = self._resolve_pair_fragments(row)
            seq = self._normalize_seq(extract_rich_sequence(track1)).astype(np.float32)
            query_dt, targets = self._build_query_and_targets(track1, track2)

            gt_label = row.get("gt_label", None)
            items.append(
                {
                    "anchor_id": str(row.get("anchor_id", "unknown")),
                    "candidate_id": str(row.get("candidate_id", "unknown")),
                    "seq": seq,
                    "query_dt": query_dt,
                    "targets": targets,
                    "gt_label": -1 if gt_label is None else int(gt_label),
                }
            )

        return items


def heteroscedastic_collate_fn(batch: Sequence[List[Dict[str, Any]]]):
    seqs: List[torch.Tensor] = []
    query_dts: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    labels: List[int] = []
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
            seqs.append(torch.from_numpy(item["seq"]))
            query_dts.append(torch.from_numpy(item["query_dt"]))
            targets.append(torch.from_numpy(item["targets"]))
            labels.append(int(item["gt_label"]))
        start = end

    if not seqs:
        raise RuntimeError("Empty batch after collation.")

    max_seq = max(t.size(0) for t in seqs)
    max_q = max(t.size(0) for t in query_dts)
    nf = seqs[0].size(1)

    ps = torch.zeros(len(seqs), max_seq, nf, dtype=torch.float32)
    ms = torch.ones(len(seqs), max_seq, dtype=torch.bool)

    qdt = torch.zeros(len(query_dts), max_q, dtype=torch.float32)
    qmask = torch.ones(len(query_dts), max_q, dtype=torch.bool)
    tgt = torch.zeros(len(targets), max_q, 2, dtype=torch.float32)

    for i, (s, dti, ti) in enumerate(zip(seqs, query_dts, targets)):
        ps[i, : s.size(0)] = s
        ms[i, : s.size(0)] = False

        qn = dti.size(0)
        qdt[i, :qn] = dti
        qmask[i, :qn] = False
        tgt[i, :qn] = ti

    gt = torch.tensor(labels, dtype=torch.int64)
    return ps, ms, qdt, qmask, tgt, gt, group_slices, anchor_ids


def compute_group_margin_loss(
    distances: torch.Tensor,
    labels: torch.Tensor,
    group_slices: Sequence[Tuple[int, int]],
    margin: float,
) -> torch.Tensor:
    group_losses: List[torch.Tensor] = []
    for start, end in group_slices:
        d = distances[start:end]
        y = labels[start:end]
        pos_idx = torch.nonzero(y == 1, as_tuple=False).squeeze(1)
        neg_idx = torch.nonzero(y == 0, as_tuple=False).squeeze(1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue

        pos = d[pos_idx].unsqueeze(1)
        neg = d[neg_idx].unsqueeze(0)
        losses = F.relu(float(margin) + pos - neg)
        group_losses.append(losses.reshape(-1).mean())

    if not group_losses:
        return distances.sum() * 0.0
    return torch.stack(group_losses).mean()


def compute_batch_loss(
    model: HeteroscedasticTrajectoryModel,
    mu: torch.Tensor,
    chol_params: torch.Tensor,
    targets: torch.Tensor,
    query_mask: torch.Tensor,
    labels: torch.Tensor,
    group_slices: Sequence[Tuple[int, int]],
    *,
    neg_margin: float,
    neg_loss_weight: float,
) -> Tuple[torch.Tensor, BatchStats]:
    pos_mask = labels == 1

    if bool(pos_mask.any().item()):
        nll_pos = model.compute_nll(
            mu=mu[pos_mask],
            chol_params=chol_params[pos_mask],
            targets=targets[pos_mask],
            query_mask=query_mask[pos_mask],
        )
    else:
        nll_pos = mu.sum() * 0.0

    distances = model.compute_mahalanobis(mu=mu, chol_params=chol_params, targets=targets, query_mask=query_mask)
    sep_loss = compute_group_margin_loss(distances, labels, group_slices, margin=neg_margin)

    total = nll_pos + float(neg_loss_weight) * sep_loss

    known_pos = distances[labels == 1]
    known_neg = distances[labels == 0]
    pos_mean = float(known_pos.mean().item()) if known_pos.numel() > 0 else float("nan")
    neg_mean = float(known_neg.mean().item()) if known_neg.numel() > 0 else float("nan")

    stats = BatchStats(
        loss=float(total.detach().item()),
        nll_pos=float(nll_pos.detach().item()),
        sep_loss=float(sep_loss.detach().item()),
        pos_maha_mean=pos_mean,
        neg_maha_mean=neg_mean,
        num_groups=int(len(group_slices)),
    )
    return total, stats


@torch.no_grad()
def evaluate_loader(
    model: HeteroscedasticTrajectoryModel,
    loader: DataLoader,
    device: torch.device,
    *,
    neg_margin: float,
    neg_loss_weight: float,
) -> Dict[str, float]:
    model.eval()

    sum_loss = 0.0
    sum_nll = 0.0
    sum_sep = 0.0
    sum_groups = 0

    pos_maha: List[float] = []
    neg_maha: List[float] = []

    pairwise_correct = 0
    pairwise_total = 0
    top1_correct = 0
    top1_total = 0

    for seq, seq_mask, qdt, qmask, tgt, labels, group_slices, _anchor_ids in loader:
        seq = seq.to(device)
        seq_mask = seq_mask.to(device)
        qdt = qdt.to(device)
        qmask = qmask.to(device)
        tgt = tgt.to(device)
        labels = labels.to(device)

        mu, chol = model(seq, seq_mask, qdt)
        loss, stats = compute_batch_loss(
            model=model,
            mu=mu,
            chol_params=chol,
            targets=tgt,
            query_mask=qmask,
            labels=labels,
            group_slices=group_slices,
            neg_margin=neg_margin,
            neg_loss_weight=neg_loss_weight,
        )

        denom = max(stats.num_groups, 1)
        sum_loss += float(loss.item()) * denom
        sum_nll += stats.nll_pos * denom
        sum_sep += stats.sep_loss * denom
        sum_groups += stats.num_groups

        d = model.compute_mahalanobis(mu=mu, chol_params=chol, targets=tgt, query_mask=qmask)
        pos_vals = d[labels == 1].detach().cpu().numpy()
        neg_vals = d[labels == 0].detach().cpu().numpy()
        if len(pos_vals) > 0:
            pos_maha.extend(float(v) for v in pos_vals)
        if len(neg_vals) > 0:
            neg_maha.extend(float(v) for v in neg_vals)

        for start, end in group_slices:
            dg = d[start:end]
            yg = labels[start:end]
            pos_idx = torch.nonzero(yg == 1, as_tuple=False).squeeze(1)
            neg_idx = torch.nonzero(yg == 0, as_tuple=False).squeeze(1)
            if pos_idx.numel() > 0 and neg_idx.numel() > 0:
                pos = dg[pos_idx].unsqueeze(1)
                neg = dg[neg_idx].unsqueeze(0)
                pairwise_correct += int((pos < neg).sum().item())
                pairwise_total += int(pos.numel() * neg.numel())

                top1_idx = int(torch.argmin(dg).item())
                top1_correct += int(int(yg[top1_idx].item()) == 1)
                top1_total += 1

    norm = max(sum_groups, 1)
    return {
        "val_loss": float(sum_loss / norm),
        "val_nll_pos": float(sum_nll / norm),
        "val_sep_loss": float(sum_sep / norm),
        "val_pos_maha_mean": float(np.mean(pos_maha)) if pos_maha else float("nan"),
        "val_neg_maha_mean": float(np.mean(neg_maha)) if neg_maha else float("nan"),
        "val_pairwise_acc": float(pairwise_correct / pairwise_total) if pairwise_total > 0 else 0.0,
        "val_top1_acc": float(top1_correct / top1_total) if top1_total > 0 else 0.0,
        "val_pairwise_total": int(pairwise_total),
        "val_top1_total": int(top1_total),
    }


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    device = torch.device(args.device)
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = HeteroscedasticAnchorDataset(
        dataset_path=dataset_path,
        split="train",
        max_candidates=args.cands_per_anchor,
        max_query_points=args.max_query_points,
        time_win=args.time_win,
        seed=args.seed,
        allowed_scenarios=args.train_scenarios,
    )
    val_ds = HeteroscedasticAnchorDataset(
        dataset_path=dataset_path,
        split="val",
        max_candidates=args.cands_per_anchor,
        max_query_points=args.max_query_points,
        time_win=args.time_win,
        seed=args.seed,
        allowed_scenarios=args.val_scenarios,
    )

    if len(train_ds) == 0:
        raise RuntimeError("No train anchor groups available.")
    if len(val_ds) == 0:
        raise RuntimeError("No val anchor groups available.")

    norm_stats = train_ds.compute_normalization_stats(sample_pairs=args.norm_sample_pairs)
    train_ds.set_norm_stats(norm_stats)
    val_ds.set_norm_stats(norm_stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.anchors_per_batch,
        shuffle=True,
        collate_fn=heteroscedastic_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.anchors_per_batch,
        shuffle=False,
        collate_fn=heteroscedastic_collate_fn,
    )

    model_config = {
        "input_size": 8,
        "d_model": int(args.d_model),
        "nhead": int(args.nhead),
        "dim_feedforward": int(args.dim_feedforward),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "pred_hidden": int(args.pred_hidden),
        "cov_eps": float(args.cov_eps),
        "pool_weight_first": float(args.pool_weight_first),
        "pool_weight_last": float(args.pool_weight_last),
        "pool_weight_mean": float(args.pool_weight_mean),
    }
    model = HeteroscedasticTrajectoryModel(**model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Train groups: {len(train_ds)}, Val groups: {len(val_ds)}")
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_state = None
    best_metrics: Dict[str, Any] = {}
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        run_nll = 0.0
        run_sep = 0.0
        run_groups = 0

        for seq, seq_mask, qdt, qmask, tgt, labels, group_slices, _anchor_ids in train_loader:
            seq = seq.to(device)
            seq_mask = seq_mask.to(device)
            qdt = qdt.to(device)
            qmask = qmask.to(device)
            tgt = tgt.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            mu, chol = model(seq, seq_mask, qdt)
            loss, stats = compute_batch_loss(
                model=model,
                mu=mu,
                chol_params=chol,
                targets=tgt,
                query_mask=qmask,
                labels=labels,
                group_slices=group_slices,
                neg_margin=float(args.neg_margin),
                neg_loss_weight=float(args.neg_loss_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            scale = max(stats.num_groups, 1)
            run_loss += stats.loss * scale
            run_nll += stats.nll_pos * scale
            run_sep += stats.sep_loss * scale
            run_groups += stats.num_groups

        scheduler.step()

        val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            neg_margin=float(args.neg_margin),
            neg_loss_weight=float(args.neg_loss_weight),
        )
        val_obj = float(val_metrics["val_loss"])

        denom = max(run_groups, 1)
        train_loss = run_loss / denom
        train_nll = run_nll / denom
        train_sep = run_sep / denom

        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_nll_pos": float(train_nll),
            "train_sep_loss": float(train_sep),
            "val_loss": float(val_metrics["val_loss"]),
            "val_nll_pos": float(val_metrics["val_nll_pos"]),
            "val_sep_loss": float(val_metrics["val_sep_loss"]),
            "val_pos_maha_mean": float(val_metrics["val_pos_maha_mean"]),
            "val_neg_maha_mean": float(val_metrics["val_neg_maha_mean"]),
            "val_pairwise_acc": float(val_metrics["val_pairwise_acc"]),
            "val_top1_acc": float(val_metrics["val_top1_acc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_nll={train_nll:.4f} train_sep={train_sep:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} val_nll={val_metrics['val_nll_pos']:.4f} "
            f"val_sep={val_metrics['val_sep_loss']:.4f} "
            f"val_pos={val_metrics['val_pos_maha_mean']:.3f} val_neg={val_metrics['val_neg_maha_mean']:.3f} "
            f"val_top1={val_metrics['val_top1_acc']:.4f}"
        )

        if val_obj + 1e-6 < best_val:
            best_val = val_obj
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
        "model_type": "heteroscedastic",
        "model_state_dict": best_state,
        "model_config": model_config,
        "seq_mean": norm_stats["mean"],
        "seq_std": norm_stats["std"],
        "train_config": {
            "dataset_path": str(dataset_path),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "anchors_per_batch": int(args.anchors_per_batch),
            "cands_per_anchor": int(args.cands_per_anchor),
            "max_query_points": int(args.max_query_points),
            "time_win": float(args.time_win),
            "neg_margin": float(args.neg_margin),
            "neg_loss_weight": float(args.neg_loss_weight),
            "seed": int(args.seed),
            "train_scenarios": list(args.train_scenarios) if args.train_scenarios is not None else None,
            "val_scenarios": list(args.val_scenarios) if args.val_scenarios is not None else None,
        },
        "best_metrics": best_metrics,
        "history": history,
    }

    torch.save(checkpoint, str(output_path))
    hist_path = output_path.with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved checkpoint to {output_path}")
    print(f"Saved history to {hist_path}")
    print(f"Best validation objective: {best_val:.6f}")


if __name__ == "__main__":
    main()
