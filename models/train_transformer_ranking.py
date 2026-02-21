#!/usr/bin/env python
"""
Train Siamese Transformer with anchor-level weighted pairwise ranking loss.

- Training unit: anchor + candidate group
- Loss: 0.8 * GT cross-class pairwise margin + 0.2 * Bhattacharyya soft ranking
- Mining: online semi-hard negatives from epoch N
"""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rich_sequence_dataset import extract_endpoint_features, extract_rich_sequence  # noqa: E402
from models.transformer_model import SiameseTransformerNetwork  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer ranking model.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_ranking_dataset.jsonl"),
        help="Path to ranking dataset JSONL produced by build_ranking_dataset.py.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_ranking_model.pth"),
        help="Checkpoint output path.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--anchors-per-batch", type=int, default=12)
    parser.add_argument("--cands-per-anchor", type=int, default=6)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--gt-weight", type=float, default=0.5)
    parser.add_argument("--soft-weight", type=float, default=0.2)
    parser.add_argument("--sep-weight", type=float, default=0.15)
    parser.add_argument("--pos-margin", type=float, default=1.0)
    parser.add_argument("--neg-margin", type=float, default=5.0)
    parser.add_argument("--top1-weight", type=float, default=0.15)
    parser.add_argument("--soft-anneal-epochs", type=int, default=10)

    parser.add_argument("--enable-mining-epoch", type=int, default=3)
    parser.add_argument("--max-neg-per-pos", type=int, default=3)

    parser.add_argument("--pos-pair-weight", type=float, default=1.0,
                        help="Multiplicative weight on positive-side GT margin losses (default 1.0, recommend 5.0).")
    parser.add_argument("--oversample-pos-groups", action="store_true", default=False,
                        help="Oversample anchor groups containing GT positives (2x weight).")

    parser.add_argument("--norm-sample-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    # Model config
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


@dataclass
class BatchStats:
    loss: float
    gt_term: float
    soft_term: float
    sep_term: float
    top1_term: float
    effective_soft_weight: float
    semi_hard_selected: int
    hard_selected: int
    random_selected: int
    num_groups: int


class RankingAnchorDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        split: str,
        max_candidates: int,
        seed: int = 42,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = str(split)
        self.max_candidates = int(max_candidates)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.norm_stats = norm_stats
        self.fragments_path = self._default_fragments_path(self.dataset_path)
        self.fragment_store: Dict[str, Dict[str, Any]] = {}

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Ranking dataset not found: {self.dataset_path}")

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        with open(self.dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if str(row.get("split", "train")) != self.split:
                    continue
                anchor_key = str(row["anchor_key"])
                grouped.setdefault(anchor_key, []).append(row)

        self.groups: List[List[Dict[str, Any]]] = [
            sorted(rows, key=lambda r: float(r.get("bhat_cost", 1e6)))
            for _k, rows in sorted(grouped.items(), key=lambda kv: kv[0])
            if rows
        ]

        self._load_fragment_store()

    def __len__(self) -> int:
        return len(self.groups)

    def _default_fragments_path(self, dataset_path: Path) -> Path:
        return dataset_path.parent / f"{dataset_path.stem}.fragments.jsonl"

    def _load_fragment_store(self) -> None:
        # Backward compatibility: old datasets may embed full tracks per row.
        if not self.fragments_path.exists():
            return
        with open(self.fragments_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ref = str(row["fragment_ref"])
                frag = row["fragment"]
                self.fragment_store[ref] = frag

    def _resolve_pair_fragments(self, row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if "track_candidate" in row and "track_anchor" in row:
            return row["track_candidate"], row["track_anchor"]

        cand_ref = row.get("candidate_ref")
        anchor_ref = row.get("anchor_ref")
        if cand_ref is None or anchor_ref is None:
            raise KeyError("Row missing both embedded tracks and fragment refs (candidate_ref/anchor_ref).")

        try:
            fa = self.fragment_store[str(cand_ref)]
            fb = self.fragment_store[str(anchor_ref)]
        except KeyError as exc:
            raise KeyError(
                f"Missing fragment_ref '{exc.args[0]}' in fragment store {self.fragments_path}"
            ) from exc
        return fa, fb

    def set_norm_stats(self, norm_stats: Dict[str, np.ndarray]) -> None:
        self.norm_stats = {
            "mean": np.asarray(norm_stats["mean"], dtype=np.float32),
            "std": np.asarray(norm_stats["std"], dtype=np.float32),
            "ep_mean": np.asarray(norm_stats["ep_mean"], dtype=np.float32),
            "ep_std": np.asarray(norm_stats["ep_std"], dtype=np.float32),
        }

    def compute_normalization_stats(self, sample_pairs: int = 2000) -> Dict[str, np.ndarray]:
        rows: List[Dict[str, Any]] = []
        for group in self.groups:
            rows.extend(group)

        if not rows:
            raise RuntimeError("Cannot compute normalization stats on empty dataset split.")

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

        needed = self.max_candidates - len(selected)
        if needed > 0:
            selected.extend(remaining[:needed])

        return sorted(selected, key=lambda r: float(r.get("bhat_cost", 1e6)))

    def _normalize_seq(self, seq: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return seq
        return (seq - self.norm_stats["mean"]) / self.norm_stats["std"]

    def _normalize_ep(self, ep: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return ep
        return (ep - self.norm_stats["ep_mean"]) / self.norm_stats["ep_std"]

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        rows = self._select_candidates(self.groups[idx])
        group_items: List[Dict[str, Any]] = []

        for row in rows:
            fa, fb = self._resolve_pair_fragments(row)
            seq_a = self._normalize_seq(extract_rich_sequence(fa)).astype(np.float32)
            seq_b = self._normalize_seq(extract_rich_sequence(fb)).astype(np.float32)
            ep = self._normalize_ep(extract_endpoint_features(fa, fb)).astype(np.float32)
            gt_label = row.get("gt_label", None)
            group_items.append(
                {
                    "anchor_id": str(row["anchor_id"]),
                    "candidate_id": str(row["candidate_id"]),
                    "seq_a": seq_a,
                    "seq_b": seq_b,
                    "endpoint": ep,
                    "gt_label": -1 if gt_label is None else int(gt_label),
                    "bhat_cost": float(row["bhat_cost"]),
                }
            )

        return group_items


def ranking_collate_fn(batch: Sequence[List[Dict[str, Any]]]):
    seqs_a: List[torch.Tensor] = []
    seqs_b: List[torch.Tensor] = []
    eps: List[torch.Tensor] = []
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
            gt_labels.append(int(item["gt_label"]))
            bhat_costs.append(float(item["bhat_cost"]))
        start = end

    if not seqs_a:
        raise RuntimeError("Empty ranking batch after collation.")

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
    gt = torch.tensor(gt_labels, dtype=torch.int64)
    bhat = torch.tensor(bhat_costs, dtype=torch.float32)

    return pa, ma, pb, mb, ep, gt, bhat, group_slices, anchor_ids


def _select_mined_negatives(
    pos_score: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float,
    max_neg_per_pos: int,
) -> Tuple[torch.Tensor, int, int, int]:
    deltas = neg_scores - pos_score
    semi_mask = (deltas > 0.0) & (deltas < float(margin) + 0.3)
    hard_mask = deltas <= 0.0

    semi_idx = torch.nonzero(semi_mask, as_tuple=False).squeeze(1)
    hard_idx = torch.nonzero(hard_mask, as_tuple=False).squeeze(1)

    if semi_idx.numel() > 0:
        order = torch.argsort(deltas[semi_idx])
        chosen = semi_idx[order[: int(max_neg_per_pos)]]
        return neg_scores[chosen], int(chosen.numel()), 0, 0

    if hard_idx.numel() > 0:
        order = torch.argsort(deltas[hard_idx])
        chosen = hard_idx[order[: int(max_neg_per_pos)]]
        return neg_scores[chosen], 0, int(chosen.numel()), 0

    rand_idx = int(torch.randint(low=0, high=neg_scores.numel(), size=(1,), device=neg_scores.device).item())
    return neg_scores[rand_idx : rand_idx + 1], 0, 0, 1


def compute_weighted_pairwise_margin_loss(
    scores: torch.Tensor,
    gt_labels: torch.Tensor,
    bhat_costs: torch.Tensor,
    group_slices: Sequence[Tuple[int, int]],
    margin: float,
    gt_weight: float,
    effective_soft_weight: float,
    sep_weight: float,
    pos_margin: float,
    neg_margin: float,
    top1_weight: float,
    enable_mining: bool,
    max_neg_per_pos: int,
    pos_pair_weight: float = 1.0,
) -> Tuple[torch.Tensor, BatchStats]:
    device = scores.device
    group_losses: List[torch.Tensor] = []
    gt_terms: List[torch.Tensor] = []
    soft_terms: List[torch.Tensor] = []
    sep_terms: List[torch.Tensor] = []
    top1_terms: List[torch.Tensor] = []

    semi_hard_selected = 0
    hard_selected = 0
    random_selected = 0

    for start, end in group_slices:
        s = scores[start:end]
        y = gt_labels[start:end]
        b = bhat_costs[start:end]

        pos_idx = torch.nonzero(y == 1, as_tuple=False).squeeze(1)
        neg_idx = torch.nonzero(y == 0, as_tuple=False).squeeze(1)

        gt_pair_losses: List[torch.Tensor] = []
        if pos_idx.numel() > 0 and neg_idx.numel() > 0:
            pos_scores = s[pos_idx]
            neg_scores = s[neg_idx]

            for p in pos_scores:
                if enable_mining:
                    mined, c_semi, c_hard, c_rand = _select_mined_negatives(
                        p,
                        neg_scores,
                        margin=margin,
                        max_neg_per_pos=max_neg_per_pos,
                    )
                    semi_hard_selected += c_semi
                    hard_selected += c_hard
                    random_selected += c_rand
                    losses = F.relu(float(margin) + p - mined) * float(pos_pair_weight)
                else:
                    losses = F.relu(float(margin) + p - neg_scores) * float(pos_pair_weight)
                gt_pair_losses.append(losses.reshape(-1))

        if gt_pair_losses:
            gt_term = torch.cat(gt_pair_losses).mean()
        else:
            gt_term = torch.zeros((), device=device)

        n = s.numel()
        soft_pair_losses: List[torch.Tensor] = []
        for i in range(n):
            for j in range(i + 1, n):
                yi = int(y[i].item())
                yj = int(y[j].item())
                if yi >= 0 and yj >= 0 and yi != yj:
                    continue

                bi = float(b[i].item())
                bj = float(b[j].item())
                if not np.isfinite(bi) or not np.isfinite(bj):
                    continue
                gap = abs(bi - bj)
                if gap <= 1e-9:
                    continue

                if bi < bj:
                    better, worse = i, j
                else:
                    better, worse = j, i

                w = 0.3 * min(max(gap / 2.0, 0.25), 1.0)
                l = F.relu(float(margin) + s[better] - s[worse]) * float(w)
                soft_pair_losses.append(l)

        if soft_pair_losses:
            soft_term = torch.stack(soft_pair_losses).mean()
        else:
            soft_term = torch.zeros((), device=device)

        sep_losses: List[torch.Tensor] = []
        for idx in range(s.numel()):
            label = int(y[idx].item())
            if label < 0:
                continue
            cost = F.softplus(s[idx])
            if label == 1:
                sep_losses.append(F.relu(cost - float(pos_margin)) ** 2)
            else:
                sep_losses.append(F.relu(float(neg_margin) - cost) ** 2)

        if sep_losses:
            sep_term = torch.stack(sep_losses).mean()
        else:
            sep_term = torch.zeros((), device=device)

        known_mask = y >= 0
        known_s = s[known_mask]
        known_y = y[known_mask]
        known_pos_mask = known_y == 1
        known_neg_mask = known_y == 0
        if (
            known_s.numel() >= 2
            and bool(known_pos_mask.any().item())
            and bool(known_neg_mask.any().item())
        ):
            logits = -known_s
            pos_logits = logits[known_pos_mask]
            top1_term = -torch.logsumexp(pos_logits, dim=0) + torch.logsumexp(logits, dim=0)
        else:
            top1_term = torch.zeros((), device=device)

        group_total = (
            float(gt_weight) * gt_term
            + float(effective_soft_weight) * soft_term
            + float(sep_weight) * sep_term
            + float(top1_weight) * top1_term
        )
        group_losses.append(group_total)
        gt_terms.append(gt_term)
        soft_terms.append(soft_term)
        sep_terms.append(sep_term)
        top1_terms.append(top1_term)

    if group_losses:
        total_loss = torch.stack(group_losses).mean()
    else:
        total_loss = scores.sum() * 0.0

    stats = BatchStats(
        loss=float(total_loss.detach().item()),
        gt_term=float(torch.stack(gt_terms).mean().detach().item()) if gt_terms else 0.0,
        soft_term=float(torch.stack(soft_terms).mean().detach().item()) if soft_terms else 0.0,
        sep_term=float(torch.stack(sep_terms).mean().detach().item()) if sep_terms else 0.0,
        top1_term=float(torch.stack(top1_terms).mean().detach().item()) if top1_terms else 0.0,
        effective_soft_weight=float(effective_soft_weight),
        semi_hard_selected=int(semi_hard_selected),
        hard_selected=int(hard_selected),
        random_selected=int(random_selected),
        num_groups=int(len(group_slices)),
    )
    return total_loss, stats


@torch.no_grad()
def evaluate_anchor_metrics(
    model: SiameseTransformerNetwork,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    pairwise_correct = 0
    pairwise_total = 0
    top1_correct = 0
    top1_total = 0
    spearman_vals: List[float] = []

    for pa, ma, pb, mb, ep, gt, bhat, group_slices, _anchor_ids in loader:
        pa = pa.to(device)
        ma = ma.to(device)
        pb = pb.to(device)
        mb = mb.to(device)
        ep = ep.to(device)
        gt = gt.to(device)
        bhat = bhat.to(device)

        scores = model(pa, ma, pb, mb, ep).squeeze(1)

        for start, end in group_slices:
            s = scores[start:end]
            y = gt[start:end]
            b = bhat[start:end]

            pos_idx = torch.nonzero(y == 1, as_tuple=False).squeeze(1)
            neg_idx = torch.nonzero(y == 0, as_tuple=False).squeeze(1)

            if pos_idx.numel() > 0 and neg_idx.numel() > 0:
                pos_scores = s[pos_idx].unsqueeze(1)
                neg_scores = s[neg_idx].unsqueeze(0)
                pairwise_correct += int((pos_scores < neg_scores).sum().item())
                pairwise_total += int(pos_scores.numel() * neg_scores.numel())

                top1_idx = int(torch.argmin(s).item())
                top1_correct += int(int(y[top1_idx].item()) == 1)
                top1_total += 1

            if s.numel() >= 2:
                b_np = b.detach().cpu().numpy().astype(np.float64)
                s_np = s.detach().cpu().numpy().astype(np.float64)
                if np.all(np.isfinite(b_np)) and np.all(np.isfinite(s_np)):
                    spearman_vals.append(_spearman_corr(s_np, b_np))

    return {
        "pairwise_acc_gt_crossclass": float(pairwise_correct / pairwise_total) if pairwise_total > 0 else 0.0,
        "anchor_top1_acc": float(top1_correct / top1_total) if top1_total > 0 else 0.0,
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

    train_ds = RankingAnchorDataset(
        dataset_path=dataset_path,
        split="train",
        max_candidates=args.cands_per_anchor,
        seed=args.seed,
        norm_stats=None,
    )
    val_ds = RankingAnchorDataset(
        dataset_path=dataset_path,
        split="val",
        max_candidates=args.cands_per_anchor,
        seed=args.seed,
        norm_stats=None,
    )

    if len(train_ds) == 0:
        raise RuntimeError("No train anchor groups found in ranking dataset.")
    if len(val_ds) == 0:
        raise RuntimeError("No val anchor groups found in ranking dataset.")

    norm_stats = train_ds.compute_normalization_stats(sample_pairs=args.norm_sample_pairs)
    train_ds.set_norm_stats(norm_stats)
    val_ds.set_norm_stats(norm_stats)

    train_sampler = None
    train_shuffle = True
    if args.oversample_pos_groups:
        weights = []
        for group in train_ds.groups:
            has_pos = any(r.get("gt_label") == 1 for r in group)
            weights.append(2.0 if has_pos else 1.0)
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.anchors_per_batch,
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=ranking_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.anchors_per_batch,
        shuffle=False,
        collate_fn=ranking_collate_fn,
    )

    model_config = {
        "input_size": 8,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "endpoint_dim": args.endpoint_dim,
        "training_objective": "ranking",
        "pool_weight_first": args.pool_weight_first,
        "pool_weight_last": args.pool_weight_last,
        "pool_weight_mean": args.pool_weight_mean,
    }
    model = SiameseTransformerNetwork(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Train groups: {len(train_ds)}, Val groups: {len(val_ds)}")
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_top1 = -1.0
    best_state = None
    best_metrics: Dict[str, Any] = {}
    no_improve = 0

    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_gt = 0.0
        running_soft = 0.0
        running_sep = 0.0
        running_top1 = 0.0
        running_groups = 0

        mining_semi = 0
        mining_hard = 0
        mining_random = 0

        anneal_epochs = max(int(args.soft_anneal_epochs), 1)
        anneal_factor = max(0.0, 1.0 - (float(epoch) / float(anneal_epochs)))
        effective_soft_weight = float(args.soft_weight) * anneal_factor

        for pa, ma, pb, mb, ep, gt, bhat, group_slices, _anchor_ids in train_loader:
            pa = pa.to(device)
            ma = ma.to(device)
            pb = pb.to(device)
            mb = mb.to(device)
            ep = ep.to(device)
            gt = gt.to(device)
            bhat = bhat.to(device)

            optimizer.zero_grad()
            scores = model(pa, ma, pb, mb, ep).squeeze(1)

            enable_mining = epoch >= int(args.enable_mining_epoch)
            loss, stats = compute_weighted_pairwise_margin_loss(
                scores=scores,
                gt_labels=gt,
                bhat_costs=bhat,
                group_slices=group_slices,
                margin=float(args.margin),
                gt_weight=float(args.gt_weight),
                effective_soft_weight=float(effective_soft_weight),
                sep_weight=float(args.sep_weight),
                pos_margin=float(args.pos_margin),
                neg_margin=float(args.neg_margin),
                top1_weight=float(args.top1_weight),
                enable_mining=enable_mining,
                max_neg_per_pos=int(args.max_neg_per_pos),
                pos_pair_weight=float(args.pos_pair_weight),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            running_loss += stats.loss * max(stats.num_groups, 1)
            running_gt += stats.gt_term * max(stats.num_groups, 1)
            running_soft += stats.soft_term * max(stats.num_groups, 1)
            running_sep += stats.sep_term * max(stats.num_groups, 1)
            running_top1 += stats.top1_term * max(stats.num_groups, 1)
            running_groups += stats.num_groups
            mining_semi += stats.semi_hard_selected
            mining_hard += stats.hard_selected
            mining_random += stats.random_selected

        scheduler.step()

        val_metrics = evaluate_anchor_metrics(model, val_loader, device)
        val_top1 = float(val_metrics["anchor_top1_acc"])

        denom = max(running_groups, 1)
        train_loss = running_loss / denom
        train_gt = running_gt / denom
        train_soft = running_soft / denom
        train_sep = running_sep / denom
        train_top1 = running_top1 / denom
        total_mined = mining_semi + mining_hard + mining_random
        random_fallback_rate = float(mining_random / total_mined) if total_mined > 0 else 0.0

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_gt_term": float(train_gt),
            "train_soft_term": float(train_soft),
            "train_sep_term": float(train_sep),
            "train_top1_term": float(train_top1),
            "effective_soft_weight": float(effective_soft_weight),
            "val_pairwise_acc_gt_crossclass": float(val_metrics["pairwise_acc_gt_crossclass"]),
            "val_anchor_top1_acc": float(val_metrics["anchor_top1_acc"]),
            "val_mean_anchor_spearman": float(val_metrics["mean_anchor_spearman"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "semi_hard_selected": int(mining_semi),
            "hard_selected": int(mining_hard),
            "random_selected": int(mining_random),
            "random_fallback_rate": float(random_fallback_rate),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} gt={train_gt:.4f} soft={train_soft:.4f} "
            f"sep={train_sep:.4f} top1={train_top1:.4f} sw={effective_soft_weight:.4f} | "
            f"val_top1={val_metrics['anchor_top1_acc']:.4f} "
            f"val_pair={val_metrics['pairwise_acc_gt_crossclass']:.4f} "
            f"val_spear={val_metrics['mean_anchor_spearman']:.4f} | "
            f"fallback={random_fallback_rate:.3f}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = copy.deepcopy(epoch_record)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(args.patience):
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint state.")

    ckpt = {
        "model_state_dict": best_state,
        "model_config": model_config,
        "seq_mean": norm_stats["mean"],
        "seq_std": norm_stats["std"],
        "ep_mean": norm_stats["ep_mean"],
        "ep_std": norm_stats["ep_std"],
        "best_val_metrics": best_metrics,
        "history": history,
        "train_config": {
            "dataset_path": str(dataset_path),
            "anchors_per_batch": int(args.anchors_per_batch),
            "cands_per_anchor": int(args.cands_per_anchor),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "margin": float(args.margin),
            "gt_weight": float(args.gt_weight),
            "soft_weight": float(args.soft_weight),
            "sep_weight": float(args.sep_weight),
            "pos_margin": float(args.pos_margin),
            "neg_margin": float(args.neg_margin),
            "top1_weight": float(args.top1_weight),
            "soft_anneal_epochs": int(args.soft_anneal_epochs),
            "enable_mining_epoch": int(args.enable_mining_epoch),
            "max_neg_per_pos": int(args.max_neg_per_pos),
            "pos_pair_weight": float(args.pos_pair_weight),
            "oversample_pos_groups": bool(args.oversample_pos_groups),
            "seed": int(args.seed),
        },
    }

    torch.save(ckpt, output_path)
    print(f"Saved ranking checkpoint to: {output_path}")

    history_path = output_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump({"history": history, "best": best_metrics}, f, indent=2)
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()
