"""
Training Script for Siamese Transformer Model

Uses 8-feature raw sequences with combined BCE + Contrastive + TripletMargin loss.
AdamW with warmup + CosineAnnealing, label smoothing, 100 epochs.

Usage:
    conda activate i24
    python models/train_transformer.py
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from pathlib import Path
from typing import List, Tuple, Set

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))

from models.transformer_model import SiameseTransformerNetwork
from models.rich_sequence_dataset import RichSequenceDataset, rich_collate_fn


class LabelSmoothBCE(nn.Module):
    """BCE with label smoothing to reduce overconfidence."""

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(pred, target)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, labels):
        distances = F.pairwise_distance(emb_a, emb_b)
        labels = labels.squeeze()
        loss_sim = labels * distances.pow(2)
        loss_dis = (1 - labels) * F.relu(self.margin - distances).pow(2)
        return (loss_sim + loss_dis).mean()


def mine_triplets(emb_a, emb_b, labels):
    """
    Mine semi-hard triplets from a batch.

    For each positive pair (anchor, positive), find a negative with
    distance > d(anchor, positive) but < d(anchor, positive) + margin.
    Falls back to hardest negative if no semi-hard found.
    """
    labels_flat = labels.squeeze()
    pos_mask = labels_flat == 1
    neg_mask = labels_flat == 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return None, None, None

    # Use emb_a as anchors, emb_b as candidates
    anchors = emb_a[pos_mask]
    positives = emb_b[pos_mask]
    neg_embs = emb_b[neg_mask]

    if len(anchors) == 0 or len(neg_embs) == 0:
        return None, None, None

    # For each anchor, find hardest negative
    # Compute pairwise distances: (n_pos, n_neg)
    dist_neg = torch.cdist(anchors, neg_embs)  # (n_pos, n_neg)
    hardest_neg_idx = dist_neg.argmin(dim=1)  # (n_pos,)
    negatives = neg_embs[hardest_neg_idx]

    return anchors, positives, negatives


def evaluate(model, loader, device):
    """Evaluate model on a loader with samples already normalized by the dataset."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    bce_fn = nn.BCELoss()

    with torch.no_grad():
        for batch in loader:
            pa, la, ma, pb, lb, mb, eps, labels = batch
            pa, ma = pa.to(device), ma.to(device)
            pb, mb = pb.to(device), mb.to(device)
            eps, labels = eps.to(device), labels.to(device)

            similarity = model(pa, ma, pb, mb, eps)
            total_loss += bce_fn(similarity, labels).item() * len(labels)
            all_preds.extend(similarity.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        'loss': total_loss / len(all_labels),
        'auc': roc_auc_score(all_labels, all_preds),
        'ap': average_precision_score(all_labels, all_preds),
        'acc': accuracy_score(all_labels, (all_preds > 0.5).astype(int)),
    }


def vehicle_disjoint_split(
    dataset: RichSequenceDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int], int]:
    """
    Split by vehicle IDs so no vehicle appears in both train and validation.
    Cross-split negative pairs are dropped.
    """
    all_vehicle_ids: Set[str] = set()
    for i in range(len(dataset)):
        ga, gb = dataset.get_pair_vehicle_ids(i)
        if ga is not None:
            all_vehicle_ids.add(str(ga))
        if gb is not None:
            all_vehicle_ids.add(str(gb))

    vehicle_ids = sorted(all_vehicle_ids)
    if len(vehicle_ids) < 2:
        raise RuntimeError("Need at least 2 vehicles for disjoint split.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vehicle_ids))
    vehicle_ids = [vehicle_ids[i] for i in perm]

    n_val = max(1, int(len(vehicle_ids) * val_ratio))
    n_val = min(n_val, len(vehicle_ids) - 1)
    val_vehicles = set(vehicle_ids[:n_val])
    train_vehicles = set(vehicle_ids[n_val:])

    train_indices: List[int] = []
    val_indices: List[int] = []
    dropped = 0

    for i in range(len(dataset)):
        ga, gb = dataset.get_pair_vehicle_ids(i)
        ga = str(ga) if ga is not None else None
        gb = str(gb) if gb is not None else None
        if ga is None or gb is None:
            dropped += 1
            continue
        if ga in train_vehicles and gb in train_vehicles:
            train_indices.append(i)
        elif ga in val_vehicles and gb in val_vehicles:
            val_indices.append(i)
        else:
            dropped += 1

    if not train_indices or not val_indices:
        raise RuntimeError(
            f"Vehicle-disjoint split produced empty partition "
            f"(train={len(train_indices)}, val={len(val_indices)})."
        )

    return train_indices, val_indices, dropped


def _count_labels(dataset: RichSequenceDataset, indices: List[int]) -> Tuple[int, int]:
    pos = sum(1 for i in indices if dataset.get_pair_label(i) == 1)
    neg = len(indices) - pos
    return pos, neg


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = RichSequenceDataset(
        dataset_names=['i', 'ii', 'iii'],
        normalize=False,
        random_seed=42,
        negative_sampling_mode="stratified",
        negative_gap_bins=(5.0, 10.0, 15.0),
        negative_bin_weights=(0.4, 0.35, 0.25),
        negative_time_gap_max=15.0,
    )
    print(f"Dataset: {len(dataset)} pairs")

    train_indices, val_indices, dropped = vehicle_disjoint_split(dataset, val_ratio=0.2, seed=42)
    train_pos, train_neg = _count_labels(dataset, train_indices)
    val_pos, val_neg = _count_labels(dataset, val_indices)
    print(
        f"Vehicle-disjoint split: train={len(train_indices)} (pos={train_pos}, neg={train_neg}), "
        f"val={len(val_indices)} (pos={val_pos}, neg={val_neg}), dropped_cross_split={dropped}"
    )

    norm_stats = dataset.compute_normalization_stats_from_indices(
        train_indices,
        sample_size=1000,
        balanced=True,
    )
    dataset.set_normalization_stats(norm_stats)
    dataset.normalize = True

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=rich_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=rich_collate_fn)

    model_config = {
        'input_size': 8,
        'd_model': 64,
        'nhead': 4,
        'dim_feedforward': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'endpoint_dim': 4,
        'pool_weight_first': 0.2,
        'pool_weight_last': 0.5,
        'pool_weight_mean': 0.3,
    }
    model = SiameseTransformerNetwork(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Transformer parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    # Warmup + CosineAnnealing
    warmup_steps = 500
    total_steps = 100 * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    best_val_auc = 0.0
    best_state = None
    patience = 20
    no_improve = 0
    step = 0

    print(f"\n{'=' * 60}")
    print("Training Siamese Transformer")
    print(f"{'=' * 60}")

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        bce_fn = LabelSmoothBCE(smoothing=0.05)
        cont_fn = ContrastiveLoss(margin=2.0)
        triplet_fn = nn.TripletMarginLoss(margin=1.0)

        for batch in train_loader:
            pa, la, ma, pb, lb, mb, eps, labels = batch
            pa, ma = pa.to(device), ma.to(device)
            pb, mb = pb.to(device), mb.to(device)
            eps, labels = eps.to(device), labels.to(device)

            optimizer.zero_grad()
            similarity = model(pa, ma, pb, mb, eps)
            emb_a, emb_b = model.get_embeddings(pa, ma, pb, mb)

            bce = bce_fn(similarity, labels)
            cont = cont_fn(emb_a, emb_b, labels)
            anchors, positives, negatives = mine_triplets(emb_a, emb_b, labels)
            tri = triplet_fn(anchors, positives, negatives) if anchors is not None and len(anchors) > 0 else torch.tensor(0.0, device=device)
            loss = 0.4 * bce + 0.3 * cont + 0.3 * tri

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            total_loss += loss.item() * len(labels)
            preds = (similarity > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_loss = total_loss / total
        train_acc = correct / total
        val_metrics = evaluate(model, val_loader, device)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_AUC={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.3f} lr={lr:.6f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save
    output_dir = project_root / "models" / "outputs"
    output_dir.mkdir(exist_ok=True)

    model.load_state_dict(best_state)
    model.to(device)
    final = evaluate(model, val_loader, device)
    print(f"\nBest model: AUC={final['auc']:.4f}, AP={final['ap']:.4f}, Acc={final['acc']:.4f}")

    save_path = output_dir / "transformer_stitch_model.pth"
    torch.save({
        'model_state_dict': best_state,
        'model_config': model_config,
        'seq_mean': dataset.norm_stats['mean'] if dataset.norm_stats else np.zeros(8),
        'seq_std': dataset.norm_stats['std'] if dataset.norm_stats else np.ones(8),
        'ep_mean': dataset.norm_stats['ep_mean'] if dataset.norm_stats else np.zeros(4),
        'ep_std': dataset.norm_stats['ep_std'] if dataset.norm_stats else np.ones(4),
        'val_metrics': final,
    }, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
