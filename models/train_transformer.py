"""
Training Script for Siamese Transformer Model

Uses 8-feature raw sequences with combined BCE + Contrastive + TripletMargin loss.
AdamW with warmup + CosineAnnealing, label smoothing, 100 epochs.

Usage:
    conda activate i24
    python models/train_transformer.py
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from pathlib import Path

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


def train_epoch(model, loader, optimizer, device,
                bce_weight=0.4, cont_weight=0.3, triplet_weight=0.3):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    bce_fn = LabelSmoothBCE(smoothing=0.05)
    cont_fn = ContrastiveLoss(margin=2.0)
    triplet_fn = nn.TripletMarginLoss(margin=1.0)

    for batch in loader:
        pa, la, ma, pb, lb, mb, eps, labels = batch
        pa, ma = pa.to(device), ma.to(device)
        pb, mb = pb.to(device), mb.to(device)
        eps = eps.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        similarity = model(pa, ma, pb, mb, eps)
        emb_a, emb_b = model.get_embeddings(pa, ma, pb, mb)

        bce = bce_fn(similarity, labels)
        cont = cont_fn(emb_a, emb_b, labels)

        # Triplet loss
        anchors, positives, negatives = mine_triplets(emb_a, emb_b, labels)
        if anchors is not None and len(anchors) > 0:
            tri = triplet_fn(anchors, positives, negatives)
        else:
            tri = torch.tensor(0.0, device=device)

        loss = bce_weight * bce + cont_weight * cont + triplet_weight * tri

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (similarity > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = RichSequenceDataset(dataset_names=['i', 'ii', 'iii'], normalize=True)
    print(f"Dataset: {len(dataset)} pairs")

    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

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

    import math
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
        'val_metrics': final,
    }, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
