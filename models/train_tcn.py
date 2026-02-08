"""
Training Script for Siamese TCN Model

Uses 8-feature raw sequences with combined BCE + Contrastive loss.
CosineAnnealingLR, 80 epochs, early stopping patience=15.

Usage:
    conda activate i24
    python models/train_tcn.py
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

from models.tcn_model import SiameseTCN
from models.rich_sequence_dataset import RichSequenceDataset, rich_collate_fn


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


def train_epoch(model, loader, optimizer, device, bce_weight=0.5, contrastive_weight=0.5):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    bce_fn = nn.BCELoss()
    cont_fn = ContrastiveLoss(margin=2.0)

    for batch in loader:
        pa, la, ma, pb, lb, mb, eps, labels = batch
        pa = pa.to(device)
        pb = pb.to(device)
        eps = eps.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        similarity = model(pa, pb, eps)
        emb_a, emb_b = model.get_embeddings(pa, pb)

        bce = bce_fn(similarity, labels)
        cont = cont_fn(emb_a, emb_b, labels)
        loss = bce_weight * bce + contrastive_weight * cont

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
            pa, pb, eps, labels = pa.to(device), pb.to(device), eps.to(device), labels.to(device)

            similarity = model(pa, pb, eps)
            total_loss += bce_fn(similarity, labels).item() * len(labels)
            all_preds.extend(similarity.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    loss = total_loss / len(all_labels)

    return {'loss': loss, 'auc': auc, 'ap': ap, 'acc': acc}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    dataset = RichSequenceDataset(dataset_names=['i', 'ii', 'iii'], normalize=True)
    print(f"Dataset: {len(dataset)} pairs")

    # Split
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=rich_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=rich_collate_fn)

    # Model
    model_config = {
        'input_size': 8,
        'channels': (32, 64, 64),
        'kernel_size': 3,
        'dropout': 0.2,
        'endpoint_dim': 4,
    }
    model = SiameseTCN(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TCN parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    # Training loop
    best_val_auc = 0.0
    best_state = None
    patience = 15
    no_improve = 0

    print(f"\n{'=' * 60}")
    print("Training Siamese TCN")
    print(f"{'=' * 60}")

    for epoch in range(80):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_metrics['loss']:.4f} val_AUC={val_metrics['auc']:.4f} "
                  f"val_acc={val_metrics['acc']:.3f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save
    output_dir = project_root / "models" / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Final evaluation
    model.load_state_dict(best_state)
    model.to(device)
    final = evaluate(model, val_loader, device)
    print(f"\nBest model: AUC={final['auc']:.4f}, AP={final['ap']:.4f}, Acc={final['acc']:.4f}")

    save_path = output_dir / "tcn_stitch_model.pth"
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
