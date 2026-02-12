"""
Training Script for MLP Stitch Model

Trains the MLP on raw summary features extracted from trajectory fragment pairs.
Uses 5-fold cross-validation with early stopping.

Usage:
    conda activate i24
    python models/train_mlp.py
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))

from models.mlp_model import (
    MLPStitchModel, extract_pair_features, TOTAL_INPUT_DIM,
)


class MLPPairDataset(Dataset):
    """Pre-computed feature vectors for MLP training."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_pairs(dataset_names, data_dir, max_time_gap=5.0):
    """Load fragment pairs and extract MLP features."""
    data_dir = Path(data_dir)
    all_features = []
    all_labels = []

    for name in dataset_names:
        gt_path = data_dir / f"GT_{name}.json"
        raw_path = data_dir / f"RAW_{name}.json"

        print(f"Loading {name}...")
        with open(raw_path, 'r') as f:
            raw_data = json.load(f)
        print(f"  {len(raw_data)} fragments")

        # Group by GT id
        vehicle_frags = defaultdict(list)
        for frag in raw_data:
            gt_id = _get_gt_id(frag)
            if gt_id:
                vehicle_frags[gt_id].append(frag)

        # Positive pairs
        positives = []
        for gt_id, frags in vehicle_frags.items():
            frags_sorted = sorted(frags, key=lambda f: f.get('first_timestamp', f['timestamp'][0]))
            for i in range(len(frags_sorted) - 1):
                fa, fb = frags_sorted[i], frags_sorted[i + 1]
                t_end = fa.get('last_timestamp', fa['timestamp'][-1])
                t_start = fb.get('first_timestamp', fb['timestamp'][0])
                if t_end < t_start and (t_start - t_end) <= max_time_gap:
                    if fa.get('direction') == fb.get('direction'):
                        positives.append((fa, fb))
        print(f"  Positives: {len(positives)}")

        # Negative pairs (balanced)
        import random
        negatives = []
        target = len(positives)
        attempts = 0
        while len(negatives) < target and attempts < target * 50:
            attempts += 1
            ia = random.randint(0, len(raw_data) - 1)
            ib = random.randint(0, len(raw_data) - 1)
            if ia == ib:
                continue
            fa, fb = raw_data[ia], raw_data[ib]
            ga, gb = _get_gt_id(fa), _get_gt_id(fb)
            if ga is None or gb is None or ga == gb:
                continue
            if fa.get('direction') != fb.get('direction'):
                continue
            t_end = fa.get('last_timestamp', fa['timestamp'][-1])
            t_start = fb.get('first_timestamp', fb['timestamp'][0])
            if t_end < t_start and (t_start - t_end) <= max_time_gap * 2:
                negatives.append((fa, fb))
        print(f"  Negatives: {len(negatives)}")

        # Extract features
        for fa, fb in positives:
            all_features.append(extract_pair_features(fa, fb))
            all_labels.append(1)
        for fa, fb in negatives:
            all_features.append(extract_pair_features(fa, fb))
            all_labels.append(0)

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)

    # Replace NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"\nTotal: {len(labels)} pairs, {features.shape[1]} features")
    print(f"  Positive: {labels.sum():.0f}, Negative: {(1 - labels).sum():.0f}")
    return features, labels


def _get_gt_id(fragment):
    if 'gt_ids' in fragment and len(fragment['gt_ids']) > 0:
        first = fragment['gt_ids'][0]
        if isinstance(first, list) and len(first) > 0:
            if isinstance(first[0], dict) and '$oid' in first[0]:
                return first[0]['$oid']
    return None


def train_fold(model, train_loader, val_loader, device, epochs=80, patience=20, lr=0.001):
    """Train one fold with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.BCELoss()

    best_val_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(feats)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                preds = model(feats)
                val_loss += criterion(preds, labels).item() * len(labels)
                val_preds.extend(preds.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
        val_loss /= len(val_loader.dataset)

        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_AUC={val_auc:.4f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return best_state, best_val_auc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = project_root / "Siamese-Network" / "data"
    features, labels = load_pairs(['i', 'ii', 'iii'], data_dir)

    # Compute and save normalization stats for inference
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-6

    # 5-fold cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n{'=' * 60}")
    print(f"5-Fold Cross-Validation")
    print(f"{'=' * 60}")

    best_overall_auc = 0.0
    best_overall_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"\nFold {fold + 1}/{n_folds}")

        train_ds = MLPPairDataset(features[train_idx], labels[train_idx])
        val_ds = MLPPairDataset(features[val_idx], labels[val_idx])
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128)

        model = MLPStitchModel(input_dim=TOTAL_INPUT_DIM).to(device)
        best_state, best_auc = train_fold(model, train_loader, val_loader, device)

        # Final evaluation on this fold's val set
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        all_preds, all_labels_v = [], []
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats = feats.to(device)
                preds = model(feats)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels_v.extend(lbls.numpy().flatten())

        auc = roc_auc_score(all_labels_v, all_preds)
        ap = average_precision_score(all_labels_v, all_preds)
        acc = accuracy_score(all_labels_v, [1 if p > 0.5 else 0 for p in all_preds])

        fold_results.append({'auc': auc, 'ap': ap, 'acc': acc})
        print(f"  Fold {fold+1} Results: AUC={auc:.4f}, AP={ap:.4f}, Acc={acc:.4f}")

        if auc > best_overall_auc:
            best_overall_auc = auc
            best_overall_state = best_state

    # Summary
    print(f"\n{'=' * 60}")
    print("Cross-Validation Summary")
    print(f"{'=' * 60}")
    for metric in ['auc', 'ap', 'acc']:
        values = [r[metric] for r in fold_results]
        print(f"  {metric.upper()}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    # Save best model
    output_dir = project_root / "models" / "outputs"
    output_dir.mkdir(exist_ok=True)

    save_path = output_dir / "mlp_stitch_model.pth"
    torch.save({
        'model_state_dict': best_overall_state,
        'input_dim': TOTAL_INPUT_DIM,
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'cv_results': fold_results,
    }, save_path)
    print(f"\nSaved best model to {save_path}")

    # Also train a final model on all data
    print(f"\nTraining final model on all data...")
    full_ds = MLPPairDataset(features, labels)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=True)
    final_model = MLPStitchModel(input_dim=TOTAL_INPUT_DIM).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(60):
        final_model.train()
        for feats, lbls in full_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(final_model(feats), lbls)
            loss.backward()
            optimizer.step()

    final_path = output_dir / "mlp_stitch_final.pth"
    torch.save({
        'model_state_dict': {k: v.cpu() for k, v in final_model.state_dict().items()},
        'input_dim': TOTAL_INPUT_DIM,
        'feat_mean': feat_mean,
        'feat_std': feat_std,
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
