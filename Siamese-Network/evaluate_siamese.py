"""
Evaluation Script for Siamese Network

Evaluates the trained Siamese network and compares with Logistic Regression baseline
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from siamese_dataset import TrajectoryPairDataset, collate_fn
from siamese_model import SiameseTrajectoryNetwork


class SiameseEvaluator:
    """Evaluation manager for Siamese Network"""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: Trained Siamese network
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> dict:
        """
        Evaluate model on dataset

        Args:
            data_loader: DataLoader for evaluation data

        Returns:
            Dictionary of metrics and predictions
        """
        all_similarities = []
        all_labels = []
        all_embeddings_a = []
        all_embeddings_b = []

        print("\nEvaluating model...")
        for seq_a, len_a, seq_b, len_b, labels in tqdm(data_loader):
            # Move to device
            seq_a = seq_a.to(self.device)
            len_a = len_a.to(self.device)
            seq_b = seq_b.to(self.device)
            len_b = len_b.to(self.device)

            # Forward pass
            similarity, emb_a, emb_b = self.model(seq_a, len_a, seq_b, len_b)

            # Store results
            all_similarities.extend(similarity.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embeddings_a.append(emb_a.cpu().numpy())
            all_embeddings_b.append(emb_b.cpu().numpy())

        # Convert to arrays
        similarities = np.array(all_similarities).squeeze()
        labels = np.array(all_labels).squeeze()
        embeddings_a = np.vstack(all_embeddings_a)
        embeddings_b = np.vstack(all_embeddings_b)

        # Compute metrics
        predictions = (similarities > 0.5).astype(int)

        metrics = {
            'similarities': similarities,
            'labels': labels,
            'predictions': predictions,
            'embeddings_a': embeddings_a,
            'embeddings_b': embeddings_b,
            'roc_auc': roc_auc_score(labels, similarities),
            'avg_precision': average_precision_score(labels, similarities),
            'accuracy': np.mean(predictions == labels),
            'confusion_matrix': confusion_matrix(labels, predictions)
        }

        return metrics

    def print_metrics(self, metrics: dict, dataset_name: str = "Test"):
        """Print evaluation metrics"""
        print("\n" + "="*60)
        print(f"{dataset_name} SET EVALUATION RESULTS")
        print("="*60)

        print(f"\nROC-AUC Score: {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['avg_precision']:.4f}")
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")

        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"                 Predicted")
        print(f"                 Neg    Pos")
        print(f"Actual  Neg  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
        print(f"        Pos  [{cm[1,0]:5d}  {cm[1,1]:5d}]")

        print("\nClassification Report:")
        print(classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=['Different Vehicle', 'Same Vehicle']
        ))


def compare_with_logistic_regression(siamese_metrics: dict, output_dir: Path):
    """
    Compare Siamese network with Logistic Regression baseline

    Args:
        siamese_metrics: Metrics from Siamese network
        output_dir: Directory to save comparison plots
    """
    # Load logistic regression results if available
    lr_dir = Path(r"D:\ASU Academics\Thesis & Research\02_Code\Logistic-Regression")
    lr_data_file = lr_dir / "training_dataset_advanced.npz"

    if not lr_data_file.exists():
        print("\nWarning: Logistic Regression results not found for comparison")
        return

    # For now, create placeholder comparison
    # In practice, you would load the trained LR model and evaluate on the same test set
    print("\n" + "="*60)
    print("MODEL COMPARISON: Siamese Network vs Logistic Regression")
    print("="*60)

    print("\nSiamese Network Performance:")
    print(f"  ROC-AUC: {siamese_metrics['roc_auc']:.4f}")
    print(f"  Accuracy: {siamese_metrics['accuracy']*100:.2f}%")
    print(f"  Average Precision: {siamese_metrics['avg_precision']:.4f}")

    print("\nNote: To complete comparison, evaluate LR model on same test set")


def plot_evaluation_results(metrics: dict, output_dir: Path):
    """Plot evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(metrics['labels'], metrics['similarities'])
    ax.plot(fpr, tpr, linewidth=2, label=f"Siamese Network (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Precision-Recall Curve
    ax = axes[0, 1]
    precision, recall, _ = precision_recall_curve(metrics['labels'], metrics['similarities'])
    ax.plot(recall, precision, linewidth=2, label=f"Siamese Network (AP={metrics['avg_precision']:.3f})")
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Confusion Matrix
    ax = axes[1, 0]
    cm = metrics['confusion_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Different', 'Same'])
    ax.set_yticklabels(['Different', 'Same'])
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    plt.colorbar(im, ax=ax)

    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                          ha="center", va="center",
                          color="black" if cm_normalized[i, j] < 0.5 else "white")

    # Similarity Distribution
    ax = axes[1, 1]
    pos_mask = metrics['labels'] == 1
    neg_mask = metrics['labels'] == 0
    ax.hist(metrics['similarities'][pos_mask], bins=50, alpha=0.6,
            label='Same Vehicle', color='green', density=True)
    ax.hist(metrics['similarities'][neg_mask], bins=50, alpha=0.6,
            label='Different Vehicle', color='red', density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Similarity Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nEvaluation plots saved to: {output_dir / 'evaluation_results.png'}")


def load_checkpoint(checkpoint_path: Path, model: nn.Module, device: torch.device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
    return model


def main():
    """Main evaluation function"""
    # Paths - use relative path for portability to Sol
    script_dir = Path(__file__).parent
    model_dir = script_dir / "outputs"
    checkpoint_path = model_dir / "best_accuracy.pth"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_siamese.py")
        return

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model (same architecture as training)
    model_config = {
        'input_size': 4,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'similarity_hidden_dim': 64
    }

    model = SiameseTrajectoryNetwork(**model_config)
    model = load_checkpoint(checkpoint_path, model, device)
    model = model.to(device)

    # Create test dataset (use one scenario for testing)
    print("\n" + "="*60)
    print("LOADING TEST DATASET")
    print("="*60)

    test_dataset = TrajectoryPairDataset(
        dataset_names=['iii'],  # Use scenario iii as test set
        normalize=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Evaluate
    evaluator = SiameseEvaluator(model, device)
    metrics = evaluator.evaluate(test_loader)

    # Print metrics
    evaluator.print_metrics(metrics, "Test")

    # Plot results
    plot_evaluation_results(metrics, model_dir)

    # Compare with logistic regression
    compare_with_logistic_regression(metrics, model_dir)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
