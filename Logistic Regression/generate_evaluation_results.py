"""
Generate Logistic Regression Evaluation Results Visualization

Creates a 2x2 evaluation plot matching the Siamese Network format:
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix (Normalized)
- Probability Score Distribution

Author: Raswanth Prasath
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score, accuracy_score
)
from sklearn.model_selection import train_test_split

# Configuration
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "lr_analysis_plots"
MODEL_PATH = SCRIPT_DIR / "model_artifacts" / "consensus_top10_full47.pkl"
DATA_PATH = SCRIPT_DIR / "training_dataset_advanced.npz"

# Use consistent styling
plt.style.use('seaborn-v0_8-whitegrid')


def load_model_and_data():
    """Load the trained LR model and dataset."""
    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['feature_names']

    # Load data
    data = np.load(DATA_PATH, allow_pickle=True)
    X = data['X']
    y = data['y']
    dataset_features = [str(f) for f in data['feature_names']]

    # Find feature indices for the model features
    feature_indices = [dataset_features.index(f) for f in features]

    # Select the features used by the model
    X_selected = X[:, feature_indices]

    return model, scaler, X_selected, y, features


def compute_metrics(model, scaler, X, y, test_size=0.2, random_state=42):
    """Compute evaluation metrics on a held-out test set."""
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Get predictions and probabilities
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class

    # Compute metrics
    metrics = {
        'labels': y_test,
        'predictions': y_pred,
        'probabilities': y_prob,
        'roc_auc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    return metrics


def plot_evaluation_results(metrics: dict, output_dir: Path):
    """Plot evaluation results in 2x2 grid format."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'])
    ax.plot(fpr, tpr, linewidth=2, color='#2171b5',
            label=f"Logistic Regression (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # Precision-Recall Curve
    ax = axes[0, 1]
    precision, recall, _ = precision_recall_curve(metrics['labels'], metrics['probabilities'])
    ax.plot(recall, precision, linewidth=2, color='#2171b5',
            label=f"Logistic Regression (AP={metrics['avg_precision']:.3f})")
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0.5, 1.02])

    # Confusion Matrix (Normalized)
    ax = axes[1, 0]
    cm = metrics['confusion_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Different', 'Same'], fontsize=11)
    ax.set_yticklabels(['Different', 'Same'], fontsize=11)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    # Annotate confusion matrix
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                   ha="center", va="center", color=text_color, fontsize=12, fontweight='bold')

    # Probability Score Distribution
    ax = axes[1, 1]
    pos_mask = metrics['labels'] == 1
    neg_mask = metrics['labels'] == 0

    ax.hist(metrics['probabilities'][pos_mask], bins=30, alpha=0.7,
            label='Same Vehicle', color='#2ca02c', density=True, edgecolor='white', linewidth=0.5)
    ax.hist(metrics['probabilities'][neg_mask], bins=30, alpha=0.7,
            label='Different Vehicle', color='#d62728', density=True, edgecolor='white', linewidth=0.5)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Probability Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Probability Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper center')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout(pad=2.0)

    # Save
    output_path = output_dir / 'evaluation_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nEvaluation plots saved to: {output_path}")
    return output_path


def print_metrics_summary(metrics: dict):
    """Print summary of evaluation metrics."""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION EVALUATION RESULTS")
    print("="*60)
    print(f"\nModel: 10-Feature Consensus (consensus_top10_full47.pkl)")
    print(f"\nTest Set Metrics:")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Average Precision: {metrics['avg_precision']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']*100:.2f}%")

    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives:  {tn:4d} (Different predicted as Different)")
    print(f"  False Positives: {fp:4d} (Different predicted as Same)")
    print(f"  False Negatives: {fn:4d} (Same predicted as Different)")
    print(f"  True Positives:  {tp:4d} (Same predicted as Same)")

    print(f"\nDerived Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("="*60)


def main():
    """Main function to generate evaluation results."""
    print("Loading model and data...")
    model, scaler, X, y, features = load_model_and_data()

    print(f"Model features ({len(features)}): {features}")
    print(f"Dataset size: {len(y)} pairs ({sum(y)} positive, {len(y)-sum(y)} negative)")

    print("\nComputing evaluation metrics...")
    metrics = compute_metrics(model, scaler, X, y)

    print_metrics_summary(metrics)

    print("\nGenerating evaluation plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = plot_evaluation_results(metrics, OUTPUT_DIR)

    print(f"\nDone! Evaluation results saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    main()
