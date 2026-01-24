"""
Train and Evaluate Logistic Regression Models

Compares:
1. Baseline (basic features)
2. Advanced (with Bhattacharyya distance and projection features)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler

# Paths
WORKING_DIR = Path(r"D:\ASU Academics\Thesis & Research\02_Code\Logistic-Regression")


def load_dataset(filename):
    """Load dataset from npz file"""
    data = np.load(WORKING_DIR / filename, allow_pickle=True)
    return data['X'], data['y'], list(data['feature_names'])


def train_and_evaluate_model(X, y, feature_names, model_name="Model"):
    """Train and evaluate a logistic regression model"""
    print("\n" + "="*60)
    print(f"TRAINING: {model_name}")
    print("="*60)
    print(f"Dataset size: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Positive samples: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.2f}%)")
    print(f"Negative samples: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.2f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\nTraining logistic regression...")
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        C=1.0
    )
    model.fit(X_train_scaled, y_train)

    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluation metrics
    print("\n" + "-"*60)
    print("TEST SET PERFORMANCE")
    print("-"*60)
    print(classification_report(y_test, y_pred, target_names=['Different Vehicle', 'Same Vehicle']))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Average Precision
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision Score: {avg_precision:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"Actual  Neg  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
    print(f"        Pos  [{cm[1,0]:5d}  {cm[1,1]:5d}]")

    # Feature importance
    print("\n" + "-"*60)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("-"*60)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

    print(feature_importance.head(15).to_string(index=False))

    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'feature_importance': feature_importance,
        'confusion_matrix': cm
    }


def plot_comparison(results_basic, results_advanced):
    """Plot comparison between basic and advanced models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ROC Curves
    ax = axes[0, 0]
    fpr_basic, tpr_basic, _ = roc_curve(results_basic['y_test'], results_basic['y_pred_proba'])
    fpr_adv, tpr_adv, _ = roc_curve(results_advanced['y_test'], results_advanced['y_pred_proba'])

    ax.plot(fpr_basic, tpr_basic, label=f"Basic (AUC={results_basic['roc_auc']:.3f})", linewidth=2)
    ax.plot(fpr_adv, tpr_adv, label=f"Advanced (AUC={results_advanced['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Precision-Recall Curves
    ax = axes[0, 1]
    prec_basic, rec_basic, _ = precision_recall_curve(results_basic['y_test'], results_basic['y_pred_proba'])
    prec_adv, rec_adv, _ = precision_recall_curve(results_advanced['y_test'], results_advanced['y_pred_proba'])

    ax.plot(rec_basic, prec_basic, label=f"Basic (AP={results_basic['avg_precision']:.3f})", linewidth=2)
    ax.plot(rec_adv, prec_adv, label=f"Advanced (AP={results_advanced['avg_precision']:.3f})", linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Confusion Matrices
    ax = axes[1, 0]
    cm_basic = results_basic['confusion_matrix']
    cm_normalized = cm_basic.astype('float') / cm_basic.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.set_title('Basic Model - Confusion Matrix', fontsize=12, fontweight='bold')
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Different', 'Same'])
    ax.set_yticklabels(['Different', 'Same'])
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm_basic[i, j]}\n({cm_normalized[i, j]:.2%})',
                          ha="center", va="center", color="black" if cm_normalized[i, j] < 0.5 else "white")

    ax = axes[1, 1]
    cm_adv = results_advanced['confusion_matrix']
    cm_normalized = cm_adv.astype('float') / cm_adv.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.set_title('Advanced Model - Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Different', 'Same'])
    ax.set_yticklabels(['Different', 'Same'])
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm_adv[i, j]}\n({cm_normalized[i, j]:.2%})',
                          ha="center", va="center", color="black" if cm_normalized[i, j] < 0.5 else "white")

    plt.tight_layout()
    plt.savefig(WORKING_DIR / 'outputs' / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nPlot saved to: {WORKING_DIR / 'outputs' / 'model_comparison.png'}")


def main():
    """Main execution"""
    print("="*60)
    print("LOGISTIC REGRESSION MODEL TRAINING & EVALUATION")
    print("="*60)

    # Create outputs directory
    (WORKING_DIR / 'outputs').mkdir(exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    X_basic, y_basic, features_basic = load_dataset("training_dataset_combined.npz")
    X_advanced, y_advanced, features_advanced = load_dataset("training_dataset_advanced.npz")

    # Train and evaluate basic model
    results_basic = train_and_evaluate_model(X_basic, y_basic, features_basic, "BASIC MODEL (28 features)")

    # Train and evaluate advanced model
    results_advanced = train_and_evaluate_model(X_advanced, y_advanced, features_advanced, "ADVANCED MODEL (47 features)")

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Metric':<30} {'Basic':<15} {'Advanced':<15} {'Improvement':<15}")
    print("-"*60)

    metrics = [
        ('ROC-AUC', results_basic['roc_auc'], results_advanced['roc_auc']),
        ('Average Precision', results_basic['avg_precision'], results_advanced['avg_precision']),
    ]

    for metric_name, basic_val, adv_val in metrics:
        improvement = ((adv_val - basic_val) / basic_val * 100) if basic_val > 0 else 0
        print(f"{metric_name:<30} {basic_val:<15.4f} {adv_val:<15.4f} {improvement:>+7.2f}%")

    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(results_basic, results_advanced)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
