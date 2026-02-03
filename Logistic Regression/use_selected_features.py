"""
Practical Script: Train Logistic Regression with Selected Features

Based on feature evaluation results, this script trains models using optimal feature subsets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

# Configuration
WORKING_DIR = Path("/home/raswanth/I24/I24-postprocessing-lite/Logistic Regression")
OUTPUT_DIR = WORKING_DIR / "model_artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)


# Recommended feature subsets from evaluation
FEATURE_SUBSETS = {
    'advanced_minimal_5': [
        'projection_error_x_max',
        'y_diff',
        'projection_error_y_max',
        'bhattacharyya_coeff',
        'length_diff'
    ],
    'advanced_optimal_20': None,  # Load from permutation importance
    'combined_minimal_5': [
        'y_diff',
        'spatial_gap',
        'duration_a',
        'length_a',
        'vel_b_mean'
    ],
    'combined_optimal_10': None  # Load from permutation importance
}


def load_top_features_from_permutation(n_features=20):
    """Load top N features from permutation importance results"""
    perm_file = WORKING_DIR / 'feature_selection_outputs' / 'permutation_importance.csv'

    if not perm_file.exists():
        print(f"Warning: {perm_file} not found. Run feature_evaluation.py first.")
        return None

    perm_df = pd.read_csv(perm_file)
    return perm_df.head(n_features)['feature'].tolist()


def train_model_with_selected_features(dataset='advanced', feature_subset='optimal'):
    """
    Train logistic regression with selected features

    Args:
        dataset: 'advanced' (47 features) or 'combined' (28 features)
        feature_subset: 'minimal' (5 features), 'optimal' (10-20 features), or list of feature names
    """
    print("="*80)
    print(f"TRAINING MODEL: {dataset.upper()} dataset, {feature_subset} features")
    print("="*80)

    # Load dataset
    if dataset == 'advanced':
        data_file = WORKING_DIR / 'training_dataset_advanced.npz'
        if feature_subset == 'minimal':
            selected_features = FEATURE_SUBSETS['advanced_minimal_5']
        elif feature_subset == 'optimal':
            selected_features = load_top_features_from_permutation(20)
        else:
            selected_features = feature_subset
    else:  # combined
        data_file = WORKING_DIR / 'training_dataset_combined.npz'
        if feature_subset == 'minimal':
            selected_features = FEATURE_SUBSETS['combined_minimal_5']
        elif feature_subset == 'optimal':
            selected_features = load_top_features_from_permutation(10)
        else:
            selected_features = feature_subset

    if selected_features is None:
        print("Error: Could not load feature subset")
        return None

    # Load data
    print(f"\nLoading {data_file.name}...")
    data = np.load(data_file, allow_pickle=True)
    X_full = data['X']
    y = data['y']
    all_features = [str(f) for f in data['feature_names']]

    print(f"Full dataset: {X_full.shape[0]} samples, {X_full.shape[1]} features")

    # Select features
    feature_indices = [all_features.index(f) for f in selected_features]
    X = X_full[:, feature_indices]

    print(f"\nSelected features ({len(selected_features)}):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\nTraining logistic regression...")
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    print("\n" + "-"*80)
    print("EVALUATION RESULTS")
    print("-"*80)

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=5, scoring='roc_auc', n_jobs=-1
    )
    print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")

    # Test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    print(f"\nTest Set Performance:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Different', 'Same']))

    # Feature coefficients
    print("\n" + "-"*80)
    print("FEATURE COEFFICIENTS")
    print("-"*80)
    coef_df = pd.DataFrame({
        'feature': selected_features,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

    print(coef_df.to_string(index=False))

    # Save model
    model_name = f"{dataset}_{feature_subset}_{len(selected_features)}features"
    model_file = OUTPUT_DIR / f"{model_name}.pkl"

    model_package = {
        'model': model,
        'scaler': scaler,
        'features': selected_features,
        'feature_indices': feature_indices,
        'all_features': all_features,
        'performance': {
            'cv_roc_auc': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std(),
            'test_roc_auc': roc_auc,
            'test_avg_precision': avg_precision
        }
    }

    with open(model_file, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"\n{'='*80}")
    print(f"Model saved to: {model_file}")
    print(f"{'='*80}")

    return model_package


def demonstrate_usage():
    """Demonstrate how to use the saved model"""
    print("\n" + "="*80)
    print("USAGE EXAMPLE: HOW TO USE SAVED MODEL IN PIPELINE")
    print("="*80)

    example_code = '''
# Load the saved model
import pickle

# Load model package
with open('model_artifacts/combined_optimal_10features.pkl', 'rb') as f:
    model_pkg = pickle.load(f)

model = model_pkg['model']
scaler = model_pkg['scaler']
selected_features = model_pkg['features']
all_features = model_pkg['all_features']

# Example: Extract features from two fragments
def compute_fragment_pair_features(frag_a, frag_b):
    """Extract features for a fragment pair"""
    from enhanced_dataset_creation import extract_all_features

    # Extract all features
    all_feats = extract_all_features(frag_a, frag_b)

    # Select only the features used by the model
    selected_vals = [all_feats[f] for f in selected_features]

    return np.array(selected_vals).reshape(1, -1)

# Use model for prediction
features = compute_fragment_pair_features(fragment_a, fragment_b)
features_scaled = scaler.transform(features)
similarity_score = model.predict_proba(features_scaled)[0, 1]

# Decision threshold
if similarity_score > 0.5:
    print(f"Fragments should be stitched (confidence: {similarity_score:.3f})")
else:
    print(f"Fragments are different vehicles (confidence: {1-similarity_score:.3f})")

# Or convert to cost (like Bhattacharyya)
cost = (1 - similarity_score) * 5  # Scale to match Bhattacharyya range
if cost < stitch_thresh:
    # Stitch the fragments
    pass
'''

    print(example_code)


def main():
    """Main execution"""
    print("="*80)
    print("TRAINING MODELS WITH SELECTED FEATURES")
    print("="*80)

    # Train recommended models
    models_to_train = [
        ('combined', 'minimal'),   # 5 features
        ('combined', 'optimal'),   # 10 features
        # ('advanced', 'minimal'),   # 5 features (uncomment if you have advanced dataset)
        # ('advanced', 'optimal'),   # 20 features
    ]

    results = {}

    for dataset, subset in models_to_train:
        try:
            model_pkg = train_model_with_selected_features(dataset, subset)
            if model_pkg:
                results[f"{dataset}_{subset}"] = model_pkg
            print()
        except Exception as e:
            print(f"\nError training {dataset} {subset}: {e}\n")

    # Summary comparison
    if results:
        print("\n" + "="*80)
        print("SUMMARY COMPARISON")
        print("="*80)

        print(f"\n{'Model':<30} {'Features':<10} {'CV ROC-AUC':<15} {'Test ROC-AUC':<15}")
        print("-"*80)

        for name, pkg in results.items():
            perf = pkg['performance']
            n_feats = len(pkg['features'])
            print(f"{name:<30} {n_feats:<10} "
                  f"{perf['cv_roc_auc']:.4f}          {perf['test_roc_auc']:.4f}")

    # Show usage example
    demonstrate_usage()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll models saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
