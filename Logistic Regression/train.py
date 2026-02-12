import argparse
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "Logistic Regression/data/training_dataset_advanced.npz"
LEGACY_DATASET = "Logistic Regression/training_dataset_advanced.npz"


def resolve_existing_path(path_str, fallbacks=None):
    """Resolve an existing file path with backward-compatible fallbacks."""
    candidates = []
    p = Path(path_str)
    candidates.append(p)
    if not p.is_absolute():
        candidates.append(ROOT / p)

    for fb in fallbacks or []:
        fbp = Path(fb)
        candidates.append(fbp)
        if not fbp.is_absolute():
            candidates.append(ROOT / fbp)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Could not resolve existing path for: {path_str}")


def build_eval_pipeline(C):
    """Create a leakage-safe evaluation pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, solver='lbfgs', C=C)),
    ])


def evaluate_protocols(X_subset, y, C, eval_protocol='random', source_split_tag=None):
    """
    Evaluate model using random split and/or source-holdout split.

    Returns:
        dict with keys in {'random', 'source_holdout'}.
        Each value includes train/test metrics and CV ROC-AUC.
    """
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if eval_protocol in ('random', 'both'):
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=42, stratify=y
        )
        eval_pipe = build_eval_pipeline(C)
        eval_pipe.fit(X_train, y_train)

        y_proba_train = eval_pipe.predict_proba(X_train)[:, 1]
        y_proba_test = eval_pipe.predict_proba(X_test)[:, 1]
        logits_test = eval_pipe.decision_function(X_test)
        cv_scores = cross_val_score(eval_pipe, X_subset, y, cv=cv, scoring='roc_auc')

        results['random'] = {
            'train_roc_auc': float(roc_auc_score(y_train, y_proba_train)),
            'test_roc_auc': float(roc_auc_score(y_test, y_proba_test)),
            'train_ap': float(average_precision_score(y_train, y_proba_train)),
            'test_ap': float(average_precision_score(y_test, y_proba_test)),
            'cv_roc_auc_mean': float(cv_scores.mean()),
            'cv_roc_auc_std': float(cv_scores.std()),
            'logits_test': logits_test,
        }

    if eval_protocol in ('source_holdout', 'both'):
        if source_split_tag is None:
            print("Warning: source_holdout requested but source_split_tag not found; skipping.")
        else:
            tags = np.asarray(source_split_tag).astype(str)
            unique_tags, counts = np.unique(tags, return_counts=True)
            if len(unique_tags) < 2:
                print("Warning: source_holdout requested but fewer than 2 source tags; skipping.")
            else:
                if {'advanced_keepall', 'v4_diverse_curated'}.issubset(set(unique_tags)):
                    train_tag = 'advanced_keepall'
                    test_tag = 'v4_diverse_curated'
                else:
                    # Fallback: use two largest source partitions.
                    order = np.argsort(counts)[::-1]
                    train_tag = unique_tags[order[0]]
                    test_tag = unique_tags[order[1]]

                train_mask = tags == train_tag
                test_mask = tags == test_tag
                X_train, y_train = X_subset[train_mask], y[train_mask]
                X_test, y_test = X_subset[test_mask], y[test_mask]

                if (len(np.unique(y_train)) < 2) or (len(np.unique(y_test)) < 2):
                    print("Warning: source_holdout split has a single class in train/test; skipping.")
                else:
                    eval_pipe = build_eval_pipeline(C)
                    eval_pipe.fit(X_train, y_train)

                    y_proba_train = eval_pipe.predict_proba(X_train)[:, 1]
                    y_proba_test = eval_pipe.predict_proba(X_test)[:, 1]
                    logits_test = eval_pipe.decision_function(X_test)
                    # CV is measured on training source only.
                    cv_scores = cross_val_score(eval_pipe, X_train, y_train, cv=cv, scoring='roc_auc')

                    results['source_holdout'] = {
                        'train_roc_auc': float(roc_auc_score(y_train, y_proba_train)),
                        'test_roc_auc': float(roc_auc_score(y_test, y_proba_test)),
                        'train_ap': float(average_precision_score(y_train, y_proba_train)),
                        'test_ap': float(average_precision_score(y_test, y_proba_test)),
                        'cv_roc_auc_mean': float(cv_scores.mean()),
                        'cv_roc_auc_std': float(cv_scores.std()),
                        'logits_test': logits_test,
                        'train_tag': train_tag,
                        'test_tag': test_tag,
                    }

    if not results:
        raise RuntimeError("No evaluation protocol could be executed. Check metadata and flags.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train top-10 consensus LR model")
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help='Path to training dataset NPZ file '
                             '(canonical: Logistic Regression/data/..., '
                             'legacy path still supported)')
    parser.add_argument('--output', default='Logistic Regression/model_artifacts/consensus_top10_full47.pkl',
                        help='Path for output model pickle')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization parameter C')
    parser.add_argument(
        '--eval-protocol',
        choices=['random', 'source_holdout', 'both'],
        default='random',
        help=('Evaluation protocol: random split only (default), '
              'source-holdout only, or both.')
    )
    parser.add_argument('--run-diagnostics', action='store_true',
                        help='Run LR diagnostics after training (optional)')
    parser.add_argument('--diagnostics-output-dir',
                        default='Logistic Regression/feature_selection_outputs',
                        help='Output directory for diagnostics artifacts')
    args = parser.parse_args()

    dataset_path = resolve_existing_path(
        args.dataset,
        fallbacks=[DEFAULT_DATASET, LEGACY_DATASET],
    )

    # Load the full 47-feature dataset
    data = np.load(str(dataset_path), allow_pickle=True)
    X = data['X']
    y = data['y']
    all_features = list(data['feature_names'])
    source_split_tag = data['source_split_tag'] if 'source_split_tag' in data else None

    # Top 10 consensus features (from full 47-feature analysis)
    top10_features = [
        'y_diff',
        'time_gap',
        'projection_error_x_max',
        'length_diff',
        'width_diff',
        'projection_error_y_max',
        'bhattacharyya_coeff',
        'projection_error_x_mean',
        'curvature_diff',
        'projection_error_x_std',
    ]

    print('=' * 70)
    print('TRAINING TOP 10 CONSENSUS FEATURES MODEL')
    print('=' * 70)
    print(f'\nDataset file: {dataset_path}')
    print(f'\nSelected features:')
    for i, f in enumerate(top10_features, 1):
        print(f'  {i:2}. {f}')

    # Get indices of selected features
    indices = [all_features.index(f) for f in top10_features]
    X_subset = X[:, indices]

    print(f'\nDataset: {X_subset.shape[0]} samples, {X_subset.shape[1]} features')

    eval_results = evaluate_protocols(
        X_subset=X_subset,
        y=y,
        C=args.C,
        eval_protocol=args.eval_protocol,
        source_split_tag=source_split_tag,
    )
    primary_key = 'random' if 'random' in eval_results else 'source_holdout'
    primary = eval_results[primary_key]

    train_roc = primary['train_roc_auc']
    test_roc = primary['test_roc_auc']
    train_ap = primary['train_ap']
    test_ap = primary['test_ap']
    cv_scores_mean = primary['cv_roc_auc_mean']
    cv_scores_std = primary['cv_roc_auc_std']

    print(f'\n' + '=' * 70)
    print('PERFORMANCE METRICS')
    print('=' * 70)
    print(f'Train ROC-AUC: {train_roc:.4f}')
    print(f'Test ROC-AUC:  {test_roc:.4f}')
    print(f'Train AP:      {train_ap:.4f}')
    print(f'Test AP:       {test_ap:.4f}')
    print(f'CV ROC-AUC:    {cv_scores_mean:.4f} +/- {cv_scores_std:.4f}')
    print(f'Eval protocol: {args.eval_protocol} (primary={primary_key})')

    if 'source_holdout' in eval_results:
        sh = eval_results['source_holdout']
        print('\nSource-holdout metrics:')
        print(f"  train_tag={sh['train_tag']}  test_tag={sh['test_tag']}")
        print(f"  Test ROC-AUC: {sh['test_roc_auc']:.4f}")
        print(f"  Test AP:      {sh['test_ap']:.4f}")

    # Check logit spread (for logit-based cost function)
    logits_test = primary['logits_test']
    print(f'\nLogit spread (test set):')
    print(f'  min={logits_test.min():.2f}, max={logits_test.max():.2f}')
    print(f'  mean={logits_test.mean():.2f}, std={logits_test.std():.2f}')
    print(f'  5th pct={np.percentile(logits_test, 5):.2f}, 95th pct={np.percentile(logits_test, 95):.2f}')

    # Retrain on full data for deployment
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X_subset)
    lr_full = LogisticRegression(max_iter=1000, solver='lbfgs', C=args.C)
    lr_full.fit(X_scaled_full, y)

    # Show coefficients
    print(f'\n' + '=' * 70)
    print('FEATURE COEFFICIENTS')
    print('=' * 70)
    print(f'{"Feature":<28} {"Coefficient":>12}')
    print('-' * 42)
    sorted_idx = np.argsort(np.abs(lr_full.coef_[0]))[::-1]
    for idx in sorted_idx:
        print(f'{top10_features[idx]:<28} {lr_full.coef_[0][idx]:>+12.4f}')

    # Save model artifacts
    artifacts = {
        'model': lr_full,
        'scaler': scaler_full,
        'feature_names': top10_features,
        'n_features': len(top10_features),
        'metrics': {
            'train_roc_auc': train_roc,
            'test_roc_auc': test_roc,
            'train_ap': train_ap,
            'test_ap': test_ap,
            'cv_roc_auc_mean': cv_scores_mean,
            'cv_roc_auc_std': cv_scores_std,
            'eval_protocol': args.eval_protocol,
            'primary_eval': primary_key,
        },
        'created': datetime.now().isoformat()
    }

    if 'random' in eval_results:
        r = eval_results['random']
        artifacts['metrics']['random_eval'] = {
            'train_roc_auc': r['train_roc_auc'],
            'test_roc_auc': r['test_roc_auc'],
            'train_ap': r['train_ap'],
            'test_ap': r['test_ap'],
            'cv_roc_auc_mean': r['cv_roc_auc_mean'],
            'cv_roc_auc_std': r['cv_roc_auc_std'],
        }
    if 'source_holdout' in eval_results:
        s = eval_results['source_holdout']
        artifacts['metrics']['source_holdout_eval'] = {
            'train_tag': s['train_tag'],
            'test_tag': s['test_tag'],
            'train_roc_auc': s['train_roc_auc'],
            'test_roc_auc': s['test_roc_auc'],
            'train_ap': s['train_ap'],
            'test_ap': s['test_ap'],
            'cv_roc_auc_mean': s['cv_roc_auc_mean'],
            'cv_roc_auc_std': s['cv_roc_auc_std'],
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f'\nModel saved to: {output_path}')

    if args.run_diagnostics:
        from lr_diagnostics import run_diagnostics

        print("\nRunning LR diagnostics...")
        result = run_diagnostics(
            dataset_path=str(dataset_path),
            model_path=str(output_path),
            output_dir=args.diagnostics_output_dir,
            threshold=0.5,
            topk_influential=250,
            use_statsmodels=True,
        )
        print(f"Diagnostics summary: {result['summary_path']}")


if __name__ == '__main__':
    main()
