#!/usr/bin/env python3
"""
Comprehensive Feature Selection for All 47 Features
====================================================
Evaluates the full 47-feature set using multiple methods:
1. Coefficient Analysis (statistical significance)
2. Permutation Importance
3. L1 Lasso Regularization
4. Recursive Feature Elimination (RFE)

Finds the optimal number of features by comparing subsets of different sizes.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
import os
import json
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = "feature_selection_outputs_full47"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load the full 47-feature dataset."""
    print("=" * 70)
    print("LOADING 47-FEATURE DATASET")
    print("=" * 70)

    data = np.load('training_dataset_advanced.npz', allow_pickle=True)
    X = data['X']
    y = data['y']
    feature_names = list(data['feature_names'])

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Positive class: {np.sum(y)} ({100*np.mean(y):.1f}%)")
    print(f"Negative class: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.1f}%)")

    return X, y, feature_names


def vif_analysis(X, feature_names, threshold=10):
    """
    Calculate Variance Inflation Factor to detect multicollinearity.
    Returns features with VIF < threshold.
    """
    print("\n" + "=" * 70)
    print("VIF ANALYSIS (Multicollinearity Detection)")
    print("=" * 70)

    # Scale data for VIF
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate VIF for each feature
    vif_data = []
    for i, name in enumerate(feature_names):
        try:
            vif = variance_inflation_factor(X_scaled, i)
            vif_data.append({'feature': name, 'VIF': vif})
        except Exception as e:
            vif_data.append({'feature': name, 'VIF': np.inf})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    vif_df.to_csv(f"{OUTPUT_DIR}/vif_analysis_full47.csv", index=False)

    # Identify high VIF features
    high_vif = vif_df[vif_df['VIF'] > threshold]
    low_vif = vif_df[vif_df['VIF'] <= threshold]

    print(f"\nFeatures with VIF > {threshold} (high multicollinearity):")
    for _, row in high_vif.iterrows():
        print(f"  - {row['feature']}: VIF={row['VIF']:.2f}")

    print(f"\nFeatures with VIF <= {threshold}: {len(low_vif)}")

    return vif_df, list(low_vif['feature'])


def coefficient_analysis(X, y, feature_names):
    """
    Fit logistic regression and analyze coefficients with statistical tests.
    """
    print("\n" + "=" * 70)
    print("COEFFICIENT ANALYSIS (Statistical Significance)")
    print("=" * 70)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit statsmodels logistic regression for p-values
    X_with_const = sm.add_constant(X_scaled)
    try:
        model = sm.Logit(y, X_with_const)
        result = model.fit(disp=0, maxiter=1000)

        coef_data = []
        for i, name in enumerate(feature_names):
            idx = i + 1  # +1 for constant
            coef_data.append({
                'feature': name,
                'coefficient': result.params[idx],
                'std_error': result.bse[idx],
                'z_score': result.tvalues[idx],
                'p_value': result.pvalues[idx],
                'abs_coefficient': abs(result.params[idx]),
                'significant': result.pvalues[idx] < 0.05
            })
    except Exception as e:
        print(f"Statsmodels failed: {e}")
        # Fallback to sklearn
        lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        lr.fit(X_scaled, y)
        coef_data = []
        for i, name in enumerate(feature_names):
            coef_data.append({
                'feature': name,
                'coefficient': lr.coef_[0][i],
                'std_error': np.nan,
                'z_score': np.nan,
                'p_value': np.nan,
                'abs_coefficient': abs(lr.coef_[0][i]),
                'significant': False
            })

    coef_df = pd.DataFrame(coef_data).sort_values('abs_coefficient', ascending=False)
    coef_df.to_csv(f"{OUTPUT_DIR}/coefficient_analysis_full47.csv", index=False)

    # Top features by coefficient
    print("\nTop 15 features by |coefficient|:")
    for i, row in coef_df.head(15).iterrows():
        sig = "*" if row['significant'] else ""
        print(f"  {row['feature']:30} coef={row['coefficient']:+.4f} p={row['p_value']:.4f}{sig}")

    sig_count = coef_df['significant'].sum()
    print(f"\nSignificant features (p < 0.05): {sig_count}/{len(feature_names)}")

    return coef_df


def permutation_importance_analysis(X, y, feature_names):
    """
    Calculate permutation importance using cross-validation.
    """
    print("\n" + "=" * 70)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Scale and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit model
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    lr.fit(X_train, y_train)

    # Calculate permutation importance
    perm_result = permutation_importance(
        lr, X_test, y_test, n_repeats=30, random_state=42, scoring='roc_auc'
    )

    perm_data = []
    for i, name in enumerate(feature_names):
        perm_data.append({
            'feature': name,
            'importance_mean': perm_result.importances_mean[i],
            'importance_std': perm_result.importances_std[i]
        })

    perm_df = pd.DataFrame(perm_data).sort_values('importance_mean', ascending=False)
    perm_df.to_csv(f"{OUTPUT_DIR}/permutation_importance_full47.csv", index=False)

    print("\nTop 15 features by permutation importance:")
    for i, row in perm_df.head(15).iterrows():
        print(f"  {row['feature']:30} importance={row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

    return perm_df


def l1_lasso_analysis(X, y, feature_names):
    """
    Use L1 regularization to identify important features.
    """
    print("\n" + "=" * 70)
    print("L1 LASSO REGULARIZATION ANALYSIS")
    print("=" * 70)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test different regularization strengths
    C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    reg_path = []
    for C in C_values:
        lr = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=2000)
        lr.fit(X_scaled, y)
        n_nonzero = np.sum(lr.coef_[0] != 0)
        reg_path.append({'C': C, 'n_features': n_nonzero})
        print(f"  C={C:5.3f}: {n_nonzero} non-zero features")

    pd.DataFrame(reg_path).to_csv(f"{OUTPUT_DIR}/l1_regularization_path_full47.csv", index=False)

    # Use moderate C for feature selection
    lr = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=2000)
    lr.fit(X_scaled, y)

    l1_data = []
    for i, name in enumerate(feature_names):
        l1_data.append({
            'feature': name,
            'coefficient': lr.coef_[0][i],
            'abs_coefficient': abs(lr.coef_[0][i]),
            'selected': lr.coef_[0][i] != 0
        })

    l1_df = pd.DataFrame(l1_data).sort_values('abs_coefficient', ascending=False)
    l1_df.to_csv(f"{OUTPUT_DIR}/l1_lasso_features_full47.csv", index=False)

    selected = l1_df[l1_df['selected']]['feature'].tolist()
    print(f"\nL1 selected features (C=0.5): {len(selected)}")

    return l1_df, selected


def rfe_analysis(X, y, feature_names, n_features_to_select=15):
    """
    Recursive Feature Elimination analysis.
    """
    print("\n" + "=" * 70)
    print("RECURSIVE FEATURE ELIMINATION (RFE)")
    print("=" * 70)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # RFE with logistic regression
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    rfe = RFE(lr, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X_scaled, y)

    rfe_data = []
    for i, name in enumerate(feature_names):
        rfe_data.append({
            'feature': name,
            'ranking': rfe.ranking_[i],
            'selected': rfe.support_[i]
        })

    rfe_df = pd.DataFrame(rfe_data).sort_values('ranking')
    rfe_df.to_csv(f"{OUTPUT_DIR}/rfe_ranking_full47.csv", index=False)

    selected = rfe_df[rfe_df['selected']]['feature'].tolist()
    print(f"\nRFE selected features (top {n_features_to_select}):")
    for name in selected:
        print(f"  - {name}")

    return rfe_df, selected


def evaluate_feature_subset(X, y, feature_names, selected_features, subset_name):
    """
    Evaluate a feature subset using cross-validation.
    """
    # Get indices of selected features
    indices = [feature_names.index(f) for f in selected_features if f in feature_names]
    X_subset = X[:, indices]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    # Split for test evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cross-validation
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')

    # Test set evaluation
    lr.fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_ap = average_precision_score(y_test, y_proba)

    return {
        'subset': subset_name,
        'n_features': len(indices),
        'cv_roc_auc_mean': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std(),
        'test_roc_auc': test_roc_auc,
        'test_avg_precision': test_ap,
        'features': selected_features
    }


def find_optimal_features(X, y, feature_names, perm_df, coef_df, rfe_df, l1_df):
    """
    Systematically compare feature subsets of different sizes.
    """
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL NUMBER OF FEATURES")
    print("=" * 70)

    # Get feature rankings from different methods
    perm_ranked = perm_df.sort_values('importance_mean', ascending=False)['feature'].tolist()
    coef_ranked = coef_df.sort_values('abs_coefficient', ascending=False)['feature'].tolist()
    rfe_ranked = rfe_df.sort_values('ranking')['feature'].tolist()

    # Test different subset sizes
    subset_sizes = list(range(1, 48))

    results = []

    for n in subset_sizes:
        # Permutation-based subset
        perm_subset = perm_ranked[:n]
        result = evaluate_feature_subset(X, y, feature_names, perm_subset, f"Top {n} (Permutation)")
        results.append(result)

        # Coefficient-based subset
        coef_subset = coef_ranked[:n]
        result = evaluate_feature_subset(X, y, feature_names, coef_subset, f"Top {n} (Coefficient)")
        results.append(result)

        # RFE-based subset (using ranking)
        
        rfe_subset = rfe_ranked[:n]
        result = evaluate_feature_subset(X, y, feature_names, rfe_subset, f"Top {n} (RFE)")
        results.append(result)

    # Also test L1 selected features
    l1_selected = l1_df[l1_df['selected']]['feature'].tolist()
    if len(l1_selected) > 0:
        result = evaluate_feature_subset(X, y, feature_names, l1_selected, f"L1 Selected ({len(l1_selected)})")
        results.append(result)

    # All features
    result = evaluate_feature_subset(X, y, feature_names, feature_names, "All 47 Features")
    results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_roc_auc', ascending=False)

    # Save summary (without feature lists for readability)
    summary_df = results_df[['subset', 'n_features', 'cv_roc_auc_mean', 'cv_roc_auc_std', 'test_roc_auc', 'test_avg_precision']]
    summary_df.to_csv(f"{OUTPUT_DIR}/feature_subset_comparison_full47.csv", index=False)

    print("\nFeature Subset Comparison (sorted by test ROC-AUC):")
    print("-" * 90)
    print(f"{'Subset':<30} {'N':<5} {'CV ROC-AUC':>15} {'Test ROC-AUC':>15} {'Test AP':>15}")
    print("-" * 90)
    for _, row in summary_df.head(20).iterrows():
        print(f"{row['subset']:<30} {row['n_features']:<5} {row['cv_roc_auc_mean']:.4f}±{row['cv_roc_auc_std']:.4f}  {row['test_roc_auc']:.4f}          {row['test_avg_precision']:.4f}")

    return results_df


def consensus_ranking(perm_df, coef_df, rfe_df, l1_df, feature_names):
    """
    Create consensus ranking from all methods.
    """
    print("\n" + "=" * 70)
    print("CONSENSUS FEATURE RANKING")
    print("=" * 70)

    consensus = {}

    # Permutation ranking
    perm_ranked = perm_df.sort_values('importance_mean', ascending=False)['feature'].tolist()
    for i, f in enumerate(perm_ranked):
        consensus[f] = consensus.get(f, 0) + (len(feature_names) - i)

    # Coefficient ranking
    coef_ranked = coef_df.sort_values('abs_coefficient', ascending=False)['feature'].tolist()
    for i, f in enumerate(coef_ranked):
        consensus[f] = consensus.get(f, 0) + (len(feature_names) - i)

    # RFE ranking
    rfe_ranked = rfe_df.sort_values('ranking')['feature'].tolist()
    for i, f in enumerate(rfe_ranked):
        consensus[f] = consensus.get(f, 0) + (len(feature_names) - i)

    # L1 bonus (selected features get extra points)
    l1_selected = set(l1_df[l1_df['selected']]['feature'].tolist())
    for f in l1_selected:
        consensus[f] = consensus.get(f, 0) + 20

    # Sort by consensus score
    consensus_ranked = sorted(consensus.items(), key=lambda x: x[1], reverse=True)

    consensus_df = pd.DataFrame(consensus_ranked, columns=['feature', 'consensus_score'])
    consensus_df['rank'] = range(1, len(consensus_df) + 1)

    # Add individual method ranks
    perm_rank_map = {f: i+1 for i, f in enumerate(perm_ranked)}
    coef_rank_map = {f: i+1 for i, f in enumerate(coef_ranked)}
    rfe_rank_map = {f: i+1 for i, f in enumerate(rfe_ranked)}

    consensus_df['perm_rank'] = consensus_df['feature'].map(perm_rank_map)
    consensus_df['coef_rank'] = consensus_df['feature'].map(coef_rank_map)
    consensus_df['rfe_rank'] = consensus_df['feature'].map(rfe_rank_map)
    consensus_df['l1_selected'] = consensus_df['feature'].isin(l1_selected)

    consensus_df.to_csv(f"{OUTPUT_DIR}/consensus_ranking_full47.csv", index=False)

    print("\nTop 20 Features by Consensus Ranking:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Feature':<30} {'Score':<8} {'Perm':<6} {'Coef':<6} {'RFE':<6} {'L1':<5}")
    print("-" * 80)
    for _, row in consensus_df.head(20).iterrows():
        l1_mark = "✓" if row['l1_selected'] else ""
        print(f"{row['rank']:<5} {row['feature']:<30} {row['consensus_score']:<8.0f} {row['perm_rank']:<6} {row['coef_rank']:<6} {row['rfe_rank']:<6} {l1_mark:<5}")

    return consensus_df


def train_optimal_model(X, y, feature_names, optimal_features, output_name):
    """
    Train final model with optimal features and save artifacts.
    """
    print("\n" + "=" * 70)
    print(f"TRAINING OPTIMAL MODEL: {output_name}")
    print("=" * 70)

    # Get indices
    indices = [feature_names.index(f) for f in optimal_features if f in feature_names]
    X_subset = X[:, indices]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train final model
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    lr.fit(X_train, y_train)

    # Evaluate
    y_proba_train = lr.predict_proba(X_train)[:, 1]
    y_proba_test = lr.predict_proba(X_test)[:, 1]

    train_roc = roc_auc_score(y_train, y_proba_train)
    test_roc = roc_auc_score(y_test, y_proba_test)
    train_ap = average_precision_score(y_train, y_proba_train)
    test_ap = average_precision_score(y_test, y_proba_test)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')

    print(f"\nModel trained with {len(optimal_features)} features:")
    print(f"  Features: {optimal_features}")
    print(f"\nPerformance:")
    print(f"  Train ROC-AUC: {train_roc:.4f}")
    print(f"  Test ROC-AUC:  {test_roc:.4f}")
    print(f"  Train AP:      {train_ap:.4f}")
    print(f"  Test AP:       {test_ap:.4f}")
    print(f"  CV ROC-AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Retrain on full data for deployment
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X_subset)
    lr_full = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    lr_full.fit(X_scaled_full, y)

    # Save artifacts
    artifacts = {
        'model': lr_full,
        'scaler': scaler_full,
        'feature_names': optimal_features,
        'n_features': len(optimal_features),
        'metrics': {
            'train_roc_auc': train_roc,
            'test_roc_auc': test_roc,
            'train_ap': train_ap,
            'test_ap': test_ap,
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std()
        },
        'created': datetime.now().isoformat()
    }

    artifact_path = f"model_artifacts/{output_name}.pkl"
    os.makedirs("model_artifacts", exist_ok=True)
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"\nModel saved to: {artifact_path}")

    return artifacts


def generate_report(results_df, consensus_df, vif_df, optimal_features):
    """
    Generate comprehensive text report.
    """
    report_path = f"{OUTPUT_DIR}/feature_selection_report_full47.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE FEATURE SELECTION REPORT (47 FEATURES)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("=" * 80 + "\n")
        f.write("BEST PERFORMING SUBSETS\n")
        f.write("=" * 80 + "\n\n")

        best_results = results_df.head(10)
        for _, row in best_results.iterrows():
            f.write(f"  {row['subset']:<35} ROC-AUC: {row['test_roc_auc']:.4f}  AP: {row['test_avg_precision']:.4f}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("OPTIMAL FEATURE SET\n")
        f.write("=" * 80 + "\n\n")

        best_row = results_df.iloc[0]
        f.write(f"Best subset: {best_row['subset']}\n")
        f.write(f"Number of features: {best_row['n_features']}\n")
        f.write(f"Test ROC-AUC: {best_row['test_roc_auc']:.4f}\n")
        f.write(f"Test AP: {best_row['test_avg_precision']:.4f}\n")
        f.write(f"CV ROC-AUC: {best_row['cv_roc_auc_mean']:.4f} ± {best_row['cv_roc_auc_std']:.4f}\n\n")

        f.write("Features:\n")
        for feat in optimal_features:
            f.write(f"  - {feat}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("CONSENSUS RANKING (TOP 20)\n")
        f.write("=" * 80 + "\n\n")

        for _, row in consensus_df.head(20).iterrows():
            f.write(f"  {row['rank']:2}. {row['feature']:<30} score={row['consensus_score']:.0f}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("HIGH MULTICOLLINEARITY FEATURES (VIF > 10)\n")
        f.write("=" * 80 + "\n\n")

        high_vif = vif_df[vif_df['VIF'] > 10]
        for _, row in high_vif.iterrows():
            f.write(f"  - {row['feature']}: VIF={row['VIF']:.2f}\n")

    print(f"\nReport saved to: {report_path}")


def main():
    """Main execution flow."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE FEATURE SELECTION FOR ALL 47 FEATURES")
    print("=" * 80)

    # Load data
    X, y, feature_names = load_data()

    # 1. VIF Analysis
    vif_df, low_vif_features = vif_analysis(X, feature_names)

    # 2. Coefficient Analysis
    coef_df = coefficient_analysis(X, y, feature_names)

    # 3. Permutation Importance
    perm_df = permutation_importance_analysis(X, y, feature_names)

    # 4. L1 Lasso
    l1_df, l1_selected = l1_lasso_analysis(X, y, feature_names)

    # 5. RFE
    rfe_df, rfe_selected = rfe_analysis(X, y, feature_names, n_features_to_select=15)

    # 6. Compare subsets of different sizes
    results_df = find_optimal_features(X, y, feature_names, perm_df, coef_df, rfe_df, l1_df)

    # 7. Consensus ranking
    consensus_df = consensus_ranking(perm_df, coef_df, rfe_df, l1_df, feature_names)

    # 8. Identify optimal features
    best_row = results_df.iloc[0]
    optimal_features = best_row['features']
    optimal_n = best_row['n_features']

    print("\n" + "=" * 70)
    print("OPTIMAL FEATURE SET IDENTIFIED")
    print("=" * 70)
    print(f"\nBest subset: {best_row['subset']}")
    print(f"Number of features: {optimal_n}")
    print(f"Test ROC-AUC: {best_row['test_roc_auc']:.4f}")
    print(f"Features: {optimal_features}")

    # 9. Train optimal model
    artifacts = train_optimal_model(X, y, feature_names, optimal_features, f"optimal_{optimal_n}features_full47")

    # Also train models with consensus top-9/10/11/15
    consensus_top9 = consensus_df.head(9)['feature'].tolist()
    consensus_top10 = consensus_df.head(10)['feature'].tolist()
    consensus_top11 = consensus_df.head(11)['feature'].tolist()
    consensus_top15 = consensus_df.head(15)['feature'].tolist()

    train_optimal_model(X, y, feature_names, consensus_top9, "consensus_9features_full47")
    train_optimal_model(X, y, feature_names, consensus_top10, "consensus_10features_full47")
    train_optimal_model(X, y, feature_names, consensus_top11, "consensus_11features_full47")
    train_optimal_model(X, y, feature_names, consensus_top15, "consensus_15features_full47")


    # 10. Generate report
    generate_report(results_df, consensus_df, vif_df, optimal_features)

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print(f"Model artifacts saved to: model_artifacts/")

    return results_df, consensus_df, optimal_features


if __name__ == "__main__":
    main()
