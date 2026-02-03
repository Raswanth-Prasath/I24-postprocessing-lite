"""
Comprehensive Feature Evaluation for Logistic Regression
Implements multiple methods from research literature:
1. Coefficient analysis with statistical significance
2. Permutation importance
3. L1 regularization (Lasso) for feature selection
4. Recursive Feature Elimination
5. Cross-validation performance comparison

Based on:
- Islam et al. (2024): L1/L2 regularization
- Zakharov & Dupont (2011): Ensemble feature selection
- SHAP-based methods for interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Optional seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Configuration
WORKING_DIR = Path("/home/raswanth/I24/I24-postprocessing-lite/Logistic Regression")
OUTPUT_DIR = WORKING_DIR / "feature_selection_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

class FeatureEvaluator:
    """Comprehensive feature evaluation for logistic regression"""

    def __init__(self, X, y, feature_names, test_size=0.2, random_state=42):
        """
        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            test_size: Test set proportion
            random_state: Random seed
        """
        self.X = X
        self.y = y
        self.feature_names = np.array(feature_names)
        self.n_features = X.shape[1]
        self.random_state = random_state

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Results storage
        self.results = {}

    def evaluate_coefficients_with_significance(self):
        """
        Method 1: Coefficient analysis with statistical significance
        Based on Wald test and confidence intervals
        """
        print("\n" + "="*80)
        print("METHOD 1: COEFFICIENT ANALYSIS WITH STATISTICAL SIGNIFICANCE")
        print("="*80)

        # Train standard logistic regression
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )
        model.fit(self.X_train_scaled, self.y_train)

        # Extract coefficients
        coefficients = model.coef_[0]

        # Compute standard errors using Hessian approximation
        # For logistic regression: SE = sqrt(diag(inv(X'WX)))
        # where W = diag(p(1-p)) for predicted probabilities p
        y_pred_proba = model.predict_proba(self.X_train_scaled)[:, 1]
        W = np.diag(y_pred_proba * (1 - y_pred_proba))

        try:
            # Compute Hessian: X'WX
            XtWX = self.X_train_scaled.T @ W @ self.X_train_scaled
            # Add small regularization for numerical stability
            XtWX_reg = XtWX + 1e-8 * np.eye(self.n_features)
            # Compute variance-covariance matrix
            var_covar = np.linalg.inv(XtWX_reg)
            std_errors = np.sqrt(np.diag(var_covar))
        except np.linalg.LinAlgError:
            print("Warning: Could not compute standard errors. Using bootstrap instead.")
            std_errors = self._bootstrap_std_errors(model)

        # Wald test: z = coef / SE
        z_scores = coefficients / (std_errors + 1e-10)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

        # 95% confidence intervals
        ci_lower = coefficients - 1.96 * std_errors
        ci_upper = coefficients + 1.96 * std_errors

        # Odds ratios
        odds_ratios = np.exp(coefficients)

        # Create DataFrame
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'std_error': std_errors,
            'z_score': z_scores,
            'p_value': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'odds_ratio': odds_ratios,
            'abs_coefficient': np.abs(coefficients),
            'significant': p_values < 0.05
        }).sort_values('abs_coefficient', ascending=False)

        self.results['coefficients'] = coef_df

        # Print top features
        print("\nTop 20 Features by Absolute Coefficient:")
        print(coef_df[['feature', 'coefficient', 'p_value', 'odds_ratio', 'significant']].head(20).to_string(index=False))

        # Save to CSV
        coef_df.to_csv(OUTPUT_DIR / 'coefficient_analysis.csv', index=False)
        print(f"\nSaved to: {OUTPUT_DIR / 'coefficient_analysis.csv'}")

        return coef_df

    def _bootstrap_std_errors(self, model, n_bootstrap=100):
        """Compute standard errors using bootstrap"""
        boot_coefs = []
        n_samples = len(self.X_train_scaled)

        for _ in range(n_bootstrap):
            # Resample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = self.X_train_scaled[indices]
            y_boot = self.y_train[indices]

            # Fit model
            model_boot = LogisticRegression(
                penalty='l2', C=1.0, class_weight='balanced',
                random_state=self.random_state, max_iter=1000
            )
            model_boot.fit(X_boot, y_boot)
            boot_coefs.append(model_boot.coef_[0])

        return np.std(boot_coefs, axis=0)

    def evaluate_permutation_importance(self, n_repeats=10):
        """
        Method 2: Permutation importance
        Measures performance drop when feature values are shuffled
        """
        print("\n" + "="*80)
        print("METHOD 2: PERMUTATION IMPORTANCE")
        print("="*80)

        # Train model
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )
        model.fit(self.X_train_scaled, self.y_train)

        # Compute permutation importance
        print(f"\nComputing permutation importance ({n_repeats} repeats)...")
        perm_importance = permutation_importance(
            model,
            self.X_test_scaled,
            self.y_test,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring='roc_auc',
            n_jobs=-1
        )

        # Create DataFrame
        perm_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        self.results['permutation'] = perm_df

        # Print top features
        print("\nTop 20 Features by Permutation Importance:")
        print(perm_df.head(20).to_string(index=False))

        # Save to CSV
        perm_df.to_csv(OUTPUT_DIR / 'permutation_importance.csv', index=False)
        print(f"\nSaved to: {OUTPUT_DIR / 'permutation_importance.csv'}")

        return perm_df

    def evaluate_l1_regularization(self, cv=5):
        """
        Method 3: L1 regularization (Lasso) for feature selection
        Based on Islam et al. (2024)
        """
        print("\n" + "="*80)
        print("METHOD 3: L1 REGULARIZATION (LASSO)")
        print("="*80)

        # Use cross-validation to find optimal C
        print("\nFinding optimal regularization strength (C)...")
        C_range = np.logspace(-3, 2, 20)

        model_cv = LogisticRegressionCV(
            Cs=C_range,
            penalty='l1',
            solver='liblinear',
            cv=cv,
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
            scoring='roc_auc',
            n_jobs=-1
        )
        model_cv.fit(self.X_train_scaled, self.y_train)

        best_C = model_cv.C_[0]
        print(f"Best C: {best_C:.6f}")

        # Get coefficients for different C values
        lasso_results = []
        for C in C_range:
            model = LogisticRegression(
                penalty='l1',
                C=C,
                solver='liblinear',
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            )
            model.fit(self.X_train_scaled, self.y_train)

            # Count non-zero coefficients
            non_zero = np.sum(model.coef_[0] != 0)

            # Compute ROC-AUC
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            lasso_results.append({
                'C': C,
                'n_features_selected': non_zero,
                'roc_auc': roc_auc
            })

        lasso_path_df = pd.DataFrame(lasso_results)
        self.results['lasso_path'] = lasso_path_df

        # Get feature importance from best model
        best_model = model_cv
        lasso_df = pd.DataFrame({
            'feature': self.feature_names,
            'l1_coefficient': best_model.coef_[0],
            'abs_l1_coefficient': np.abs(best_model.coef_[0]),
            'selected': best_model.coef_[0] != 0
        }).sort_values('abs_l1_coefficient', ascending=False)

        self.results['lasso'] = lasso_df

        # Print results
        n_selected = lasso_df['selected'].sum()
        print(f"\nFeatures selected by L1: {n_selected}/{self.n_features} ({n_selected/self.n_features*100:.1f}%)")
        print("\nTop 20 Features by L1 Coefficient:")
        print(lasso_df[lasso_df['selected']].head(20).to_string(index=False))

        # Save to CSV
        lasso_df.to_csv(OUTPUT_DIR / 'l1_lasso_features.csv', index=False)
        lasso_path_df.to_csv(OUTPUT_DIR / 'l1_regularization_path.csv', index=False)
        print(f"\nSaved to: {OUTPUT_DIR / 'l1_lasso_features.csv'}")

        return lasso_df, lasso_path_df

    def evaluate_rfe(self, n_features_to_select=20, step=1):
        """
        Method 4: Recursive Feature Elimination
        """
        print("\n" + "="*80)
        print("METHOD 4: RECURSIVE FEATURE ELIMINATION (RFE)")
        print("="*80)

        # Base estimator
        estimator = LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )

        print(f"\nSelecting top {n_features_to_select} features...")

        # Perform RFE
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=step
        )
        rfe.fit(self.X_train_scaled, self.y_train)

        # Create DataFrame
        rfe_df = pd.DataFrame({
            'feature': self.feature_names,
            'ranking': rfe.ranking_,
            'selected': rfe.support_
        }).sort_values('ranking')

        self.results['rfe'] = rfe_df

        # Evaluate performance
        y_pred_proba = rfe.predict_proba(self.X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        print(f"\nRFE Performance (ROC-AUC): {roc_auc:.4f}")
        print(f"\nTop {n_features_to_select} Features Selected by RFE:")
        print(rfe_df[rfe_df['selected']].to_string(index=False))

        # Save to CSV
        rfe_df.to_csv(OUTPUT_DIR / 'rfe_feature_ranking.csv', index=False)
        print(f"\nSaved to: {OUTPUT_DIR / 'rfe_feature_ranking.csv'}")

        return rfe_df

    def compare_feature_subsets(self):
        """
        Method 5: Compare performance with different feature subsets
        """
        print("\n" + "="*80)
        print("METHOD 5: FEATURE SUBSET PERFORMANCE COMPARISON")
        print("="*80)

        # Define feature subsets
        subsets = {
            'All Features': list(range(self.n_features)),
            'Top 30 (Coefficient)': None,
            'Top 20 (Coefficient)': None,
            'Top 10 (Coefficient)': None,
            'Top 30 (Permutation)': None,
            'Top 20 (Permutation)': None,
            'Top 10 (Permutation)': None,
            'L1 Selected': None,
            'RFE Selected': None
        }

        # Get indices for top features
        if 'coefficients' in self.results:
            coef_df = self.results['coefficients']
            subsets['Top 30 (Coefficient)'] = [np.where(self.feature_names == f)[0][0] for f in coef_df['feature'].head(30)]
            subsets['Top 20 (Coefficient)'] = [np.where(self.feature_names == f)[0][0] for f in coef_df['feature'].head(20)]
            subsets['Top 10 (Coefficient)'] = [np.where(self.feature_names == f)[0][0] for f in coef_df['feature'].head(10)]

        if 'permutation' in self.results:
            perm_df = self.results['permutation']
            subsets['Top 30 (Permutation)'] = [np.where(self.feature_names == f)[0][0] for f in perm_df['feature'].head(30)]
            subsets['Top 20 (Permutation)'] = [np.where(self.feature_names == f)[0][0] for f in perm_df['feature'].head(20)]
            subsets['Top 10 (Permutation)'] = [np.where(self.feature_names == f)[0][0] for f in perm_df['feature'].head(10)]

        if 'lasso' in self.results:
            lasso_df = self.results['lasso']
            subsets['L1 Selected'] = [np.where(self.feature_names == f)[0][0] for f in lasso_df[lasso_df['selected']]['feature']]

        if 'rfe' in self.results:
            rfe_df = self.results['rfe']
            subsets['RFE Selected'] = [np.where(self.feature_names == f)[0][0] for f in rfe_df[rfe_df['selected']]['feature']]

        # Remove None subsets
        subsets = {k: v for k, v in subsets.items() if v is not None}

        # Evaluate each subset
        print("\nEvaluating feature subsets...")
        comparison_results = []

        for subset_name, feature_indices in subsets.items():
            # Select features
            X_train_subset = self.X_train_scaled[:, feature_indices]
            X_test_subset = self.X_test_scaled[:, feature_indices]

            # Train model
            model = LogisticRegression(
                penalty='l2',
                C=1.0,
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            )

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_subset, self.y_train,
                cv=5, scoring='roc_auc', n_jobs=-1
            )

            # Test performance
            model.fit(X_train_subset, self.y_train)
            y_pred_proba = model.predict_proba(X_test_subset)[:, 1]
            test_roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            test_ap = average_precision_score(self.y_test, y_pred_proba)

            comparison_results.append({
                'subset': subset_name,
                'n_features': len(feature_indices),
                'cv_roc_auc_mean': cv_scores.mean(),
                'cv_roc_auc_std': cv_scores.std(),
                'test_roc_auc': test_roc_auc,
                'test_avg_precision': test_ap
            })

        comparison_df = pd.DataFrame(comparison_results).sort_values('test_roc_auc', ascending=False)
        self.results['comparison'] = comparison_df

        # Print results
        print("\nFeature Subset Performance:")
        print(comparison_df.to_string(index=False))

        # Save to CSV
        comparison_df.to_csv(OUTPUT_DIR / 'feature_subset_comparison.csv', index=False)
        print(f"\nSaved to: {OUTPUT_DIR / 'feature_subset_comparison.csv'}")

        return comparison_df

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        # Set style
        if HAS_SEABORN:
            sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # Create subplots
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Top 20 Coefficients with CI
        if 'coefficients' in self.results:
            ax1 = fig.add_subplot(gs[0, :2])
            coef_df = self.results['coefficients'].head(20).copy()

            # Sort by coefficient value for better visualization
            coef_df = coef_df.sort_values('coefficient')

            y_pos = np.arange(len(coef_df))
            colors = ['red' if c < 0 else 'blue' for c in coef_df['coefficient']]

            ax1.barh(y_pos, coef_df['coefficient'], color=colors, alpha=0.6)
            ax1.errorbar(coef_df['coefficient'], y_pos,
                        xerr=1.96 * coef_df['std_error'],
                        fmt='none', ecolor='black', capsize=3)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(coef_df['feature'], fontsize=9)
            ax1.set_xlabel('Coefficient Value', fontsize=11)
            ax1.set_title('Top 20 Features by Coefficient (with 95% CI)', fontsize=13, fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax1.grid(axis='x', alpha=0.3)

        # 2. P-value significance
        if 'coefficients' in self.results:
            ax2 = fig.add_subplot(gs[0, 2])
            coef_df = self.results['coefficients']

            # Create significance categories
            sig_counts = pd.Series({
                'p < 0.001': (coef_df['p_value'] < 0.001).sum(),
                '0.001 ≤ p < 0.01': ((coef_df['p_value'] >= 0.001) & (coef_df['p_value'] < 0.01)).sum(),
                '0.01 ≤ p < 0.05': ((coef_df['p_value'] >= 0.01) & (coef_df['p_value'] < 0.05)).sum(),
                'p ≥ 0.05': (coef_df['p_value'] >= 0.05).sum()
            })

            colors_sig = ['darkgreen', 'green', 'orange', 'red']
            ax2.pie(sig_counts, labels=sig_counts.index, autopct='%1.1f%%',
                   colors=colors_sig, startangle=90)
            ax2.set_title('Statistical Significance\nDistribution', fontsize=12, fontweight='bold')

        # 3. Permutation Importance
        if 'permutation' in self.results:
            ax3 = fig.add_subplot(gs[1, :2])
            perm_df = self.results['permutation'].head(20).sort_values('importance_mean')

            y_pos = np.arange(len(perm_df))
            ax3.barh(y_pos, perm_df['importance_mean'], color='green', alpha=0.6)
            ax3.errorbar(perm_df['importance_mean'], y_pos,
                        xerr=perm_df['importance_std'],
                        fmt='none', ecolor='black', capsize=3)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(perm_df['feature'], fontsize=9)
            ax3.set_xlabel('Permutation Importance (ROC-AUC Drop)', fontsize=11)
            ax3.set_title('Top 20 Features by Permutation Importance', fontsize=13, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)

        # 4. L1 Regularization Path
        if 'lasso_path' in self.results:
            ax4 = fig.add_subplot(gs[1, 2])
            lasso_path = self.results['lasso_path']

            ax4_twin = ax4.twinx()
            ax4.plot(lasso_path['C'], lasso_path['n_features_selected'],
                    'b-o', label='Features Selected', markersize=4)
            ax4_twin.plot(lasso_path['C'], lasso_path['roc_auc'],
                         'r-s', label='ROC-AUC', markersize=4)

            ax4.set_xscale('log')
            ax4.set_xlabel('Regularization Strength (C)', fontsize=11)
            ax4.set_ylabel('Number of Features', fontsize=11, color='b')
            ax4_twin.set_ylabel('ROC-AUC', fontsize=11, color='r')
            ax4.tick_params(axis='y', labelcolor='b')
            ax4_twin.tick_params(axis='y', labelcolor='r')
            ax4.set_title('L1 Regularization Path', fontsize=12, fontweight='bold')
            ax4.grid(alpha=0.3)

        # 5. Feature Subset Comparison
        if 'comparison' in self.results:
            ax5 = fig.add_subplot(gs[2, :])
            comp_df = self.results['comparison'].sort_values('test_roc_auc')

            x_pos = np.arange(len(comp_df))

            # Plot bars
            bars1 = ax5.bar(x_pos - 0.2, comp_df['cv_roc_auc_mean'], 0.4,
                           yerr=comp_df['cv_roc_auc_std'],
                           label='CV ROC-AUC', alpha=0.7, capsize=3)
            bars2 = ax5.bar(x_pos + 0.2, comp_df['test_roc_auc'], 0.4,
                           label='Test ROC-AUC', alpha=0.7)

            # Add feature count labels
            for i, (idx, row) in enumerate(comp_df.iterrows()):
                ax5.text(i, -0.05, f"n={row['n_features']}",
                        ha='center', va='top', fontsize=8, rotation=0)

            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(comp_df['subset'], rotation=45, ha='right', fontsize=9)
            ax5.set_ylabel('ROC-AUC Score', fontsize=11)
            ax5.set_title('Performance Comparison Across Feature Subsets', fontsize=13, fontweight='bold')
            ax5.legend(fontsize=10)
            ax5.grid(axis='y', alpha=0.3)
            ax5.set_ylim(bottom=-0.1)

        plt.savefig(OUTPUT_DIR / 'feature_evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved visualization to: {OUTPUT_DIR / 'feature_evaluation_comprehensive.png'}")
        plt.close()

        # Additional plot: Coefficient vs Permutation Importance correlation
        if 'coefficients' in self.results and 'permutation' in self.results:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Merge dataframes
            merged = self.results['coefficients'].merge(
                self.results['permutation'],
                on='feature',
                how='inner'
            )

            # Create scatter plot
            scatter = ax.scatter(
                merged['abs_coefficient'],
                merged['importance_mean'],
                s=100,
                c=merged['significant'].map({True: 'green', False: 'red'}),
                alpha=0.6,
                edgecolors='black'
            )

            # Add labels for top features
            top_features = merged.nlargest(15, 'abs_coefficient')
            for _, row in top_features.iterrows():
                ax.annotate(
                    row['feature'],
                    (row['abs_coefficient'], row['importance_mean']),
                    fontsize=8,
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords='offset points'
                )

            ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
            ax.set_ylabel('Permutation Importance', fontsize=12)
            ax.set_title('Coefficient vs Permutation Importance\n(Green = Significant, Red = Not Significant)',
                        fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)

            # Add correlation
            corr = merged['abs_coefficient'].corr(merged['importance_mean'])
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'coefficient_vs_permutation.png', dpi=300, bbox_inches='tight')
            print(f"Saved correlation plot to: {OUTPUT_DIR / 'coefficient_vs_permutation.png'}")
            plt.close()

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FEATURE EVALUATION SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nDataset: {self.n_features} features, {len(self.y)} samples")
        report_lines.append(f"Positive class: {np.sum(self.y == 1)} ({np.sum(self.y == 1)/len(self.y)*100:.1f}%)")
        report_lines.append(f"Negative class: {np.sum(self.y == 0)} ({np.sum(self.y == 0)/len(self.y)*100:.1f}%)")

        # Top features consensus
        report_lines.append("\n" + "=" * 80)
        report_lines.append("TOP FEATURES CONSENSUS (Appearing in Multiple Methods)")
        report_lines.append("=" * 80)

        # Collect top 10 from each method
        top_features_by_method = {}

        if 'coefficients' in self.results:
            top_features_by_method['Coefficient'] = set(self.results['coefficients'].head(10)['feature'])

        if 'permutation' in self.results:
            top_features_by_method['Permutation'] = set(self.results['permutation'].head(10)['feature'])

        if 'lasso' in self.results:
            lasso_top = self.results['lasso'][self.results['lasso']['selected']].head(10)
            top_features_by_method['L1 Lasso'] = set(lasso_top['feature'])

        if 'rfe' in self.results:
            rfe_top = self.results['rfe'][self.results['rfe']['selected']].head(10)
            top_features_by_method['RFE'] = set(rfe_top['feature'])

        # Count occurrences
        all_features = set()
        for features in top_features_by_method.values():
            all_features.update(features)

        feature_counts = {}
        for feature in all_features:
            count = sum(1 for features in top_features_by_method.values() if feature in features)
            feature_counts[feature] = count

        # Sort by consensus
        consensus_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

        report_lines.append(f"\nFeatures appearing in multiple top-10 lists:")
        for feature, count in consensus_features[:20]:
            methods = [method for method, features in top_features_by_method.items() if feature in features]
            report_lines.append(f"  {feature:40s} - {count}/{len(top_features_by_method)} methods: {', '.join(methods)}")

        # Best performing subset
        if 'comparison' in self.results:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("BEST PERFORMING FEATURE SUBSET")
            report_lines.append("=" * 80)

            best_subset = self.results['comparison'].iloc[0]
            report_lines.append(f"\nSubset: {best_subset['subset']}")
            report_lines.append(f"Number of features: {best_subset['n_features']}")
            report_lines.append(f"Test ROC-AUC: {best_subset['test_roc_auc']:.4f}")
            report_lines.append(f"Test Average Precision: {best_subset['test_avg_precision']:.4f}")
            report_lines.append(f"CV ROC-AUC: {best_subset['cv_roc_auc_mean']:.4f} ± {best_subset['cv_roc_auc_std']:.4f}")

        # Statistical significance summary
        if 'coefficients' in self.results:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("STATISTICAL SIGNIFICANCE SUMMARY")
            report_lines.append("=" * 80)

            coef_df = self.results['coefficients']
            n_sig = coef_df['significant'].sum()
            report_lines.append(f"\nSignificant features (p < 0.05): {n_sig}/{self.n_features} ({n_sig/self.n_features*100:.1f}%)")

            report_lines.append("\nMost significant features (lowest p-values):")
            for _, row in coef_df.nsmallest(10, 'p_value').iterrows():
                report_lines.append(f"  {row['feature']:40s} - p={row['p_value']:.6f}, coef={row['coefficient']:+.4f}")

        # Save report
        report_text = "\n".join(report_lines)
        with open(OUTPUT_DIR / 'feature_evaluation_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\nSaved report to: {OUTPUT_DIR / 'feature_evaluation_report.txt'}")

        return report_text


def main():
    """Main execution"""
    print("="*80)
    print("COMPREHENSIVE FEATURE EVALUATION FOR LOGISTIC REGRESSION")
    print("="*80)

    # Load dataset
    dataset_choice = input("\nWhich dataset to evaluate?\n1. Advanced (47 features)\n2. Combined (28 features)\nChoice (1/2): ").strip()

    if dataset_choice == '1':
        dataset_file = 'training_dataset_advanced.npz'
        dataset_name = 'Advanced (47 features)'
    else:
        dataset_file = 'training_dataset_combined.npz'
        dataset_name = 'Combined (28 features)'

    print(f"\nLoading {dataset_name}...")
    data = np.load(WORKING_DIR / dataset_file, allow_pickle=True)
    X = data['X']
    y = data['y']
    feature_names = [str(f) for f in data['feature_names']]

    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Create evaluator
    evaluator = FeatureEvaluator(X, y, feature_names)

    # Run all evaluation methods
    print("\n" + "="*80)
    print("RUNNING ALL EVALUATION METHODS")
    print("="*80)

    # Method 1: Coefficients with significance
    evaluator.evaluate_coefficients_with_significance()

    # Method 2: Permutation importance
    evaluator.evaluate_permutation_importance(n_repeats=10)

    # Method 3: L1 regularization
    evaluator.evaluate_l1_regularization(cv=5)

    # Method 4: RFE
    n_features_rfe = min(20, X.shape[1] // 2)
    evaluator.evaluate_rfe(n_features_to_select=n_features_rfe)

    # Method 5: Feature subset comparison
    evaluator.compare_feature_subsets()

    # Create visualizations
    evaluator.create_visualizations()

    # Generate summary report
    evaluator.generate_summary_report()

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("  - coefficient_analysis.csv")
    print("  - permutation_importance.csv")
    print("  - l1_lasso_features.csv")
    print("  - l1_regularization_path.csv")
    print("  - rfe_feature_ranking.csv")
    print("  - feature_subset_comparison.csv")
    print("  - feature_evaluation_comprehensive.png")
    print("  - coefficient_vs_permutation.png")
    print("  - feature_evaluation_report.txt")


if __name__ == "__main__":
    main()
