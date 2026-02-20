"""
Cross-Scenario Validation for Model Generalization

Validates logistic regression model across different traffic scenarios (i, ii, iii).

Test Strategy:
- 9 train/test splits across scenarios
- Measures generalization via AUC drop (goal: <10%)
- Analyzes feature stability via Jaccard similarity (goal: >70%)
- Tests elastic net regularization (L1+L2) for robustness

Overfitting Indicators:
- Severe: AUC drop > 15%
- Moderate: AUC drop 10-15%
- Good: AUC drop < 10%
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict
import argparse

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from itertools import combinations


class CrossScenarioValidator:
    """
    Cross-scenario validation for generalization testing.
    """

    def __init__(self):
        self.results = []
        self.feature_stability_scores = {}

    def load_scenario_data(
        self, scenario: str, data_dir: str = "."
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load training data for a specific scenario.

        Args:
            scenario: Scenario identifier ('i', 'ii', 'iii')
            data_dir: Directory containing NPZ files

        Returns:
            (X, y, feature_names)
        """
        # Files are expected to be named like:
        # training_dataset_advanced_i.npz
        # training_dataset_advanced_ii.npz
        # training_dataset_advanced_iii.npz

        filename = f"training_dataset_advanced_{scenario}.npz"
        filepath = Path(data_dir) / filename

        if not filepath.exists():
            # Fallback: try to use single combined file and filter by scenario
            # This would require additional metadata in the dataset
            print(f"Warning: {filepath} not found")
            raise FileNotFoundError(f"Cannot find data for scenario {scenario}")

        data = np.load(filepath)
        X = data["X"]
        y = data["y"]
        feature_names = list(data["feature_names"])

        # Remove NaN
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        print(f"Scenario {scenario}: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names

    def run_cross_scenario_splits(
        self,
        scenarios: List[str] = ["i", "ii", "iii"],
        data_dir: str = ".",
    ) -> pd.DataFrame:
        """
        Run all 9 train/test splits (including single-scenario within-fold).

        Train/test splits:
        1. i → ii (free-flow → snowy)
        2. i → iii (free-flow → congested)
        3. ii → i (snowy → free-flow)
        4. ii → iii (snowy → congested)
        5. iii → i (congested → free-flow)
        6. iii → ii (congested → snowy)
        7. i+ii → iii (combined → congested)
        8. i+iii → ii (combined → snowy)
        9. ii+iii → i (combined → free-flow)

        Args:
            scenarios: List of scenario identifiers
            data_dir: Directory containing data files

        Returns:
            DataFrame with results for all splits
        """
        print("=" * 80)
        print("CROSS-SCENARIO VALIDATION")
        print("=" * 80)

        results = []
        split_number = 0

        # Single-scenario splits (3)
        for i, scenario in enumerate(scenarios):
            split_number += 1

            # Load data for this scenario
            X, y, feature_names = self.load_scenario_data(scenario, data_dir)

            # Use stratified k-fold for within-scenario validation
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train elastic net model
                model = self._train_elastic_net(X_train, y_train)

                # Evaluate
                auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
                auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

                result = {
                    "split_id": split_number,
                    "split_type": "within-scenario",
                    "train_scenario": scenario,
                    "test_scenario": scenario,
                    "fold": fold_idx,
                    "auc_train": auc_train,
                    "auc_test": auc_test,
                    "auc_drop": auc_train - auc_test,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "status": self._classify_overfitting(auc_train - auc_test),
                }
                results.append(result)

        # Cross-scenario splits (6)
        for train_scenario in scenarios:
            for test_scenario in scenarios:
                if train_scenario == test_scenario:
                    continue

                split_number += 1

                # Load data
                X_train, y_train, feature_names_train = self.load_scenario_data(
                    train_scenario, data_dir
                )
                X_test, y_test, feature_names_test = self.load_scenario_data(
                    test_scenario, data_dir
                )

                # Validate feature order matches
                assert (
                    feature_names_train == feature_names_test
                ), "Feature mismatch across scenarios"

                # Train elastic net model
                model = self._train_elastic_net(X_train, y_train)

                # Evaluate
                auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
                auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

                result = {
                    "split_id": split_number,
                    "split_type": "cross-scenario",
                    "train_scenario": train_scenario,
                    "test_scenario": test_scenario,
                    "fold": 0,
                    "auc_train": auc_train,
                    "auc_test": auc_test,
                    "auc_drop": auc_train - auc_test,
                    "n_train": X_train.shape[0],
                    "n_test": X_test.shape[0],
                    "status": self._classify_overfitting(auc_train - auc_test),
                }
                results.append(result)

        # Combined scenario splits (3)
        for hold_out_idx, hold_out_scenario in enumerate(scenarios):
            split_number += 1

            # Get training scenarios (all except hold-out)
            train_scenarios = [s for s in scenarios if s != hold_out_scenario]

            # Load and combine training data
            X_train_list = []
            y_train_list = []
            for scenario in train_scenarios:
                X, y, feature_names = self.load_scenario_data(scenario, data_dir)
                X_train_list.append(X)
                y_train_list.append(y)

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)

            # Load test data
            X_test, y_test, feature_names = self.load_scenario_data(
                hold_out_scenario, data_dir
            )

            # Train elastic net model
            model = self._train_elastic_net(X_train, y_train)

            # Evaluate
            auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            result = {
                "split_id": split_number,
                "split_type": "combined",
                "train_scenario": "+".join(train_scenarios),
                "test_scenario": hold_out_scenario,
                "fold": 0,
                "auc_train": auc_train,
                "auc_test": auc_test,
                "auc_drop": auc_train - auc_test,
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "status": self._classify_overfitting(auc_train - auc_test),
            }
            results.append(result)

        self.results = results
        return pd.DataFrame(results)

    def _train_elastic_net(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train logistic regression with elastic net regularization.

        Uses LogisticRegressionCV to find optimal L1 ratio and C.
        """
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Grid search over L1 ratios and regularization strengths
        model = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 20),
            l1_ratios=np.linspace(0.1, 0.9, 9),
            penalty="elasticnet",
            solver="saga",
            cv=5,
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled, y_train)

        # Store scaler for later use
        model.scaler = scaler

        return model

    def _classify_overfitting(self, auc_drop: float) -> str:
        """Classify overfitting severity based on AUC drop."""
        if auc_drop < 0.10:
            return "Good"
        elif auc_drop < 0.15:
            return "Moderate"
        else:
            return "Severe"

    def analyze_feature_stability(
        self, scenarios: List[str] = ["i", "ii", "iii"], data_dir: str = "."
    ) -> Dict[str, float]:
        """
        Analyze feature stability by running feature selection on each scenario.

        Computes Jaccard similarity between feature sets selected per scenario.
        Goal: Similarity > 70% indicates stable, scenario-independent features.

        Args:
            scenarios: List of scenarios
            data_dir: Directory containing data

        Returns:
            Dictionary mapping 'stability_score' and pairwise similarities
        """
        print("\n" + "=" * 80)
        print("FEATURE STABILITY ANALYSIS")
        print("=" * 80)

        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        # Run feature selection on each scenario
        selected_features_by_scenario = {}

        for scenario in scenarios:
            X, y, feature_names = self.load_scenario_data(scenario, data_dir)

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # RFE to select top 10 features
            lr = LogisticRegression(max_iter=1000, random_state=42)
            rfe = RFE(lr, n_features_to_select=10, step=1)
            rfe.fit(X_scaled, y)

            selected = set(
                [feature_names[i] for i, sel in enumerate(rfe.support_) if sel]
            )
            selected_features_by_scenario[scenario] = selected
            print(f"Scenario {scenario}: {len(selected)} features selected")

        # Compute pairwise Jaccard similarities
        stability_scores = {}
        for s1, s2 in combinations(scenarios, 2):
            set1 = selected_features_by_scenario[s1]
            set2 = selected_features_by_scenario[s2]

            jaccard = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
            stability_scores[f"{s1}_{s2}"] = jaccard
            print(f"Jaccard {s1} ↔ {s2}: {jaccard:.2%}")

        # Overall stability score
        overall_stability = np.mean(list(stability_scores.values()))
        stability_scores["overall"] = overall_stability
        print(f"\nOverall Feature Stability: {overall_stability:.2%}")

        self.feature_stability_scores = stability_scores
        return stability_scores

    def run_source_holdout_split(self, npz_path: str) -> pd.DataFrame:
        """
        Run a single source-holdout evaluation from an NPZ with source_split_tag.

        Returns one-row DataFrame with either executed metrics or skipped reason.
        """
        path = Path(npz_path)
        if not path.exists():
            return pd.DataFrame([{
                "split_type": "source_holdout",
                "status": "skipped",
                "reason": f"dataset not found: {path}",
            }])

        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        if "source_split_tag" not in data:
            return pd.DataFrame([{
                "split_type": "source_holdout",
                "status": "skipped",
                "reason": "source_split_tag missing",
            }])

        tags = np.asarray(data["source_split_tag"]).astype(str)
        unique_tags, counts = np.unique(tags, return_counts=True)
        if len(unique_tags) < 2:
            return pd.DataFrame([{
                "split_type": "source_holdout",
                "status": "skipped",
                "reason": "fewer than 2 source tags",
            }])

        if {"advanced_keepall", "v4_diverse_curated"}.issubset(set(unique_tags)):
            train_tag, test_tag = "advanced_keepall", "v4_diverse_curated"
        else:
            order = np.argsort(counts)[::-1]
            train_tag = unique_tags[order[0]]
            test_tag = unique_tags[order[1]]

        train_mask = tags == train_tag
        test_mask = tags == test_tag
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if (len(np.unique(y_train)) < 2) or (len(np.unique(y_test)) < 2):
            return pd.DataFrame([{
                "split_type": "source_holdout",
                "status": "skipped",
                "reason": "single-class split",
                "train_tag": train_tag,
                "test_tag": test_tag,
            }])

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ])
        pipe.fit(X_train, y_train)

        y_train_prob = pipe.predict_proba(X_train)[:, 1]
        y_test_prob = pipe.predict_proba(X_test)[:, 1]
        auc_train = roc_auc_score(y_train, y_train_prob)
        auc_test = roc_auc_score(y_test, y_test_prob)

        return pd.DataFrame([{
            "split_type": "source_holdout",
            "status": "executed",
            "reason": None,
            "train_tag": train_tag,
            "test_tag": test_tag,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "auc_train": float(auc_train),
            "auc_test": float(auc_test),
            "auc_drop": float(auc_train - auc_test),
        }])


def main(data_dir: str = ".", source_holdout_npz: str = None):
    """
    Main cross-scenario validation pipeline.

    Args:
        data_dir: Directory containing training data files
    """
    print("=" * 80)
    print("CROSS-SCENARIO VALIDATION FOR GENERALIZATION TESTING")
    print("=" * 80)

    validator = CrossScenarioValidator()

    # Run all train/test splits
    results_df = validator.run_cross_scenario_splits(
        scenarios=["i", "ii", "iii"], data_dir=data_dir
    )

    # Analyze feature stability
    stability_scores = validator.analyze_feature_stability(
        scenarios=["i", "ii", "iii"], data_dir=data_dir
    )

    # Save results
    output_dir = Path("feature_selection_outputs")
    output_dir.mkdir(exist_ok=True)

    # Results CSV
    results_df.to_csv(output_dir / "cross_scenario_results.csv", index=False)
    print(f"\n✓ Saved cross-scenario results")

    source_holdout_df = None
    if source_holdout_npz:
        source_holdout_df = validator.run_source_holdout_split(source_holdout_npz)
        source_holdout_df.to_csv(output_dir / "cross_scenario_source_holdout.csv", index=False)
        print("✓ Saved source-holdout results")

    # Summary statistics
    print("\n" + "=" * 80)
    print("CROSS-SCENARIO SUMMARY")
    print("=" * 80)

    # By split type
    print("\nResults by split type:")
    for split_type in ["within-scenario", "cross-scenario", "combined"]:
        subset = results_df[results_df["split_type"] == split_type]
        if len(subset) > 0:
            print(f"\n{split_type.upper()}:")
            print(f"  Count: {len(subset)}")
            print(f"  Avg AUC Train: {subset['auc_train'].mean():.4f}")
            print(f"  Avg AUC Test: {subset['auc_test'].mean():.4f}")
            print(f"  Avg AUC Drop: {subset['auc_drop'].mean():.4f}")
            print(f"  Status:")
            for status, count in subset["status"].value_counts().items():
                print(f"    {status}: {count}")

    # Overall conclusion
    overall_drop = results_df["auc_drop"].mean()
    if overall_drop < 0.10:
        conclusion = "✓ GOOD - Model generalizes well across scenarios"
    elif overall_drop < 0.15:
        conclusion = "⚠ MODERATE - Some overfitting detected, consider regularization"
    else:
        conclusion = (
            "✗ SEVERE - Significant overfitting, retrain with more data/regularization"
        )

    print(f"\n{conclusion}")

    # Feature stability conclusion
    stability = stability_scores["overall"]
    if stability > 0.70:
        stability_conclusion = "✓ GOOD - Features are stable across scenarios"
    else:
        stability_conclusion = (
            "⚠ WARNING - Features are scenario-specific, may not generalize"
        )

    print(f"Feature Stability: {stability:.2%} - {stability_conclusion}")

    # Save comprehensive report
    report = f"""CROSS-SCENARIO VALIDATION REPORT
================================

Data:
  Scenarios: i (free-flow), ii (snowy), iii (congested)
  Total splits tested: {len(results_df)}

Cross-Scenario Validation Results:
  Overall AUC Drop: {overall_drop:.4f}
  Conclusion: {conclusion}

Feature Stability Analysis:
  Overall Jaccard Similarity: {stability:.2%}
  Conclusion: {stability_conclusion}

Detailed Results by Split Type:
"""

    for split_type in ["within-scenario", "cross-scenario", "combined"]:
        subset = results_df[results_df["split_type"] == split_type]
        if len(subset) > 0:
            report += f"\n{split_type.upper()}:\n"
            report += subset.to_string(index=False)
            report += "\n"

    if source_holdout_df is not None:
        report += "\nSOURCE_HOLDOUT:\n"
        report += source_holdout_df.to_string(index=False)
        report += "\n"

    (output_dir / "cross_scenario_validation_report.md").write_text(report)
    print(f"✓ Saved comprehensive validation report")

    print("\n" + "=" * 80)
    print("CROSS-SCENARIO VALIDATION COMPLETE")
    print("=" * 80)

    return validator, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-scenario and source-holdout validation")
    parser.add_argument("--data-dir", default=".", help="Directory containing per-scenario NPZs")
    parser.add_argument(
        "--source-holdout-npz",
        default=None,
        help="Optional combined NPZ path with source_split_tag for source-holdout validation",
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir, source_holdout_npz=args.source_holdout_npz)
