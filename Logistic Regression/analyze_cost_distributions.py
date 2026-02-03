"""
Cost Scaling Calibration for Logistic Regression and Siamese Models

Problem:
- Bhattacharyya costs: range [0-10+], threshold=3
- LR probabilities: [0-1], need scaling to match Bhattacharyya range
- Siamese similarity: [0-1], current cost formula yields [0-2] range

Solution:
- Grid search over scale_factor to maximize decision agreement with Bhattacharyya
- Goal: >85% agreement on which pairs are accepted/rejected

Metrics:
- Agreement rate: Percentage of pairs with same decision (accept/reject)
- Decision matrices: Show true positives/negatives relative to Bhattacharyya
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class CostCalibrator:
    """
    Calibrate LR and Siamese cost functions against Bhattacharyya baseline.
    """

    def __init__(
        self,
        bhattacharyya_cost_fn,
        threshold: float = 3.0,
        scale_factor_range: Tuple[float, float] = (1.0, 10.0),
        time_penalty_range: Optional[Tuple[float, float]] = (0.05, 0.5),
    ):
        """
        Initialize cost calibrator.

        Args:
            bhattacharyya_cost_fn: BhattacharyyaCostFunction instance (baseline)
            threshold: Decision threshold (default 3 for stitch_thresh)
            scale_factor_range: Range to search for scale_factor
            time_penalty_range: Range for time_penalty (for Siamese tuning)
        """
        self.bhatt_cost_fn = bhattacharyya_cost_fn
        self.threshold = threshold
        self.scale_factor_range = scale_factor_range
        self.time_penalty_range = time_penalty_range

        self.calibration_results = {}
        self.bhatt_decisions = None

    def _collect_validation_pairs(
        self, data_dir: str = "."
    ) -> Tuple[List[Dict], List[Dict], np.ndarray]:
        """
        Collect fragment pairs for calibration.

        Note: This requires access to actual fragment data.
        For now, we'll use a placeholder that generates synthetic pairs
        or loads from cache.

        Args:
            data_dir: Directory containing fragment data

        Returns:
            (fragments_a, fragments_b, gt_labels)
        """
        # Try to load from cache
        cache_file = Path(data_dir) / "cost_calibration_pairs.npz"

        if cache_file.exists():
            print(f"Loading validation pairs from cache: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            return data["frags_a"], data["frags_b"], data["labels"]

        # If no cache, raise error
        raise FileNotFoundError(
            f"Validation pairs not found at {cache_file}. "
            f"Please create training data first."
        )

    def compute_bhattacharyya_decisions(
        self,
        fragments_a: List[Dict],
        fragments_b: List[Dict],
        TIME_WIN: float = 15.0,
        param: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Compute Bhattacharyya decisions (accept/reject) for all pairs.

        Args:
            fragments_a: List of first fragments
            fragments_b: List of second fragments
            TIME_WIN: Time window for stitching
            param: Parameters (cx, mx, cy, my, etc.)

        Returns:
            Boolean array where True = accept (cost < threshold), False = reject
        """
        if param is None:
            param = {"cx": 0.2, "mx": 0.1, "cy": 2, "my": 0.1}

        decisions = []
        costs = []

        for frag_a, frag_b in zip(fragments_a, fragments_b):
            cost = self.bhatt_cost_fn.compute_cost(frag_a, frag_b, TIME_WIN, param)
            costs.append(cost)
            decision = cost < self.threshold
            decisions.append(decision)

        self.bhatt_decisions = np.array(decisions)
        self.bhatt_costs = np.array(costs)

        return np.array(decisions)

    def calibrate_lr_model(
        self,
        lr_cost_fn,
        fragments_a: List[Dict],
        fragments_b: List[Dict],
        TIME_WIN: float = 15.0,
        param: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Find optimal scale_factor for LR cost function.

        Grid search over scale_factor values and compute agreement
        with Bhattacharyya decisions.

        Args:
            lr_cost_fn: LogisticRegressionCostFunction instance
            fragments_a: List of first fragments
            fragments_b: List of second fragments
            TIME_WIN: Time window
            param: Parameters

        Returns:
            Dictionary with results: {'optimal_scale_factor', 'agreement_rate', 'results_table'}
        """
        if self.bhatt_decisions is None:
            raise RuntimeError("Must call compute_bhattacharyya_decisions() first")

        print("\n" + "=" * 80)
        print("LR MODEL COST SCALING CALIBRATION")
        print("=" * 80)

        if param is None:
            param = {"cx": 0.2, "mx": 0.1, "cy": 2, "my": 0.1}

        # Grid search over scale factors
        scale_factors = np.linspace(
            self.scale_factor_range[0], self.scale_factor_range[1], 20
        )

        results = []

        for scale_factor in scale_factors:
            # Update model's scale factor
            lr_cost_fn.scale_factor = scale_factor

            # Compute decisions
            decisions = []
            for frag_a, frag_b in zip(fragments_a, fragments_b):
                cost = lr_cost_fn.compute_cost(frag_a, frag_b, TIME_WIN, param)
                decision = cost < self.threshold
                decisions.append(decision)

            decisions = np.array(decisions)

            # Compute agreement with Bhattacharyya
            agreement = np.mean(decisions == self.bhatt_decisions)
            tp = np.sum((decisions == True) & (self.bhatt_decisions == True))
            tn = np.sum((decisions == False) & (self.bhatt_decisions == False))
            fp = np.sum((decisions == True) & (self.bhatt_decisions == False))
            fn = np.sum((decisions == False) & (self.bhatt_decisions == True))

            results.append(
                {
                    "scale_factor": scale_factor,
                    "agreement_rate": agreement,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                }
            )

            print(f"Scale factor: {scale_factor:.2f} | Agreement: {agreement:.2%}")

        # Find optimal scale factor
        results_df = pd.DataFrame(results)
        best_idx = results_df["agreement_rate"].idxmax()
        optimal = results_df.loc[best_idx]

        print(f"\n✓ Optimal scale factor: {optimal['scale_factor']:.2f}")
        print(f"  Agreement rate: {optimal['agreement_rate']:.2%}")
        print(f"  TP: {optimal['tp']}, TN: {optimal['tn']}")
        print(f"  FP: {optimal['fp']}, FN: {optimal['fn']}")

        self.calibration_results["lr"] = {
            "optimal_scale_factor": optimal["scale_factor"],
            "agreement_rate": optimal["agreement_rate"],
            "results_table": results_df,
        }

        return self.calibration_results["lr"]

    def calibrate_siamese_model(
        self,
        siamese_cost_fn,
        fragments_a: List[Dict],
        fragments_b: List[Dict],
        TIME_WIN: float = 15.0,
        param: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Find optimal scale_factor and time_penalty for Siamese cost function.

        Note: Siamese is more complex, performing 2D grid search.

        Args:
            siamese_cost_fn: SiameseCostFunction instance
            fragments_a: List of first fragments
            fragments_b: List of second fragments
            TIME_WIN: Time window
            param: Parameters

        Returns:
            Dictionary with results
        """
        if self.bhatt_decisions is None:
            raise RuntimeError("Must call compute_bhattacharyya_decisions() first")

        print("\n" + "=" * 80)
        print("SIAMESE MODEL COST SCALING CALIBRATION (2D Grid Search)")
        print("=" * 80)

        if param is None:
            param = {"cx": 0.2, "mx": 0.1, "cy": 2, "my": 0.1}

        # Grid search over scale factors and time penalties
        scale_factors = np.linspace(
            self.scale_factor_range[0], self.scale_factor_range[1], 10
        )
        time_penalties = (
            np.linspace(
                self.time_penalty_range[0], self.time_penalty_range[1], 10
            )
            if self.time_penalty_range
            else [0.1]
        )

        results = []

        for scale_factor in scale_factors:
            for time_penalty in time_penalties:
                # Modify Siamese cost computation temporarily
                # Store original values
                orig_scale = getattr(siamese_cost_fn, "_scale_factor_override", None)
                orig_time = getattr(siamese_cost_fn, "_time_penalty_override", None)

                # Set overrides
                siamese_cost_fn._scale_factor_override = scale_factor
                siamese_cost_fn._time_penalty_override = time_penalty

                # Compute decisions
                decisions = []
                try:
                    for frag_a, frag_b in zip(fragments_a, fragments_b):
                        cost = siamese_cost_fn.compute_cost(
                            frag_a, frag_b, TIME_WIN, param
                        )
                        decision = cost < self.threshold
                        decisions.append(decision)

                    decisions = np.array(decisions)

                    # Compute agreement
                    agreement = np.mean(decisions == self.bhatt_decisions)

                    results.append(
                        {
                            "scale_factor": scale_factor,
                            "time_penalty": time_penalty,
                            "agreement_rate": agreement,
                        }
                    )

                finally:
                    # Restore original values
                    if orig_scale is not None:
                        siamese_cost_fn._scale_factor_override = orig_scale
                    if orig_time is not None:
                        siamese_cost_fn._time_penalty_override = orig_time

        # Find optimal combination
        results_df = pd.DataFrame(results)
        best_idx = results_df["agreement_rate"].idxmax()
        optimal = results_df.loc[best_idx]

        print(f"\n✓ Optimal Siamese parameters:")
        print(f"  Scale factor: {optimal['scale_factor']:.2f}")
        print(f"  Time penalty: {optimal['time_penalty']:.4f}")
        print(f"  Agreement rate: {optimal['agreement_rate']:.2%}")

        self.calibration_results["siamese"] = {
            "optimal_scale_factor": optimal["scale_factor"],
            "optimal_time_penalty": optimal["time_penalty"],
            "agreement_rate": optimal["agreement_rate"],
            "results_table": results_df,
        }

        return self.calibration_results["siamese"]

    def plot_cost_distributions(
        self,
        lr_costs: Optional[np.ndarray] = None,
        siamese_costs: Optional[np.ndarray] = None,
        output_file: str = "cost_distributions.png",
    ) -> None:
        """
        Visualize cost distributions for all three methods.

        Args:
            lr_costs: Array of LR costs (optional)
            siamese_costs: Array of Siamese costs (optional)
            output_file: Output filename for plot
        """
        if plt is None:
            print("matplotlib not available, skipping visualization")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Bhattacharyya distribution
        ax = axes[0, 0]
        positive_idx = self.bhatt_decisions == True
        ax.hist(
            self.bhatt_costs[positive_idx],
            bins=30,
            alpha=0.6,
            label="Same vehicle",
        )
        ax.hist(
            self.bhatt_costs[~positive_idx],
            bins=30,
            alpha=0.6,
            label="Different vehicle",
        )
        ax.axvline(self.threshold, color="r", linestyle="--", label=f"Threshold={self.threshold}")
        ax.set_xlabel("Cost")
        ax.set_ylabel("Count")
        ax.set_title("Bhattacharyya Cost Distribution")
        ax.legend()

        # Plot 2: LR costs (if available)
        if lr_costs is not None:
            ax = axes[0, 1]
            ax.hist(
                lr_costs[positive_idx],
                bins=30,
                alpha=0.6,
                label="Same vehicle",
            )
            ax.hist(
                lr_costs[~positive_idx],
                bins=30,
                alpha=0.6,
                label="Different vehicle",
            )
            ax.axvline(self.threshold, color="r", linestyle="--", label=f"Threshold={self.threshold}")
            ax.set_xlabel("Cost")
            ax.set_ylabel("Count")
            ax.set_title("LR Cost Distribution (Calibrated)")
            ax.legend()

        # Plot 3: Siamese costs (if available)
        if siamese_costs is not None:
            ax = axes[1, 0]
            ax.hist(
                siamese_costs[positive_idx],
                bins=30,
                alpha=0.6,
                label="Same vehicle",
            )
            ax.hist(
                siamese_costs[~positive_idx],
                bins=30,
                alpha=0.6,
                label="Different vehicle",
            )
            ax.axvline(self.threshold, color="r", linestyle="--", label=f"Threshold={self.threshold}")
            ax.set_xlabel("Cost")
            ax.set_ylabel("Count")
            ax.set_title("Siamese Cost Distribution (Calibrated)")
            ax.legend()

        # Plot 4: Summary comparison
        ax = axes[1, 1]
        methods = ["Bhattacharyya", "LR", "Siamese"]
        agreements = [1.0]  # Bhattacharyya is baseline (100% agreement)

        if "lr" in self.calibration_results:
            agreements.append(self.calibration_results["lr"]["agreement_rate"])
        if "siamese" in self.calibration_results:
            agreements.append(self.calibration_results["siamese"]["agreement_rate"])

        colors = ["green" if a > 0.85 else "orange" if a > 0.75 else "red" for a in agreements]
        ax.bar(methods[:len(agreements)], agreements, color=colors)
        ax.axhline(0.85, color="g", linestyle="--", alpha=0.5, label="Goal (85%)")
        ax.set_ylabel("Agreement Rate")
        ax.set_title("Decision Agreement with Bhattacharyya")
        ax.set_ylim([0.6, 1.05])
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"\n✓ Saved cost distribution plot to {output_file}")


def main(data_dir: str = "."):
    """
    Main cost scaling calibration pipeline.

    Args:
        data_dir: Directory containing data
    """
    print("=" * 80)
    print("COST SCALING CALIBRATION FOR LR AND SIAMESE MODELS")
    print("=" * 80)

    # Initialize calibrator
    from utils.stitch_cost_interface import (
        BhattacharyyaCostFunction,
        LogisticRegressionCostFunction,
        SiameseCostFunction,
    )

    bhatt_fn = BhattacharyyaCostFunction()
    calibrator = CostCalibrator(bhatt_fn, threshold=3.0)

    # Try to load validation pairs
    try:
        frags_a, frags_b, labels = calibrator._collect_validation_pairs(data_dir)
        print(f"Loaded {len(frags_a)} validation pairs")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Proceeding with empty validation set (preview mode)")
        frags_a, frags_b, labels = [], [], []

    if len(frags_a) > 0:
        # Compute Bhattacharyya decisions
        calibrator.compute_bhattacharyya_decisions(frags_a, frags_b)

        # Calibrate LR (if model exists)
        try:
            lr_fn = LogisticRegressionCostFunction(
                model_path="model_artifacts/combined_optimal_10features.pkl"
            )
            lr_results = calibrator.calibrate_lr_model(lr_fn, frags_a, frags_b)
        except Exception as e:
            print(f"Could not calibrate LR model: {e}")
            lr_results = None

        # Calibrate Siamese (if model exists)
        try:
            siamese_fn = SiameseCostFunction(
                checkpoint_path="../Siamese-Network/outputs/best_accuracy.pth"
            )
            siamese_results = calibrator.calibrate_siamese_model(
                siamese_fn, frags_a, frags_b
            )
        except Exception as e:
            print(f"Could not calibrate Siamese model: {e}")
            siamese_results = None

        # Save calibration report
        output_dir = Path("feature_selection_outputs")
        output_dir.mkdir(exist_ok=True)

        report = f"""COST SCALING CALIBRATION REPORT
==============================

Baseline Threshold: {calibrator.threshold}

Logistic Regression Results:
"""
        if lr_results:
            report += f"""  Optimal scale_factor: {lr_results['optimal_scale_factor']:.2f}
  Agreement with Bhattacharyya: {lr_results['agreement_rate']:.2%}
  Status: {'✓ GOOD' if lr_results['agreement_rate'] > 0.85 else '⚠ NEEDS TUNING'}
"""
        else:
            report += "  (Not calibrated - model not available)\n"

        report += "\nSiamese Results:\n"
        if siamese_results:
            report += f"""  Optimal scale_factor: {siamese_results['optimal_scale_factor']:.2f}
  Optimal time_penalty: {siamese_results['optimal_time_penalty']:.4f}
  Agreement with Bhattacharyya: {siamese_results['agreement_rate']:.2%}
  Status: {'✓ GOOD' if siamese_results['agreement_rate'] > 0.85 else '⚠ NEEDS TUNING'}
"""
        else:
            report += "  (Not calibrated - model not available)\n"

        (output_dir / "cost_calibration_report.md").write_text(report)
        print(f"\n✓ Saved calibration report")

        # Save calibration parameters
        calibration_params = {}
        if lr_results:
            calibration_params["logistic_regression"] = {
                "scale_factor": float(lr_results["optimal_scale_factor"]),
                "agreement_rate": float(lr_results["agreement_rate"]),
            }
        if siamese_results:
            calibration_params["siamese"] = {
                "scale_factor": float(siamese_results["optimal_scale_factor"]),
                "time_penalty": float(siamese_results["optimal_time_penalty"]),
                "agreement_rate": float(siamese_results["agreement_rate"]),
            }

        (output_dir / "optimal_scaling_params.json").write_text(
            json.dumps(calibration_params, indent=2)
        )
        print(f"✓ Saved optimal scaling parameters")

    print("\n" + "=" * 80)
    print("COST SCALING CALIBRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
