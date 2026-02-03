#!/usr/bin/env python3
"""
Automated Benchmark Script for All Scenarios and Cost Functions

Runs the full I24 pipeline with all combinations:
- Scenarios: i (free-flow), ii (snowy), iii (congested)
- Cost functions: bhattacharyya, logistic_regression, siamese

Outputs:
- benchmark_results.csv: Complete results table
- benchmark_summary.txt: Human-readable summary
- Visualizations (if matplotlib available)
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys
import time


class PipelineBenchmark:
    """Orchestrates benchmarking across scenarios and cost functions."""

    def __init__(self, pipeline_script: str = "pp_lite.py", mot_script: str = "mot_i24.py"):
        """
        Initialize benchmark.

        Args:
            pipeline_script: Path to pipeline script (pp_lite.py)
            mot_script: Path to MOT evaluation script (mot_i24.py)
        """
        self.pipeline_script = pipeline_script
        self.mot_script = mot_script
        self.results = []
        self.scenarios = ["i", "ii", "iii"]
        self.cost_functions = ["bhattacharyya", "logistic_regression", "siamese"]

    def run_pipeline(
        self,
        scenario: str,
        cost_function: str,
        scale_factor: float = None,
    ) -> bool:
        """
        Run pipeline for specific scenario and cost function.

        Args:
            scenario: Scenario identifier
            cost_function: Cost function type
            scale_factor: Optional scale factor override

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"Running pipeline: Scenario {scenario}, Cost={cost_function}")
        if scale_factor:
            print(f"  Scale factor: {scale_factor}")
        print("=" * 80)

        # Build command
        cmd = [
            "python",
            self.pipeline_script,
            "--scenario",
            scenario,
            "--cost-function",
            cost_function,
        ]

        if scale_factor:
            cmd.extend(["--scale-factor", str(scale_factor)])

        try:
            result = subprocess.run(cmd, check=True, timeout=3600)  # 1 hour timeout
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Pipeline failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            print(f"✗ Pipeline timed out")
            return False

    def evaluate_mot(self, scenario: str) -> Dict[str, float]:
        """
        Run MOT evaluation and extract metrics.

        Args:
            scenario: Scenario identifier

        Returns:
            Dictionary with MOT metrics (MOTA, MOTP, IDF1, etc.)
        """
        print(f"Evaluating MOT metrics for scenario {scenario}...")

        try:
            result = subprocess.run(
                ["python", self.mot_script, scenario],
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )

            # Parse output to extract metrics
            # Expected format (may vary based on mot_i24.py implementation)
            metrics = self._parse_mot_output(result.stdout, result.stderr)
            return metrics

        except Exception as e:
            print(f"✗ MOT evaluation failed: {e}")
            return {}

    def _parse_mot_output(self, stdout: str, stderr: str) -> Dict[str, float]:
        """
        Parse MOT evaluation output.

        Args:
            stdout: Standard output from mot_i24.py
            stderr: Standard error from mot_i24.py

        Returns:
            Dictionary with extracted metrics
        """
        output = stdout + stderr
        metrics = {}

        # Common MOT metric patterns
        metric_patterns = {
            "MOTA": r"MOTA[:\s]+([0-9.]+)",
            "MOTP": r"MOTP[:\s]+([0-9.]+)",
            "IDF1": r"IDF1[:\s]+([0-9.]+)",
            "Precision": r"Precision[:\s]+([0-9.]+)",
            "Recall": r"Recall[:\s]+([0-9.]+)",
        }

        import re

        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                except ValueError:
                    pass

        return metrics

    def run_full_benchmark(
        self, dry_run: bool = False
    ) -> pd.DataFrame:
        """
        Run full benchmark across all scenarios and cost functions.

        Args:
            dry_run: If True, print commands without running

        Returns:
            DataFrame with results
        """
        print("=" * 80)
        print("FULL PIPELINE BENCHMARK")
        print("=" * 80)
        print(f"Scenarios: {self.scenarios}")
        print(f"Cost functions: {self.cost_functions}")
        print(f"Total combinations: {len(self.scenarios) * len(self.cost_functions)}")

        combination_count = 0

        for scenario in self.scenarios:
            for cost_function in self.cost_functions:
                combination_count += 1
                print(f"\n[{combination_count}/{len(self.scenarios) * len(self.cost_functions)}]")

                if dry_run:
                    print(f"DRY RUN: Would run scenario {scenario} with {cost_function}")
                    continue

                # Run pipeline
                success = self.run_pipeline(scenario, cost_function)

                if not success:
                    print(f"⚠ Skipping MOT evaluation due to pipeline failure")
                    continue

                # Evaluate MOT
                metrics = self.evaluate_mot(scenario)

                # Store result
                result = {
                    "scenario": scenario,
                    "cost_function": cost_function,
                    "success": True,
                }
                result.update(metrics)
                self.results.append(result)

                # Small delay between runs
                time.sleep(5)

        return pd.DataFrame(self.results)

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate human-readable benchmark report.

        Args:
            results_df: Results DataFrame

        Returns:
            Report string
        """
        report = "BENCHMARK RESULTS SUMMARY\n"
        report += "=" * 80 + "\n\n"

        if len(results_df) == 0:
            report += "No results available\n"
            return report

        # Overall statistics
        report += "OVERALL STATISTICS\n"
        report += "-" * 80 + "\n"
        for metric in ["MOTA", "MOTP", "IDF1"]:
            if metric in results_df.columns:
                report += f"{metric}:\n"
                report += f"  Mean: {results_df[metric].mean():.4f}\n"
                report += f"  Std:  {results_df[metric].std():.4f}\n"
                report += f"  Min:  {results_df[metric].min():.4f}\n"
                report += f"  Max:  {results_df[metric].max():.4f}\n"

        # By scenario
        report += "\nRESULTS BY SCENARIO\n"
        report += "-" * 80 + "\n"
        for scenario in self.scenarios:
            subset = results_df[results_df["scenario"] == scenario]
            if len(subset) > 0:
                report += f"\nScenario {scenario}:\n"
                report += subset[
                    ["cost_function", "MOTA", "MOTP", "IDF1"]
                ].to_string(index=False)
                report += "\n"

        # By cost function
        report += "\nRESULTS BY COST FUNCTION\n"
        report += "-" * 80 + "\n"
        for cf in self.cost_functions:
            subset = results_df[results_df["cost_function"] == cf]
            if len(subset) > 0:
                report += f"\n{cf}:\n"
                report += subset[
                    ["scenario", "MOTA", "MOTP", "IDF1"]
                ].to_string(index=False)
                report += "\n"

        # Best performers
        report += "\nBEST PERFORMERS\n"
        report += "-" * 80 + "\n"
        for metric in ["MOTA", "MOTP", "IDF1"]:
            if metric in results_df.columns:
                best_idx = results_df[metric].idxmax()
                best_row = results_df.loc[best_idx]
                report += f"Best {metric}: {best_row['scenario']} + {best_row['cost_function']} = {best_row[metric]:.4f}\n"

        return report

    def save_results(self, results_df: pd.DataFrame, output_dir: str = ".") -> None:
        """
        Save benchmark results to files.

        Args:
            results_df: Results DataFrame
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # CSV
        csv_file = output_path / "benchmark_results.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"\n✓ Saved results to {csv_file}")

        # Summary report
        report = self.generate_report(results_df)
        report_file = output_path / "benchmark_summary.txt"
        report_file.write_text(report)
        print(f"✓ Saved summary to {report_file}")

        # JSON for programmatic access
        json_file = output_path / "benchmark_results.json"
        json_file.write_text(results_df.to_json(orient="records", indent=2))
        print(f"✓ Saved JSON results to {json_file}")

        print("\n" + report)


def main():
    """Main benchmark entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="I24 Pipeline Benchmark")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for results",
    )
    args = parser.parse_args()

    benchmark = PipelineBenchmark()

    # Run full benchmark
    results_df = benchmark.run_full_benchmark(dry_run=args.dry_run)

    # Save results
    if not args.dry_run:
        benchmark.save_results(results_df, args.output_dir)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    main()
