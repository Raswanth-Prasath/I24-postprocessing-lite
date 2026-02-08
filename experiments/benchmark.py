"""
Benchmark Runner for All Cost Function Approaches

For each cost function x scenario:
  1. Updates parameters.json
  2. Runs pp_lite.py (pipeline)
  3. Runs hota_trackeval.py (evaluation)
  4. Collects results into a comparison table

Usage:
    conda activate i24
    python experiments/benchmark.py                    # Run all
    python experiments/benchmark.py --methods lr mlp    # Only specific methods
    python experiments/benchmark.py --scenarios i ii    # Only specific scenarios
    python experiments/benchmark.py --eval-only         # Skip pipeline, just evaluate
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

# Cost function configurations
COST_CONFIGS = {
    'bhattacharyya': {
        'type': 'bhattacharyya',
    },
    'lr': {
        'type': 'logistic_regression',
        'model_path': 'Logistic Regression/model_artifacts/consensus_top10_full47.pkl',
        'scale_factor': 5.0,
        'time_penalty': 0.1,
    },
    'siamese': {
        'type': 'siamese',
        'checkpoint_path': 'Siamese-Network/outputs/best_accuracy.pth',
        'device': 'cpu',
        'scale_factor': 5.0,
        'time_penalty': 0.1,
    },
    'mlp': {
        'type': 'mlp',
        'checkpoint_path': 'models/outputs/mlp_stitch_final.pth',
        'device': 'cpu',
        'scale_factor': 5.0,
        'time_penalty': 0.1,
    },
    'tcn': {
        'type': 'tcn',
        'checkpoint_path': 'models/outputs/tcn_stitch_model.pth',
        'device': 'cpu',
        'scale_factor': 5.0,
        'time_penalty': 0.1,
    },
    'transformer': {
        'type': 'transformer',
        'checkpoint_path': 'models/outputs/transformer_stitch_model.pth',
        'device': 'cpu',
        'scale_factor': 5.0,
        'time_penalty': 0.1,
    },
}

# Stitcher thresholds per method (bhattacharyya uses default 3/4, learned models use scaled)
STITCHER_ARGS = {
    'bhattacharyya': {'stitch_thresh': 3, 'master_stitch_thresh': 4},
    'siamese': {'stitch_thresh': 3, 'master_stitch_thresh': 4},
    'lr': {'stitch_thresh': 3, 'master_stitch_thresh': 4},
    'mlp': {'stitch_thresh': 3, 'master_stitch_thresh': 4},
    'tcn': {'stitch_thresh': 3, 'master_stitch_thresh': 4},
    'transformer': {'stitch_thresh': 3, 'master_stitch_thresh': 4},
}

SCENARIOS = ['i', 'ii', 'iii']


def update_parameters(method: str):
    """Update parameters.json with the given cost function config."""
    params_path = PROJECT_ROOT / 'parameters.json'
    with open(params_path, 'r') as f:
        params = json.load(f)

    params['cost_function'] = COST_CONFIGS[method]

    # Update stitcher thresholds if specified
    if method in STITCHER_ARGS:
        for key, val in STITCHER_ARGS[method].items():
            params['stitcher_args'][key] = val

    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"Updated parameters.json: cost_function.type={method}")


def update_scenario(scenario: str):
    """Update parameters.json collection names for the given scenario."""
    params_path = PROJECT_ROOT / 'parameters.json'
    with open(params_path, 'r') as f:
        params = json.load(f)

    params['gt_collection'] = f'GT_{scenario}'
    params['raw_collection'] = f'RAW_{scenario}'
    params['reconciled_collection'] = f'REC_{scenario}'

    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)


def run_pipeline(scenario: str):
    """Run pp_lite.py for a scenario."""
    cmd = [sys.executable, str(PROJECT_ROOT / 'pp_lite.py'), scenario]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  Pipeline FAILED for scenario {scenario}")
        print(f"  stderr: {result.stderr[-500:]}" if result.stderr else "")
        return False
    return True


def run_evaluation(scenario: str, method: str):
    """Run hota_trackeval.py and capture metrics."""
    # The output file depends on which method produced it
    rec_file = PROJECT_ROOT / f'REC_{scenario}.json'
    gt_file = PROJECT_ROOT / f'GT_{scenario}.json'

    if not rec_file.exists():
        print(f"  REC_{scenario}.json not found, skipping evaluation")
        return None

    cmd = [sys.executable, str(PROJECT_ROOT / 'hota_trackeval.py'), scenario]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  Evaluation FAILED for scenario {scenario}")
        return None

    # Parse output for metrics
    metrics = parse_trackeval_output(result.stdout)
    return metrics


def parse_trackeval_output(output: str) -> dict:
    """Parse hota_trackeval.py output to extract metric values."""
    metrics = {}
    lines = output.split('\n')
    in_results = False

    for line in lines:
        line = line.strip()
        if 'Results for' in line:
            in_results = True
            continue
        if in_results and line.startswith('='):
            in_results = False
            continue

        if in_results and line and not line.startswith('-'):
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                try:
                    val = float(parts[-1])
                    metrics[key] = val
                except ValueError:
                    pass

    return metrics if metrics else None


def run_benchmark(methods, scenarios, eval_only=False):
    """Run full benchmark across methods and scenarios."""
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print(f"BENCHMARK: {len(methods)} methods x {len(scenarios)} scenarios")
    print(f"Methods: {methods}")
    print(f"Scenarios: {scenarios}")
    print(f"Eval only: {eval_only}")
    print("=" * 70)

    for method in methods:
        results[method] = {}

        # Check model exists
        config = COST_CONFIGS[method]
        if 'checkpoint_path' in config:
            path = PROJECT_ROOT / config['checkpoint_path']
            if not path.exists():
                alt = config.get('model_path')
                if alt:
                    path = PROJECT_ROOT / alt
                if not path.exists():
                    print(f"\nSKIPPING {method}: model not found at {config.get('checkpoint_path', config.get('model_path'))}")
                    continue
        if 'model_path' in config:
            path = PROJECT_ROOT / config['model_path']
            if not path.exists():
                print(f"\nSKIPPING {method}: model not found at {config['model_path']}")
                continue

        for scenario in scenarios:
            print(f"\n--- {method} / scenario {scenario} ---")

            if not eval_only:
                # Update config
                update_parameters(method)
                update_scenario(scenario)

                # Run pipeline
                success = run_pipeline(scenario)
                if not success:
                    results[method][scenario] = {'error': 'pipeline failed'}
                    continue

            # Run evaluation
            metrics = run_evaluation(scenario, method)
            if metrics:
                results[method][scenario] = metrics
                print(f"  HOTA={metrics.get('HOTA', 'N/A'):.3f} "
                      f"MOTA={metrics.get('MOTA', 'N/A'):.3f} "
                      f"IDF1={metrics.get('IDF1', 'N/A'):.3f}")
            else:
                results[method][scenario] = {'error': 'evaluation failed'}

    # Print comparison table
    print_comparison_table(results, scenarios)

    # Save results
    output_dir = PROJECT_ROOT / 'experiments' / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'benchmark_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def print_comparison_table(results, scenarios):
    """Print a formatted comparison table."""
    metrics_to_show = ['HOTA', 'DetA', 'AssA', 'MOTA', 'MOTP', 'IDF1',
                       'Precision', 'Recall', 'Sw/GT', 'Fgmt/GT', 'No.']

    for scenario in scenarios:
        print(f"\n{'=' * 80}")
        print(f"  SCENARIO {scenario}")
        print(f"{'=' * 80}")

        # Header
        methods = [m for m in results if scenario in results[m] and 'error' not in results[m][scenario]]
        if not methods:
            print("  No results available")
            continue

        header = f"{'Metric':<12}" + "".join(f"{m:>12}" for m in methods)
        print(header)
        print("-" * len(header))

        for metric in metrics_to_show:
            row = f"{metric:<12}"
            for method in methods:
                val = results[method][scenario].get(metric, None)
                if val is not None:
                    if metric in ('Sw/GT', 'Fgmt/GT'):
                        row += f"{val:>12.2f}"
                    elif metric == 'No.':
                        # Try 'No. trajs' key
                        val = results[method][scenario].get('No.', results[method][scenario].get('No. trajs', None))
                        row += f"{val:>12.0f}" if val else f"{'N/A':>12}"
                    else:
                        row += f"{val:>12.3f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark all cost functions')
    parser.add_argument('--methods', nargs='+', default=list(COST_CONFIGS.keys()),
                        choices=list(COST_CONFIGS.keys()),
                        help='Methods to benchmark')
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS,
                        choices=SCENARIOS, help='Scenarios to evaluate')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip pipeline run, only evaluate existing outputs')
    args = parser.parse_args()

    run_benchmark(args.methods, args.scenarios, args.eval_only)
