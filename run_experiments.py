#!/usr/bin/env python
"""
Batch experiment runner for I24 postprocessing pipeline.

Examples:
    # Single run
    python run_experiments.py --config parameters_LR.json --suffix i

    # One model, all scenarios
    python run_experiments.py --config parameters_LR.json --all-suffixes

    # All models, one scenario
    python run_experiments.py --all-configs --suffix i

    # Full matrix (all models x all scenarios)
    python run_experiments.py --all-configs --all-suffixes

    # Preview without running
    python run_experiments.py --all-configs --all-suffixes --dry-run

    # Run + evaluate each generated tagged output
    python run_experiments.py --all-configs --suffix i --evaluate
"""

import argparse
import glob
import os
import subprocess
import sys
import time


SUFFIXES = ["i", "ii", "iii"]

# Map config filename -> method tag
CONFIG_TAG_MAP = {
    "parameters_Bhat.json": "Bhat",
    "parameters_LR.json": "LR",
    "parameters_SNN.json": "SNN",
    "parameters_MLP.json": "MLP",
    "parameters_TCN.json": "TCN",
    "parameters_Transformer.json": "Transformer",
}


def discover_configs():
    """Find all parameters_*.json files and map to tags."""
    configs = {}
    for path in sorted(glob.glob("parameters_*.json")):
        basename = os.path.basename(path)
        if basename in CONFIG_TAG_MAP:
            configs[path] = CONFIG_TAG_MAP[basename]
        else:
            # Infer tag from filename: parameters_Foo.json -> Foo
            tag = basename.replace("parameters_", "").replace(".json", "")
            configs[path] = tag
    return configs


def run_pipeline(config_path, suffix, tag, dry_run=False):
    """Run pp_lite.py with the given config and tag."""
    cmd = [sys.executable, "pp_lite.py", suffix, "--config", config_path, "--tag", tag]
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_evaluation_suffix(suffix, dry_run=False):
    """Run hota_trackeval.py in legacy per-suffix mode."""
    cmd = [sys.executable, "hota_trackeval.py", suffix]
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    print(f"  Evaluating: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_evaluation_file(gt_file, tracker_file, name, dry_run=False):
    """Run hota_trackeval.py for an explicit GT/tracker pair."""
    cmd = [
        sys.executable,
        "hota_trackeval.py",
        "--gt-file",
        gt_file,
        "--tracker-file",
        tracker_file,
        "--name",
        name,
    ]
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    print(f"  Evaluating: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def expected_tracker_candidates(suffix, tag):
    """Return candidate tracker output filenames for a (suffix, tag) run."""
    candidates = [f"REC_{suffix}_{tag}.json"]

    # Compatibility map for older naming conventions.
    tag_upper = tag.upper()
    compat = {
        "LR": [f"REC_{suffix}_LR.json"],
        "BHAT": [f"REC_{suffix}_Bhat.json", f"REC_{suffix}.json"],
        "SNN": [f"REC_{suffix}_SNN.json"],
    }
    for path in compat.get(tag_upper, []):
        if path not in candidates:
            candidates.append(path)

    return candidates


def resolve_tracker_file(candidates, dry_run=False):
    """Resolve first existing tracker file from candidates."""
    if dry_run:
        return candidates[0]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for I24 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=None,
                        help="Path to a single config file (e.g. parameters_LR.json)")
    parser.add_argument("--suffix", default=None, choices=SUFFIXES,
                        help="Single scenario suffix: i, ii, or iii")
    parser.add_argument("--all-configs", action="store_true",
                        help="Run all discovered parameters_*.json configs")
    parser.add_argument("--all-suffixes", action="store_true",
                        help="Run all scenarios (i, ii, iii)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run HOTA evaluation after pipeline runs")
    parser.add_argument(
        "--evaluate-granularity",
        choices=["per-run", "per-suffix"],
        default="per-run",
        help="Evaluation mode when --evaluate is set (default: per-run)",
    )
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop immediately if any run fails")
    args = parser.parse_args()

    # Determine configs
    if args.all_configs:
        configs = discover_configs()
        if not configs:
            print("No parameters_*.json files found.")
            sys.exit(1)
    elif args.config:
        basename = os.path.basename(args.config)
        if basename in CONFIG_TAG_MAP:
            tag = CONFIG_TAG_MAP[basename]
        else:
            tag = basename.replace("parameters_", "").replace(".json", "")
        configs = {args.config: tag}
    else:
        parser.error("Specify --config or --all-configs")

    # Determine suffixes
    if args.all_suffixes:
        suffixes = SUFFIXES
    elif args.suffix:
        suffixes = [args.suffix]
    else:
        parser.error("Specify --suffix or --all-suffixes")

    # Build run matrix
    runs = [(cfg, suffix, tag) for cfg, tag in configs.items() for suffix in suffixes]
    total = len(runs)
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Experiment matrix: {len(configs)} config(s) x {len(suffixes)} scenario(s) = {total} run(s)\n")

    # Execute runs
    failed = []
    successful_runs = []
    t_start = time.time()

    for idx, (cfg, suffix, tag) in enumerate(runs, 1):
        print(f"[{idx}/{total}] {tag} on scenario {suffix}  (config: {cfg})")
        ok = run_pipeline(cfg, suffix, tag, dry_run=args.dry_run)
        if not ok:
            failed.append((cfg, suffix, tag))
            print("  FAILED!")
            if args.stop_on_error:
                print("Stopping due to --stop-on-error")
                break
            continue

        candidates = expected_tracker_candidates(suffix, tag)
        tracker_file = resolve_tracker_file(candidates, dry_run=args.dry_run)
        if tracker_file is None:
            print(f"  WARNING: no tracker output found for {tag}/{suffix}. Candidates: {candidates}")
        successful_runs.append({
            "config": cfg,
            "suffix": suffix,
            "tag": tag,
            "tracker_file": tracker_file,
        })

    # Evaluation
    if args.evaluate:
        if args.evaluate_granularity == "per-suffix":
            eval_suffixes = sorted(set(r["suffix"] for r in successful_runs))
            print(f"\nRunning per-suffix evaluation for scenario(s): {', '.join(eval_suffixes)}")
            for suffix in eval_suffixes:
                run_evaluation_suffix(suffix, dry_run=args.dry_run)
        else:
            print("\nRunning per-run evaluation for generated tagged outputs")
            for run in successful_runs:
                suffix = run["suffix"]
                tag = run["tag"]
                tracker_file = run["tracker_file"]
                if tracker_file is None:
                    print(f"  Skipping {tag}/{suffix}: tracker file unresolved")
                    continue

                gt_file = f"GT_{suffix}.json"
                name = f"{tag}_{suffix}"
                run_evaluation_file(gt_file, tracker_file, name, dry_run=args.dry_run)

    # Summary
    elapsed = time.time() - t_start
    print(f"\nCompleted {total - len(failed)}/{total} runs in {elapsed:.1f}s")
    if failed:
        print("Failed runs:")
        for cfg, suffix, tag in failed:
            print(f"  - {tag} on scenario {suffix} ({cfg})")
        sys.exit(1)


if __name__ == "__main__":
    main()
