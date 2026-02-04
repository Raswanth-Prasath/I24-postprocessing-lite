"""
Calibration sweep for logistic regression cost function in the stitching pipeline.

Runs pp_lite.py -> diagnose_json.py -> mot_i24.py for a grid of
scale_factor and stitch_thresh values, logging results to CSV.

Usage (recommended):
  conda run -n i24 python "Logistic Regression/calib_lr.py" --scenario i
"""

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

METRIC_HEADERS = [
    "Prcn",
    "Rcll",
    "MOTA",
    "MOTP",
    "GT",
    "MT",
    "PT",
    "ML",
    "FP",
    "FN",
    "IDsw",
    "ObjIDsw",
    "ObjFM",
    "NoTrajs",
    "FgmtGT",
    "SwGT",
]


def run_cmd(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


def parse_rec_line(mot_stdout, scenario):
    target = f"REC_{scenario}"
    rec_line = None
    for line in mot_stdout.splitlines():
        if line.strip().startswith(target):
            rec_line = line.strip()
            break
    if not rec_line:
        return None, None

    parts = rec_line.split()
    nums = parts[1:]
    if len(nums) < len(METRIC_HEADERS):
        return rec_line, None
    return rec_line, nums[: len(METRIC_HEADERS)]


def parse_float_list(value):
    items = []
    for item in value.split(","):
        item = item.strip()
        if item:
            items.append(float(item))
    return items


def update_thresholds(cfg, stitch, master):
    cfg["stitch_thresh"] = stitch
    cfg["master_stitch_thresh"] = master

    stitcher_args = cfg.get("stitcher_args")
    if isinstance(stitcher_args, dict):
        stitcher_args["stitch_thresh"] = stitch
        stitcher_args["master_stitch_thresh"] = master


def main():
    parser = argparse.ArgumentParser(description="Calibration sweep for LR cost function")
    parser.add_argument("--scenario", default="i", help="Scenario to run (default: i)")
    parser.add_argument("--scale-factors", default="5,7,10", help="Comma-separated scale factors")
    parser.add_argument("--stitch-thresholds", default="0.7,0.8", help="Comma-separated stitch thresholds")
    parser.add_argument("--master-offset", type=float, default=0.2, help="master_stitch_thresh = stitch + offset")
    parser.add_argument(
        "--output",
        default="Logistic Regression/feature_selection_outputs/calibration_sweep_gap15.csv",
        help="Output CSV path (relative to repo root)",
    )
    parser.add_argument(
        "--keep-best",
        action="store_true",
        help="Keep best config (by MOTA) in parameters.json; otherwise restore original",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    param_path = root / "parameters.json"
    output_csv = root / args.output

    base = json.loads(param_path.read_text())
    scenario = args.scenario

    scale_factors = parse_float_list(args.scale_factors)
    stitch_thresholds = parse_float_list(args.stitch_thresholds)
    if not scale_factors or not stitch_thresholds:
        raise SystemExit("No scale factors or stitch thresholds provided.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    best = {"mota": float("-inf"), "cfg": None, "rec_line": None}
    mota_idx = METRIC_HEADERS.index("MOTA")

    print("Starting sweep...")
    try:
        with output_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["scale_factor", "stitch_thresh", "master_stitch_thresh"]
                + METRIC_HEADERS
                + ["raw_line"]
            )

            for scale in scale_factors:
                for stitch in stitch_thresholds:
                    cfg = deepcopy(base)
                    cfg["cost_function"]["scale_factor"] = scale
                    master = stitch + args.master_offset
                    update_thresholds(cfg, stitch, master)
                    param_path.write_text(json.dumps(cfg, indent=4))

                    print(
                        f"\n=== Running: scale_factor={scale}, stitch_thresh={stitch} ===",
                        flush=True,
                    )

                    rec_line = None
                    metrics = None
                    try:
                        run_cmd([sys.executable, "pp_lite.py", scenario], cwd=root)
                        run_cmd(
                            [sys.executable, "diagnose_json.py", f"REC_{scenario}.json", "--fix"],
                            cwd=root,
                        )
                        mot = run_cmd([sys.executable, "mot_i24.py", scenario], cwd=root)
                        rec_line, metrics = parse_rec_line(mot.stdout, scenario)
                    except subprocess.CalledProcessError as exc:
                        print("Command failed:", exc, file=sys.stderr)
                        if exc.stdout:
                            print(exc.stdout, file=sys.stderr)
                        if exc.stderr:
                            print(exc.stderr, file=sys.stderr)
                        rec_line = f"ERROR: {exc}"

                    if metrics:
                        try:
                            mota = float(metrics[mota_idx])
                        except ValueError:
                            mota = float("-inf")
                        if mota > best["mota"]:
                            best = {"mota": mota, "cfg": deepcopy(cfg), "rec_line": rec_line}

                    row_metrics = metrics if metrics else [""] * len(METRIC_HEADERS)
                    writer.writerow([scale, stitch, master] + row_metrics + [rec_line or ""])
                    f.flush()
    finally:
        if args.keep_best and best["cfg"] is not None:
            param_path.write_text(json.dumps(best["cfg"], indent=4))
            print(f"\nBest MOTA: {best['mota']} ({best['rec_line']})")
        else:
            param_path.write_text(json.dumps(base, indent=4))
            if args.keep_best:
                print("\nNo successful runs; restored original parameters.json.")
            else:
                print("\nRestored original parameters.json.")


if __name__ == "__main__":
    main()
