#!/usr/bin/env python
"""

Threshold/scale sweep for Transformer cost function on scenario i.

Objective: maximize (HOTA + IDF1) / 2

Sweeps over (scale_factor, stitch_thresh) grid, runs pp_lite + hota_trackeval
for each point, and reports a ranked table of results.

Usage:
    conda activate i24
    python sweep_transformer.py
    python sweep_transformer.py --dry-run          # preview grid only
    python sweep_transformer.py --quick             # coarse 3x3 grid
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time


SUFFIX = "i"
BASE_CONFIG = "parameters_Transformer.json"
GT_FILE = f"GT_{SUFFIX}.json"
SWEEP_TAG_PREFIX = "T_sweep"


# --- Grid -------------------------------------------------------------------

GRID_SCALE_FACTOR = [3.0, 5.0, 7.0, 10.0, 15.0]
GRID_STITCH_THRESH = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
GRID_TIME_PENALTY = [0.1]  # hold constant for now

QUICK_SCALE_FACTOR = [5.0, 10.0, 15.0]
QUICK_STITCH_THRESH = [1.5, 3.0, 5.0]


def load_config(path):
    with open(path) as f:
        return json.load(f)


def write_config(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")


def load_calibration_diagnostics(base_cfg):
    """Load calibration artifact diagnostics (spearman/mapping) if available."""
    cost_cfg = base_cfg.get("cost_function", {})
    mode = str(cost_cfg.get("calibration_mode", "linear")).lower()
    path = cost_cfg.get("calibration_path", "models/outputs/transformer_calibration.json")
    if not path or not os.path.exists(path):
        return {
            "calibration_mode": mode,
            "calibration_path": path,
            "spearman": None,
            "target_mapping": None,
        }

    try:
        with open(path) as f:
            artifact = json.load(f)
        return {
            "calibration_mode": mode,
            "calibration_path": path,
            "spearman": artifact.get("fit_metrics", {}).get("spearman", None),
            "target_mapping": artifact.get("fit_params", {}).get("target_mapping", None),
        }
    except Exception:
        return {
            "calibration_mode": mode,
            "calibration_path": path,
            "spearman": None,
            "target_mapping": None,
        }


def run_pipeline(config_path, suffix, tag):
    """Run pp_lite.py and return True on success."""
    cmd = [sys.executable, "pp_lite.py", suffix, "--config", config_path, "--tag", tag]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    pp_lite FAILED: {result.stderr[-300:]}")
    return result.returncode == 0


def run_eval(gt_file, tracker_file, name):
    """Run hota_trackeval.py and parse metrics from stdout."""
    cmd = [
        sys.executable, "hota_trackeval.py",
        "--gt-file", gt_file,
        "--tracker-file", tracker_file,
        "--name", name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    hota_trackeval FAILED: {result.stderr[-300:]}")
        return None

    return parse_metrics(result.stdout)


def parse_metrics(stdout):
    """Extract metric values from hota_trackeval.py printed table."""
    metrics = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Lines look like:  "HOTA             0.339"
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                val = float(parts[-1])
                metrics[key] = val
            except ValueError:
                # Handle "Sw/GT", "Fgmt/GT", "No." etc.
                if "/" in parts[0] and len(parts) >= 2:
                    combined_key = parts[0]
                    try:
                        val = float(parts[-1])
                        metrics[combined_key] = val
                    except ValueError:
                        pass
                elif parts[0] == "No." and len(parts) >= 3:
                    try:
                        metrics["No. trajs"] = float(parts[-1])
                    except ValueError:
                        pass
    return metrics if metrics else None


def objective(metrics):
    """(HOTA + IDF1) / 2"""
    hota = metrics.get("HOTA", 0)
    idf1 = metrics.get("IDF1", 0)
    return (hota + idf1) / 2.0


def main():
    parser = argparse.ArgumentParser(description="Transformer threshold/scale sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print grid without running")
    parser.add_argument("--quick", action="store_true", help="Use coarse 3x3 grid")
    parser.add_argument("--suffix", default=SUFFIX, choices=["i", "ii", "iii"])
    args = parser.parse_args()

    suffix = args.suffix
    gt_file = f"GT_{suffix}.json"

    scale_factors = QUICK_SCALE_FACTOR if args.quick else GRID_SCALE_FACTOR
    stitch_thresholds = QUICK_STITCH_THRESH if args.quick else GRID_STITCH_THRESH
    time_penalties = GRID_TIME_PENALTY

    # Build grid
    grid = []
    for sf in scale_factors:
        for st in stitch_thresholds:
            for tp in time_penalties:
                grid.append((sf, st, tp))

    print(f"Sweep: {len(grid)} points  (scale_factor x stitch_thresh x time_penalty)")
    print(f"  scale_factor:   {scale_factors}")
    print(f"  stitch_thresh:  {stitch_thresholds}")
    print(f"  time_penalty:   {time_penalties}")
    print(f"  suffix:         {suffix}")
    print(f"  objective:      (HOTA + IDF1) / 2")
    print()

    # Load base config and calibration diagnostics up-front (also shown in dry-run).
    base_cfg = load_config(BASE_CONFIG)
    calib_diag = load_calibration_diagnostics(base_cfg)
    print("Calibration diagnostics:")
    print(f"  mode:          {calib_diag['calibration_mode']}")
    print(f"  artifact:      {calib_diag['calibration_path']}")
    print(f"  spearman:      {calib_diag['spearman']}")
    print(f"  target_mapping:{calib_diag['target_mapping']}")
    print()

    if args.dry_run:
        for i, (sf, st, tp) in enumerate(grid, 1):
            print(f"  [{i:2d}] sf={sf:5.1f}  st={st:4.1f}  tp={tp:.2f}")
        print(f"\nTotal: {len(grid)} runs. Use without --dry-run to execute.")
        return

    tmp_config = f"_sweep_transformer_{suffix}.json"

    results = []
    t_start = time.time()

    for i, (sf, st, tp) in enumerate(grid, 1):
        tag = f"{SWEEP_TAG_PREFIX}_sf{sf:.0f}_st{st*10:.0f}_tp{tp*100:.0f}"
        tracker_file = f"REC_{suffix}_{tag}.json"

        print(f"[{i:2d}/{len(grid)}] sf={sf:5.1f}  st={st:4.1f}  tp={tp:.2f}  tag={tag}")

        # Patch config
        cfg = copy.deepcopy(base_cfg)
        cfg["cost_function"]["scale_factor"] = sf
        cfg["cost_function"]["time_penalty"] = tp
        cfg["stitcher_args"]["stitch_thresh"] = st
        cfg["stitcher_args"]["master_stitch_thresh"] = st + 1.0
        # Point to correct scenario
        cfg["gt_collection"] = f"GT_{suffix}"
        cfg["raw_collection"] = f"RAW_{suffix}"
        cfg["reconciled_collection"] = f"REC_{suffix}"
        write_config(cfg, tmp_config)

        # Run pipeline
        ok = run_pipeline(tmp_config, suffix, tag)
        if not ok:
            results.append({
                "sf": sf, "st": st, "tp": tp, "tag": tag,
                "status": "FAIL", "obj": -1,
            })
            continue

        # Check output exists
        if not os.path.exists(tracker_file):
            print(f"    Output {tracker_file} not found!")
            results.append({
                "sf": sf, "st": st, "tp": tp, "tag": tag,
                "status": "NO_OUTPUT", "obj": -1,
            })
            continue

        # Evaluate
        metrics = run_eval(gt_file, tracker_file, tag)
        if metrics is None:
            results.append({
                "sf": sf, "st": st, "tp": tp, "tag": tag,
                "status": "EVAL_FAIL", "obj": -1,
            })
            continue

        obj = objective(metrics)
        entry = {
            "sf": sf, "st": st, "tp": tp, "tag": tag,
            "status": "OK", "obj": obj,
        }
        entry.update(metrics)
        results.append(entry)
        print(f"    HOTA={metrics.get('HOTA', 0):.3f}  IDF1={metrics.get('IDF1', 0):.3f}  "
              f"obj={obj:.3f}  Sw/GT={metrics.get('Sw/GT', -1):.2f}  Fgmt/GT={metrics.get('Fgmt/GT', -1):.2f}")

    elapsed = time.time() - t_start

    # Clean up temp config
    if os.path.exists(tmp_config):
        os.remove(tmp_config)

    # --- Report ----------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"  SWEEP RESULTS  ({len(results)} points, {elapsed:.0f}s)")
    print(f"{'=' * 100}")

    # Sort by objective descending
    ok_results = [r for r in results if r["status"] == "OK"]
    ok_results.sort(key=lambda r: r["obj"], reverse=True)

    header = f"{'Rank':>4}  {'sf':>5}  {'st':>5}  {'tp':>5}  {'OBJ':>6}  {'HOTA':>6}  {'IDF1':>6}  {'MOTA':>6}  {'Prec':>6}  {'Recall':>6}  {'Sw/GT':>6}  {'Fgmt/GT':>7}  {'FP':>6}  {'trajs':>5}"
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(ok_results, 1):
        print(
            f"{rank:4d}  {r['sf']:5.1f}  {r['st']:5.1f}  {r['tp']:5.2f}  "
            f"{r['obj']:6.3f}  {r.get('HOTA', 0):6.3f}  {r.get('IDF1', 0):6.3f}  "
            f"{r.get('MOTA', 0):6.3f}  {r.get('Precision', 0):6.3f}  {r.get('Recall', 0):6.3f}  "
            f"{r.get('Sw/GT', -1):6.2f}  {r.get('Fgmt/GT', -1):7.2f}  "
            f"{r.get('FP', 0):6.0f}  {r.get('No. trajs', 0):5.0f}"
        )

    # Sanity check flagging
    if ok_results:
        best = ok_results[0]
        print(f"\nBest: sf={best['sf']}, st={best['st']}, tp={best['tp']}  "
              f"-> obj={best['obj']:.3f} (HOTA={best.get('HOTA', 0):.3f}, IDF1={best.get('IDF1', 0):.3f})")
        sw = best.get("Sw/GT", 99)
        fgmt = best.get("Fgmt/GT", 99)
        if sw > 1.0:
            print(f"  WARNING: Sw/GT={sw:.2f} > 1.0 — excessive ID switches")
        if fgmt > 2.0:
            print(f"  WARNING: Fgmt/GT={fgmt:.2f} > 2.0 — excessive fragmentation")

    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        print(f"\nFailed points ({len(failed)}):")
        for r in failed:
            print(f"  sf={r['sf']}, st={r['st']}: {r['status']}")

    # Save to JSON for later analysis
    out_file = f"sweep_results_{suffix}.json"
    with open(out_file, "w") as f:
        json.dump({"grid": results, "best": ok_results[0] if ok_results else None}, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
