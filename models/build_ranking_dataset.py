#!/usr/bin/env python
"""
Build anchor-candidate ranking dataset by replaying pipeline candidate generation.

Storage layout (compact):
- <output>.jsonl: per (anchor,candidate) rows with references only.
- <output>.fragments.jsonl: deduplicated fragment payloads keyed by fragment_ref.

This avoids row-wise duplication of full trajectory payloads.
"""

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.evaluate_transformer import (  # noqa: E402
    _extract_fragment_id,
    _get_pair_gt_label,
    collect_pipeline_pairs_for_calibration,
)
from models.rich_sequence_dataset import _get_gt_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ranking dataset from pipeline replay.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["i", "ii", "iii"],
        help="Scenarios to replay (default: i ii iii).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_ranking_dataset.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "outputs" / "transformer_ranking_dataset_manifest.json"),
        help="Output manifest JSON file path.",
    )
    parser.add_argument("--min-group-size", type=int, default=2, help="Minimum candidates per anchor.")
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=80,
        help="Maximum candidates per anchor retained in the replay dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split assignment.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio over known GT anchor IDs.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Optional test ratio over known GT anchor IDs.")
    parser.add_argument(
        "--min-valid-anchors",
        type=int,
        default=200,
        help="Abort if filtered anchors are below this threshold.",
    )
    parser.add_argument(
        "--min-mixed-gt-anchors",
        type=int,
        default=80,
        help="Abort if anchors with both positive and negative GT labels are below this threshold.",
    )
    parser.add_argument(
        "--subsample-large-groups",
        action="store_true",
        default=True,
        help="Subsample oversized groups instead of discarding them (default: True).",
    )
    parser.add_argument(
        "--no-subsample-large-groups",
        dest="subsample_large_groups",
        action="store_false",
        help="Discard oversized groups (legacy behavior).",
    )
    return parser.parse_args()


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value


def _new_anchor_meta(anchor_gt_id: Optional[str]) -> Dict[str, Any]:
    return {
        "count": 0,
        "has_any_gt": False,
        "has_pos": False,
        "has_neg": False,
        "anchor_gt_id": anchor_gt_id,
    }


def _update_anchor_meta(
    anchor_meta: Dict[str, Dict[str, Any]],
    anchor_key: str,
    gt_label: Optional[int],
    anchor_gt_id: Optional[str],
) -> None:
    meta = anchor_meta.get(anchor_key)
    if meta is None:
        meta = _new_anchor_meta(anchor_gt_id)
        anchor_meta[anchor_key] = meta

    meta["count"] = int(meta["count"]) + 1
    if meta.get("anchor_gt_id") is None and anchor_gt_id is not None:
        meta["anchor_gt_id"] = anchor_gt_id

    if gt_label is not None:
        meta["has_any_gt"] = True
        if int(gt_label) == 1:
            meta["has_pos"] = True
        elif int(gt_label) == 0:
            meta["has_neg"] = True


def _summarize_from_anchor_meta(anchor_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sizes = [int(v["count"]) for v in anchor_meta.values()]
    any_gt = sum(1 for v in anchor_meta.values() if bool(v.get("has_any_gt", False)))
    mixed_gt = sum(1 for v in anchor_meta.values() if bool(v.get("has_pos", False)) and bool(v.get("has_neg", False)))

    return {
        "num_anchors": int(len(anchor_meta)),
        "num_rows": int(sum(sizes)),
        "min_group_size": int(min(sizes)) if sizes else 0,
        "max_group_size": int(max(sizes)) if sizes else 0,
        "mean_group_size": float(np.mean(sizes)) if sizes else 0.0,
        "anchors_with_any_gt": int(any_gt),
        "anchors_with_pos_and_neg_gt": int(mixed_gt),
    }


def _assign_split_by_vehicle_gt(
    filtered_anchor_meta: Dict[str, Dict[str, Any]],
    all_vehicle_gt_ids: set,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple:
    """Split by ALL unique vehicle GT IDs (anchor + candidate roles) to prevent cross-role leakage."""
    vehicle_ids = sorted(all_vehicle_gt_ids)

    rng = np.random.default_rng(int(seed))
    if vehicle_ids:
        perm = rng.permutation(len(vehicle_ids))
        vehicle_ids = [vehicle_ids[i] for i in perm]

    n_total = len(vehicle_ids)
    n_test = int(max(0, math.floor(n_total * float(test_ratio))))
    n_val = int(max(0, math.floor(n_total * float(val_ratio))))
    n_test = min(n_test, n_total)
    n_val = min(n_val, max(0, n_total - n_test))

    test_ids = set(vehicle_ids[:n_test])
    val_ids = set(vehicle_ids[n_test:n_test + n_val])
    train_ids = set(vehicle_ids[n_test + n_val:])

    split_by_anchor_key: Dict[str, str] = {}
    for anchor_key, meta in filtered_anchor_meta.items():
        anchor_gt_id = meta.get("anchor_gt_id")
        if anchor_gt_id is None:
            split = "train"
        elif str(anchor_gt_id) in test_ids:
            split = "test"
        elif str(anchor_gt_id) in val_ids:
            split = "val"
        else:
            split = "train"
        split_by_anchor_key[anchor_key] = split

    return split_by_anchor_key, train_ids, val_ids, test_ids


def _subsample_group_rows(
    rows: List[Dict[str, Any]],
    max_size: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Subsample an oversized anchor group, prioritising positive-labeled rows."""
    if len(rows) <= max_size:
        return rows
    pos = [r for r in rows if r.get("gt_label") == 1]
    rest = [r for r in rows if r.get("gt_label") != 1]
    keep: List[Dict[str, Any]] = list(pos)
    if len(keep) >= max_size:
        idx = rng.choice(len(keep), size=max_size, replace=False)
        return [keep[int(i)] for i in idx]
    needed = max_size - len(keep)
    if len(rest) <= needed:
        keep.extend(rest)
    else:
        idx = rng.choice(len(rest), size=needed, replace=False)
        keep.extend(rest[int(i)] for i in idx)
    return keep


def _fragments_path_for_output(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.fragments.jsonl"


def main() -> None:
    args = parse_args()

    output_path = Path(args.output_path)
    manifest_path = Path(args.manifest_path)
    fragments_path = _fragments_path_for_output(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fragments_path.parent.mkdir(parents=True, exist_ok=True)

    per_scenario_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
    global_anchor_meta: Dict[str, Dict[str, Any]] = {}
    all_vehicle_gt_ids: set = set()  # union of anchor + candidate GT IDs for C3 fix

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="ranking_dataset_tmp_", suffix=".jsonl", dir=str(output_path.parent))
    tmp_file_path = Path(tmp_path)

    fragment_written = set()
    unique_fragment_count = 0

    try:
        with open(tmp_fd, "w") as tmp_f, open(fragments_path, "w") as frag_f:
            for scenario in args.scenarios:
                scenario = str(scenario)
                records = collect_pipeline_pairs_for_calibration(scenario)
                scenario_meta: Dict[str, Dict[str, Any]] = {}

                for idx, (track1, track2, bhat_total, gap) in enumerate(records):
                    if not np.isfinite(bhat_total) or float(bhat_total) >= 1e5:
                        continue

                    anchor_id = _extract_fragment_id(track2, f"anchor_{idx}")
                    candidate_id = _extract_fragment_id(track1, f"candidate_{idx}")
                    anchor_key = f"{scenario}:{anchor_id}"

                    # Fragment references are scenario-scoped IDs for dedup storage.
                    anchor_ref = f"{scenario}:{anchor_id}"
                    candidate_ref = f"{scenario}:{candidate_id}"

                    if candidate_ref not in fragment_written:
                        frag_f.write(json.dumps({
                            "fragment_ref": candidate_ref,
                            "fragment": _to_serializable(track1),
                        }) + "\n")
                        fragment_written.add(candidate_ref)
                        unique_fragment_count += 1

                    if anchor_ref not in fragment_written:
                        frag_f.write(json.dumps({
                            "fragment_ref": anchor_ref,
                            "fragment": _to_serializable(track2),
                        }) + "\n")
                        fragment_written.add(anchor_ref)
                        unique_fragment_count += 1

                    gt_label = _get_pair_gt_label(track1, track2)
                    anchor_gt_id = _get_gt_id(track2)
                    candidate_gt_id = _get_gt_id(track1)

                    if anchor_gt_id is not None:
                        all_vehicle_gt_ids.add(str(anchor_gt_id))
                    if candidate_gt_id is not None:
                        all_vehicle_gt_ids.add(str(candidate_gt_id))

                    row = {
                        "scenario": scenario,
                        "anchor_key": str(anchor_key),
                        "anchor_id": str(anchor_id),
                        "candidate_id": str(candidate_id),
                        "anchor_ref": str(anchor_ref),
                        "candidate_ref": str(candidate_ref),
                        "gap": float(gap),
                        "bhat_cost": float(bhat_total),
                        "gt_label": None if gt_label is None else int(gt_label),
                        "has_gt_pair_label": bool(gt_label is not None),
                        "anchor_gt_id": None if anchor_gt_id is None else str(anchor_gt_id),
                        "candidate_gt_id": None if candidate_gt_id is None else str(candidate_gt_id),
                        "direction": str(track2.get("direction", "unknown")),
                        "compute_node_id": str(track2.get("compute_node_id", "0")),
                    }
                    tmp_f.write(json.dumps(row) + "\n")

                    _update_anchor_meta(scenario_meta, anchor_key, gt_label, None if anchor_gt_id is None else str(anchor_gt_id))
                    _update_anchor_meta(global_anchor_meta, anchor_key, gt_label, None if anchor_gt_id is None else str(anchor_gt_id))

                per_scenario_meta[scenario] = scenario_meta

        raw_summary_by_scenario = {
            scenario: _summarize_from_anchor_meta(meta)
            for scenario, meta in per_scenario_meta.items()
        }
        pre_filter_summary = _summarize_from_anchor_meta(global_anchor_meta)

        filtered_anchor_meta: Dict[str, Dict[str, Any]] = {}
        for anchor_key, meta in global_anchor_meta.items():
            cnt = int(meta["count"])
            if cnt < int(args.min_group_size):
                continue
            if not args.subsample_large_groups and cnt > int(args.max_group_size):
                continue
            filtered_anchor_meta[anchor_key] = meta

        post_filter_summary = _summarize_from_anchor_meta(filtered_anchor_meta)

        if post_filter_summary["num_anchors"] < int(args.min_valid_anchors):
            raise RuntimeError(
                "Insufficient valid anchor groups after filtering: "
                f"{post_filter_summary['num_anchors']} < {args.min_valid_anchors}"
            )

        if post_filter_summary["anchors_with_pos_and_neg_gt"] < int(args.min_mixed_gt_anchors):
            raise RuntimeError(
                "Insufficient anchors with mixed GT labels after filtering: "
                f"{post_filter_summary['anchors_with_pos_and_neg_gt']} < {args.min_mixed_gt_anchors}"
            )

        split_by_anchor_key, train_vehicle_ids, val_vehicle_ids, test_vehicle_ids = _assign_split_by_vehicle_gt(
            filtered_anchor_meta,
            all_vehicle_gt_ids=all_vehicle_gt_ids,
            seed=int(args.seed),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
        )

        rows_written = 0
        split_counts = {"train": 0, "val": 0, "test": 0}
        anchor_split_counts = {"train": 0, "val": 0, "test": 0}
        filtered_anchor_keys = set(filtered_anchor_meta.keys())
        cross_partition_masked = 0
        subsample_rng = np.random.default_rng(int(args.seed) + 1)

        # Build vehicle ID sets per split for label masking
        split_vehicle_sets = {
            "train": train_vehicle_ids,
            "val": val_vehicle_ids,
            "test": test_vehicle_ids,
        }

        for anchor_key, split in split_by_anchor_key.items():
            anchor_split_counts[split] = anchor_split_counts.get(split, 0) + 1

        # Collect rows per anchor group for potential subsampling
        anchor_group_rows: Dict[str, List[Dict[str, Any]]] = {}
        with open(tmp_file_path, "r") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                anchor_key = str(row["anchor_key"])
                if anchor_key not in filtered_anchor_keys:
                    continue
                anchor_group_rows.setdefault(anchor_key, []).append(row)

        with open(output_path, "w") as out_f:
            for anchor_key, rows in sorted(anchor_group_rows.items()):
                split = split_by_anchor_key.get(anchor_key, "train")
                own_vehicles = split_vehicle_sets.get(split, set())

                # C4: subsample oversized groups instead of discarding
                if args.subsample_large_groups and len(rows) > int(args.max_group_size):
                    rows = _subsample_group_rows(rows, int(args.max_group_size), subsample_rng)

                for row in rows:
                    out_row = dict(row)
                    out_row["group_size"] = min(len(rows), int(args.max_group_size))
                    out_row["split"] = split

                    # C3: mask GT label if candidate's vehicle is in a different partition
                    cgt = row.get("candidate_gt_id")
                    if cgt is not None and out_row.get("gt_label") is not None:
                        if str(cgt) not in own_vehicles and str(cgt) in all_vehicle_gt_ids:
                            out_row["gt_label"] = None
                            out_row["has_gt_pair_label"] = False
                            cross_partition_masked += 1

                    out_f.write(json.dumps(out_row) + "\n")
                    rows_written += 1
                    split_counts[split] = split_counts.get(split, 0) + 1

        manifest = {
            "generated_at": str(np.datetime64("now")),
            "scenarios": [str(s) for s in args.scenarios],
            "config": {
                "min_group_size": int(args.min_group_size),
                "max_group_size": int(args.max_group_size),
                "subsample_large_groups": bool(args.subsample_large_groups),
                "seed": int(args.seed),
                "val_ratio": float(args.val_ratio),
                "test_ratio": float(args.test_ratio),
                "min_valid_anchors": int(args.min_valid_anchors),
                "min_mixed_gt_anchors": int(args.min_mixed_gt_anchors),
                "storage_mode": "deduplicated_fragments",
                "split_strategy": "vehicle_level",
            },
            "raw_summary_by_scenario": raw_summary_by_scenario,
            "pre_filter_summary": pre_filter_summary,
            "post_filter_summary": post_filter_summary,
            "anchor_split_counts": anchor_split_counts,
            "row_split_counts": split_counts,
            "rows_written": int(rows_written),
            "unique_fragments": int(unique_fragment_count),
            "vehicle_gt_ids_total": int(len(all_vehicle_gt_ids)),
            "vehicle_gt_ids_train": int(len(train_vehicle_ids)),
            "vehicle_gt_ids_val": int(len(val_vehicle_ids)),
            "vehicle_gt_ids_test": int(len(test_vehicle_ids)),
            "cross_partition_labels_masked": int(cross_partition_masked),
            "output_path": str(output_path),
            "fragments_path": str(fragments_path),
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Saved ranking dataset rows to: {output_path}")
        print(f"Saved fragment store to: {fragments_path}")
        print(f"Saved ranking dataset manifest to: {manifest_path}")
        print(f"Filtered anchors: {post_filter_summary['num_anchors']}")
        print(f"Anchors with mixed GT: {post_filter_summary['anchors_with_pos_and_neg_gt']}")
        print(f"Rows written: {rows_written}")
        print(f"Unique fragments: {unique_fragment_count}")
        print(f"Vehicle GT IDs: {len(all_vehicle_gt_ids)} total, {len(train_vehicle_ids)} train, {len(val_vehicle_ids)} val")
        print(f"Cross-partition GT labels masked: {cross_partition_masked}")

    finally:
        try:
            if tmp_file_path.exists():
                tmp_file_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
