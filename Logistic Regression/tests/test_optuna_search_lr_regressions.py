import csv
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "optuna_search_lr.py"
)


def load_optuna_module():
    spec = importlib.util.spec_from_file_location("optuna_search_lr", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keep_best_legacy_missing_use_logit():
    mod = load_optuna_module()

    legacy_params = {
        "model_path": "Logistic Regression/model_artifacts/consensus_10features_full47.pkl",
        "scale_factor": 4.2,
        "stitch_thresh": 2.3,
        "time_penalty": 0.12,
        "master_offset": 0.8,
        # Intentionally no use_logit/logit_offset
    }

    resolved = mod.resolve_keep_best_params(legacy_params)

    assert resolved["model_path"] == legacy_params["model_path"]
    assert resolved["use_logit"] is False
    assert resolved["logit_offset"] == 5.0
    assert resolved["scale_factor"] == legacy_params["scale_factor"]
    assert resolved["time_penalty"] == legacy_params["time_penalty"]
    assert resolved["stitch_thresh"] == legacy_params["stitch_thresh"]
    assert resolved["master_offset"] == legacy_params["master_offset"]


def test_csv_schema_mismatch_uses_versioned_file(tmp_path):
    mod = load_optuna_module()

    legacy_header = [
        "trial",
        "C",
        "penalty",
        "solver",
        "model_path",
        "cv_roc_auc",
        "test_roc_auc",
        # Old schema omitted use_logit/logit_offset
        "scale_factor",
        "stitch_thresh",
        "master_stitch_thresh",
        "time_penalty",
        "scenario",
        "composite_score",
        "status",
    ]

    output_csv = tmp_path / "optuna_search_results.csv"
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(legacy_header)
        writer.writerow([1] * len(legacy_header))

    resolved_csv, write_header, legacy_csv = mod.resolve_output_csv_path(
        output_csv, mod.CSV_HEADER
    )

    assert resolved_csv != output_csv
    assert resolved_csv.name.startswith("optuna_search_results.schema_v")
    assert write_header is True
    assert legacy_csv == output_csv


def test_csv_schema_match_appends_in_place(tmp_path):
    mod = load_optuna_module()

    output_csv = tmp_path / "optuna_search_results.csv"
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(mod.CSV_HEADER)
        writer.writerow(["x"] * len(mod.CSV_HEADER))

    resolved_csv, write_header, legacy_csv = mod.resolve_output_csv_path(
        output_csv, mod.CSV_HEADER
    )

    assert resolved_csv == output_csv
    assert write_header is False
    assert legacy_csv is None
