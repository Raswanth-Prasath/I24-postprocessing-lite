import csv
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


MODULE_PATH = Path(__file__).resolve().parents[1] / "lr_diagnostics.py"


def load_diag_module():
    spec = importlib.util.spec_from_file_location("lr_diagnostics", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_synthetic_dataset_and_model(tmp_path):
    rng = np.random.RandomState(42)

    n = 300
    x0 = rng.normal(0.0, 1.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)

    logits = 2.0 * x0 - 1.0 * x1 + 0.3 * x2
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)

    X = np.column_stack([x0, x1, x2, x3]).astype(np.float64)
    feature_names = np.array(["f0", "f1", "f2", "f3"])

    dataset_path = tmp_path / "toy_dataset.npz"
    np.savez_compressed(dataset_path, X=X, y=y, feature_names=feature_names)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = LogisticRegression(max_iter=500).fit(X_scaled, y)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": ["f0", "f1", "f2", "f3"],
    }
    model_path = tmp_path / "toy_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    return dataset_path, model_path


def test_lr_diagnostics_outputs_created_and_parseable(tmp_path):
    mod = load_diag_module()
    dataset_path, model_path = make_synthetic_dataset_and_model(tmp_path)

    out_dir = tmp_path / "diag_outputs"
    result = mod.run_diagnostics(
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        output_dir=str(out_dir),
        threshold=0.5,
        topk_influential=50,
        use_statsmodels=False,
    )

    assert result["metrics_path"].exists()
    assert result["influence_path"].exists()
    assert result["summary_path"].exists()

    metrics_payload = json.loads(result["metrics_path"].read_text())
    assert "metrics_oof" in metrics_payload
    assert "roc_auc" in metrics_payload["metrics_oof"]
    assert "ece" in metrics_payload["metrics_oof"]

    with result["influence_path"].open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    required_cols = {
        "sample_index",
        "cook_distance",
        "leverage",
        "std_residual",
        "cook_flag",
        "leverage_flag",
        "residual_flag",
        "influential",
    }
    assert required_cols.issubset(set(rows[0].keys()))


def test_lr_diagnostics_manual_fallback_method(tmp_path):
    mod = load_diag_module()
    dataset_path, model_path = make_synthetic_dataset_and_model(tmp_path)

    out_dir = tmp_path / "diag_outputs_manual"
    result = mod.run_diagnostics(
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        output_dir=str(out_dir),
        threshold=0.5,
        topk_influential=30,
        use_statsmodels=False,
    )

    payload = json.loads(result["metrics_path"].read_text())
    assert payload["influence_method"] == "manual"
