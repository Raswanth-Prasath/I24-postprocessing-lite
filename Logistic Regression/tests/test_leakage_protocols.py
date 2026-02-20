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


def make_dataset_and_model(tmp_path, with_source_tags=True):
    rng = np.random.RandomState(123)
    n = 320

    x0 = rng.normal(0.0, 1.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)
    logits = 1.5 * x0 - 0.8 * x1 + 0.2 * x2
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    X = np.column_stack([x0, x1, x2, x3]).astype(np.float64)
    feature_names = np.array(["f0", "f1", "f2", "f3"])

    dataset_path = tmp_path / "dataset.npz"
    if with_source_tags:
        tags = np.array(
            ["advanced_keepall"] * (n // 2) + ["v4_diverse_curated"] * (n - n // 2),
            dtype=object,
        )
        np.savez_compressed(
            dataset_path,
            X=X,
            y=y,
            feature_names=feature_names,
            source_split_tag=tags,
        )
    else:
        np.savez_compressed(dataset_path, X=X, y=y, feature_names=feature_names)

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": ["f0", "f1", "f2", "f3"],
    }
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    return dataset_path, model_path


def test_lr_diagnostics_both_protocol_with_source_holdout(tmp_path):
    mod = load_diag_module()
    dataset_path, model_path = make_dataset_and_model(tmp_path, with_source_tags=True)

    out_dir = tmp_path / "diag_outputs"
    result = mod.run_diagnostics(
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        output_dir=str(out_dir),
        threshold=0.5,
        topk_influential=30,
        use_statsmodels=False,
        eval_protocol="both",
    )

    payload = json.loads(result["metrics_path"].read_text())
    assert payload["eval_protocol"] == "both"
    assert payload["protocol_status"]["random"]["status"] == "executed"
    assert payload["protocol_status"]["source_holdout"]["status"] == "executed"
    assert payload["metrics_random"] is not None
    assert payload["metrics_source_holdout"] is not None
    assert payload["leakage_gap"] is not None


def test_lr_diagnostics_source_holdout_fallback_without_tags(tmp_path):
    mod = load_diag_module()
    dataset_path, model_path = make_dataset_and_model(tmp_path, with_source_tags=False)

    out_dir = tmp_path / "diag_outputs_no_tags"
    result = mod.run_diagnostics(
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        output_dir=str(out_dir),
        threshold=0.5,
        topk_influential=30,
        use_statsmodels=False,
        eval_protocol="source_holdout",
    )

    payload = json.loads(result["metrics_path"].read_text())
    assert payload["eval_protocol"] == "source_holdout"
    assert payload["protocol_status"]["source_holdout"]["status"] == "skipped"
    assert payload["protocol_status"]["random"]["status"] == "executed"
    assert payload["metrics_oof"] is not None
    assert payload["leakage_gap"] is None

