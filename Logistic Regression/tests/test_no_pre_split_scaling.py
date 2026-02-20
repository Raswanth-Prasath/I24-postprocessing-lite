import importlib.util
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification


MODULE_PATH = Path(__file__).resolve().parents[1] / "train.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_lr", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(n_samples=240, n_features=10, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        random_state=random_state,
    )
    return X.astype(np.float64), y.astype(np.int64)


def test_random_protocol_never_scales_full_dataset(monkeypatch):
    mod = load_train_module()
    X, y = make_dataset()
    n_total = len(y)
    fit_sizes = []
    original_fit = mod.StandardScaler.fit

    def wrapped_fit(self, X_fit, y_fit=None, *args, **kwargs):
        fit_sizes.append(len(X_fit))
        return original_fit(self, X_fit, y_fit, *args, **kwargs)

    monkeypatch.setattr(mod.StandardScaler, "fit", wrapped_fit)

    results = mod.evaluate_protocols(
        X_subset=X,
        y=y,
        C=1.0,
        eval_protocol='random',
        source_split_tag=None,
    )

    assert 'random' in results
    assert len(fit_sizes) > 0
    assert max(fit_sizes) < n_total


def test_source_holdout_protocol_never_scales_full_dataset(monkeypatch):
    mod = load_train_module()
    X, y = make_dataset(n_samples=300)
    n_total = len(y)
    tags = np.array(
        ['advanced_keepall'] * 150 + ['v4_diverse_curated'] * 150,
        dtype=object,
    )
    fit_sizes = []
    original_fit = mod.StandardScaler.fit

    def wrapped_fit(self, X_fit, y_fit=None, *args, **kwargs):
        fit_sizes.append(len(X_fit))
        return original_fit(self, X_fit, y_fit, *args, **kwargs)

    monkeypatch.setattr(mod.StandardScaler, "fit", wrapped_fit)

    results = mod.evaluate_protocols(
        X_subset=X,
        y=y,
        C=1.0,
        eval_protocol='source_holdout',
        source_split_tag=tags,
    )

    assert 'source_holdout' in results
    assert results['source_holdout']['train_tag'] == 'advanced_keepall'
    assert results['source_holdout']['test_tag'] == 'v4_diverse_curated'
    assert len(fit_sizes) > 0
    assert max(fit_sizes) < n_total

