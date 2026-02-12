# Logistic Regression Manifest

This file defines the canonical structure for the `Logistic Regression/` workspace and
tracks migration status for legacy files.

## Canonical Layout

- `src/`: source scripts for dataset prep, training, selection, tuning, diagnostics.
- `data/`: canonical dataset location for NPZ training data.
- `models/production/`: approved production LR artifacts.
- `models/candidates/`: non-production candidate model artifacts.
- `model_artifacts/`: legacy model location retained for backward compatibility.
- `outputs/`: generated plots and experiment outputs.
- `reports/`: generated markdown/txt reports.
- `notebooks/`: exploratory notebooks.
- `archive/`: legacy/retired files and migration notes.

## Production Invariants

- Primary production model: `Logistic Regression/model_artifacts/consensus_top10_full47.pkl`
- Corresponding canonical dataset (legacy path still supported):
  `Logistic Regression/training_dataset_advanced.npz`

## Current Script Categories

### Dataset Build/Diagnostics
- `rebuild_dataset.py`
- `enhanced_dataset_creation.py`
- `diagnostics_v4.py`
- `plot_timespace_v2.py`

### Feature Selection/Analysis
- `vif_analysis.py`
- `feature_evaluation.py`
- `feature_selection_full47.py`
- `unified_feature_selection.py`
- `advanced_features.py`

### Training/Tuning
- `train.py`
- `grid_search_lr.py`
- `optuna_search_lr.py`
- `cross_scenario_validation.py`

### Evaluation/Calibration
- `generate_evaluation_results.py`
- `analyze_cost_distributions.py`
- `calib_lr.py`

## Migration Status

- `notebooks/`: active (notebooks relocated here)
- `data/`: active for new tooling; legacy dataset paths remain valid
- `model_artifacts/`: legacy-compatible
- `.ipynb_checkpoints/`, `__pycache__/`, `.pytest_cache/`: deprecated generated state

## Compatibility Rule

During migration, scripts should resolve canonical paths first and fall back to legacy
paths when canonical files do not exist.

## Dataset Aliases

To preserve compatibility during migration:
- `training_dataset_combined.npz` is currently linked from `dataset file/training_dataset_combined.npz`.
- Canonical links are also exposed in `data/`.
