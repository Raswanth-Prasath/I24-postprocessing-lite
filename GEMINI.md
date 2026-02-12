# GEMINI.md - I24-Postprocessing-Lite Context

This document provides essential context and instructions for the I24-postprocessing-lite project, a streamlined pipeline for vehicle trajectory reconstruction.

## Project Overview

**Purpose:** Reconstruct high-fidelity vehicle trajectories from fragmented and noisy raw tracking data sourced from the I-24 MOTION testbed.
**Main Technologies:**
- **Language:** Python
- **Data & ML:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `pytorch` (for Siamese Networks)
- **Optimization:** `cvxpy`, `cvxopt`
- **Graph Theory:** `networkx`
- **Database:** MongoDB (via internal I24 APIs)
- **Testing & Quality:** `pytest`, `pre-commit`

### Architecture & Core Algorithms
The pipeline consists of three primary stages:
1.  **Fragment Merging (`merge.py`):** Identifies overlapping fragments that belong to the same vehicle and merges them into connected components.
2.  **Fragment Stitching (`min_cost_flow.py`):** Connects non-overlapping fragments using an online min-cost circulation algorithm. Supports multiple cost functions:
    - **Bhattacharyya:** Geometric and appearance-based baseline.
    - **Siamese Network:** Deep learning-based similarity.
    - **Logistic Regression:** Feature-engineered ML model with calibrated scaling.
3.  **Trajectory Rectification (`reconciliation.py`):** Uses convex optimization to smooth trajectories, impute missing data, and remove outliers.

---

## Building and Running

### Environment Setup
The project uses a Conda environment typically named `i24`. 
```bash
source activate i24
# Install core I24 dependencies from GitHub as specified in installation.txt
```

### Core Commands
- **Run Pipeline:**
  ```bash
  python pp_lite.py --scenario {i,ii,iii}
  ```
  Options include `--cost-function {bhattacharyya,siamese,lr}`, `--scale-factor`, and `--config`.

- **Evaluate Results (MOT Metrics):**
  ```bash
  python mot_i24.py {i,ii,iii}
  ```

- **Run Benchmarks:**
  ```bash
  python benchmark_all_scenarios.py
  ```

- **Run Tests:**
  ```bash
  pytest .
  # Or specific ML tests:
  cd "Logistic Regression/tests" && pytest .
  ```

---

## Development Conventions

### Feature Consistency
**CRITICAL:** All feature extraction for fragment stitching MUST use the centralized `StitchFeatureExtractor` in `utils/features_stitch.py`. This ensures perfect consistency between model training and real-time inference.

### Configuration
System-wide settings are managed via `parameters.json`. Scenarios (i: free-flow, ii: snowy/slow, iii: congested) can be overridden via CLI.

### Testing & Validation
- **Unit Tests:** Located in `tests/` directories or alongside source code.
- **Cross-Scenario Validation:** Use `Logistic Regression/cross_scenario_validation.py` to ensure ML models generalize across different traffic conditions.
- **VIF Analysis:** Use `Logistic Regression/vif_analysis.py` to monitor and resolve multicollinearity in new features.

### Git Workflow
- Adhere to `.pre-commit-config.yaml` standards.
- Documentation updates should be reflected in `IMPLEMENTATION_SUMMARY.md` for major changes.

---

## Key Directory Structure
- `/`: Main pipeline entry and benchmarking.
- `/utils/`: Core utilities, feature extractors, and cost function interfaces.
- `/Logistic Regression/`: Training, feature selection, and analysis for the LR cost model.
- `/Siamese-Network/`: Siamese network implementation and artifacts.
- `/hota/`: Tracking evaluation metrics.
- `/models/` & `/outputs/`: Storage for serialized models and pipeline results.
