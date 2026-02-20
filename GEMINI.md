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
    - **Siamese Network:** Deep learning-based similarity (BiLSTM encoder).
    - **Logistic Regression:** Feature-engineered ML model with calibrated scaling (Production model: `consensus_top10_full47.pkl`).
3.  **Trajectory Rectification (`reconciliation.py`):** Uses convex optimization to smooth trajectories, impute missing data, and remove outliers.

---

## Latest Results (Feb 4, 2026)

### Pipeline Performance (Bhattacharyya vs Logistic Regression)

| Scenario | Method | MOTA | MOTP | Precision | Recall | FP | Fgmt/GT | Sw/GT |
|----------|--------|------|------|-----------|--------|------|---------|-------|
| **i** | Bhattacharyya | **0.799** | 0.644 | **0.96** | 0.84 | **3939** | 1.46 | 0.49 |
| | Logistic Regression | 0.797 | 0.644 | 0.96 | 0.83 | 10534 | 1.52 | 0.54 |
| **ii** | Bhattacharyya | **0.563** | 0.673 | **0.81** | **0.73** | **10868** | **1.08** | **0.10** |
| | Logistic Regression | 0.383 | 0.668 | 0.69 | 0.74 | 21138 | 1.69 | 16.39 |
| **iii** | Bhattacharyya | 0.414 | **0.683** | 0.76 | 0.60 | 20066 | 1.34 | 1.50 |
| | Logistic Regression | **0.444** | 0.688 | **0.80** | 0.59 | **19706** | 1.32 | 0.58 |

---

## Building and Running

### Environment Setup
The project uses a Conda environment named `i24`. 
```bash
source activate i24
```

### Core Commands
- **Run Pipeline:**
  ```bash
  python pp_lite.py i --config parameters_LR.json --tag LR
  ```
- **Evaluate Results (MOT Metrics):**
  ```bash
  python mot_i24.py i
  ```
- **Batch Experiments:**
  ```bash
  python run_experiments.py --all-configs --all-suffixes --evaluate
  ```

---

## Development Conventions & Invariants

### Mandatory Working Rules
1. Describe approach and wait for approval before writing code.
2. Break tasks into smaller sub-tasks if they affect >3 files.
3. List potential breakages and suggested tests after writing code.
4. Always write a reproducing test for bugs first.

### Model & Data Invariants
- **LR Model:** Use `Logistic Regression/model_artifacts/consensus_top10_full47.pkl` (47 features, 2,100 pairs).
- **File Naming:** 
    - RAW: `RAW_*_Bhat.json`
    - REC: `REC_*.json` (Bhattacharyya) and `REC_*_LR.json` (Logistic Regression).
- **Evaluation:** `mot_i24.py` handles all naming conventions. Sw/GT is frame-level ID switches / GT object count.
- **Feature Consistency:** Use centralized `StitchFeatureExtractor` in `utils/features_stitch.py`.

### Quality & Diagnostics
- Use `diagnose_json.py --fix` for malformed outputs.
- Prefer split-before-scale protocol for LR diagnostics.

---

## Agent Workflow (Codex/Beads)

- **Issue Tracking:** Use `bd` (beads). `bd ready`, `bd show <id>`, `bd update <id> --status in_progress`, `bd close <id>`.
- **Memory:** Persist project memory in `GEMINI.md`, `AGENTS.md` and `CLAUDE.md`.
- **Session End:** Create issues for remaining work, run quality gates, update issue status, and **MANDATORY** `git push`.

---

## Active Experiment Plan: Transformer Ranking (Feb 19, 2026)

**Goal:** Address train/eval gap by training on full-size candidate groups (matching gate evaluation conditions).

**Current Setup:**
- 12 anchors/batch × 6 cands/anchor = 72 pairs/batch.
- Dataset groups average ~48 candidates, max 80.

**Proposed Changes:**
- Increase to full groups (~48×12 = 576 pairs/batch).
- Use `105K param model` on CUDA (memory should be fine).

**Retrain Command:**
```bash
conda activate i24 && python models/train_transformer_ranking.py \
    --dataset-path models/outputs/transformer_ranking_dataset.jsonl \
    --output-path models/outputs/transformer_ranking_model.pth \
    --epochs 30 --patience 7 \
    --cands-per-anchor 80 \
    --anchors-per-batch 4
```

**Key Parameter Adjustments:**
- `--cands-per-anchor 80` (was 6): Use all candidates up to dataset max.
- `--anchors-per-batch 4` (was 12): Reduced to keep memory reasonable (4×80=320 pairs/batch vs 12×6=72).
- `--patience 7` (was 5): Increased patience for harder task.

---

## Key Directory Structure
- `/`: Main pipeline entry and benchmarking.
- `/utils/`: Core utilities, `StitchCostFunction` ABC, and feature extractors.
- `/Logistic Regression/`: Training and artifacts for the LR model.
- `/Siamese-Network/`: BiLSTM similarity model implementation.
- `/hota/`: Tracking evaluation metrics.
- `/models/` & `/outputs/`: Model storage and results.
