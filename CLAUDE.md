# Deep Learning Improvements for I24-Postprocessing-Lite Pipeline

## Overview
Replace the `stitch_cost()` function in `utils/utils_stitcher_cost.py` with learned models, compare multiple architectures, and integrate the best model into the pipeline.

**Environment**: Run in `i24` conda environment (`conda activate i24`)

---

## Latest Results (Feb 3, 2026) - Scenario i

| Method | MOTA | MOTP | Precision | Recall | FP | Fgmt/GT | Sw/GT |
|--------|------|------|-----------|--------|------|---------|-------|
| **Bhattacharyya** | **0.794** | 0.645 | **0.96** | 0.83 | **3939** | **1.55** | **0.57** |
| Logistic Regression | 0.740 | 0.645 | 0.90 | 0.83 | 10534 | 1.62 | 0.58 |
| Siamese BiLSTM | 0.637 | 0.645 | 0.81 | 0.83 | 23001 | 1.98 | 0.62 |

**Key Finding**: Bhattacharyya baseline outperforms both learned methods on scenario i:
- LR: 2.7× more FPs (10534 vs 3939), MOTA drops 0.054
- Siamese: 5.8× more FPs (23001 vs 3939), MOTA drops 0.157
- All methods achieve same recall (0.83) but precision varies significantly (0.96 → 0.90 → 0.81)

### Configuration for Each Cost Function

**Bhattacharyya** (default thresholds):
```json
"cost_function": { "type": "bhattacharyya" },
"stitcher_args": { "stitch_thresh": 3, "master_stitch_thresh": 4 }
```

**Logistic Regression**:
```json
"cost_function": {
    "type": "logistic_regression",
    "model_path": "Logistic Regression/model_artifacts/combined_optimal_10features.pkl",
    "scale_factor": 5.0,
    "time_penalty": 0.1
}
```

**Siamese** (adjusted thresholds for 0-2 cost range):
```json
"cost_function": {
    "type": "siamese",
    "checkpoint_path": "Siamese-Network/outputs/final_model.pth",
    "device": "cpu"
},
"stitcher_args": { "stitch_thresh": 0.8, "master_stitch_thresh": 1.0 }
```

---

## Current Project Structure

```
I24-postprocessing-lite/
├── Siamese-Network/           # Deep learning models
│   ├── siamese_model.py       # BiLSTM encoder + similarity head (366 lines)
│   ├── siamese_dataset.py     # Dataset loaders with augmentation (860 lines)
│   ├── train_siamese.py       # Training loop with checkpointing (480 lines)
│   ├── evaluate_siamese.py    # Metrics: ROC-AUC, AP, accuracy (294 lines)
│   ├── hard_negative_mining.py # Finer binning (0.5s/50ft) (567 lines)
│   ├── trajectory_masking.py  # Synthetic positives via masking (458 lines)
│   ├── visualize_siamese.py   # Embedding/ROC visualizations (852 lines)
│   ├── visualize_results.py   # Pipeline comparison plots (546 lines)
│   ├── pipeline_integration.py # Early integration attempt (311 lines)
│   ├── data/                  # Copies of GT/RAW JSON files
│   ├── outputs/               # Trained models & checkpoints
│   │   ├── best_accuracy.pth  # Best model (integrated)
│   │   ├── best_loss.pth      # Best by validation loss
│   │   ├── config.json        # Training hyperparameters
│   │   └── training_history.json
│   └── logs/                  # SLURM job logs
│
├── Logistic Regression/       # Enhanced LR baseline
│   ├── train_torch_lr.py      # PyTorch LR trainer
│   ├── logistic_regression.pth # Trained model
│   ├── training_dataset_advanced.npz  # 47 engineered features
│   └── training_dataset_combined.npz
│
├── utils/                     # Core utilities
│   ├── utils_mcf.py           # MOTGraphSingle (356 lines) - uses cost_fn
│   ├── utils_stitcher_cost.py # Original Bhattacharyya cost (226 lines)
│   ├── stitch_cost_interface.py # Strategy pattern abstraction (296 lines)
│   ├── utils_opt.py           # Optimization utilities (679 lines)
│   └── misc.py                # Helper functions (496 lines)
│
├── Core Pipeline Files
│   ├── pp_lite.py             # Main orchestrator (208 lines)
│   ├── merge.py               # Fragment merging (481 lines)
│   ├── min_cost_flow.py       # Stitch stage (142 lines)
│   ├── reconciliation.py      # Trajectory smoothing (223 lines)
│   └── data_feed.py           # Data reader (132 lines)
│
├── Evaluation & Utilities
│   ├── mot_i24.py             # MOT metrics (MOTA/MOTP/Fgmt/Sw) (320 lines)
│   └── diagnose_json.py       # JSON diagnostic/fix utility
│
├── Configuration
│   └── parameters.json        # Pipeline config with cost_function section
│
└── Data Files (204 MB total)
    ├── GT_i.json, GT_ii.json, GT_iii.json    # Ground truth
    ├── RAW_i.json, RAW_ii.json, RAW_iii.json # Raw fragments
    └── REC_i.json, REC_ii.json, REC_iii.json # Reconciled output
```

---

## Implementation Status

### Completed

| Component | File | Status |
|-----------|------|--------|
| **Abstraction Layer** | `utils/stitch_cost_interface.py` | ✅ StitchCostFunction ABC, BhattacharyyaCostFunction, SiameseCostFunction, CostFunctionFactory |
| **Pipeline Integration** | `utils/utils_mcf.py:62-63,105` | ✅ Uses `cost_fn.compute_cost()` |
| **Hard Negative Mining** | `Siamese-Network/hard_negative_mining.py` | ✅ Finer binning (0.5s/50ft vs 2s/150ft) |
| **Trajectory Masking** | `Siamese-Network/trajectory_masking.py` | ✅ Synthetic positives (~15K pairs from 2K) |
| **Siamese BiLSTM** | `Siamese-Network/siamese_model.py` | ✅ 128 hidden, 2 layers, bidirectional |
| **Enhanced Dataset** | `Siamese-Network/siamese_dataset.py` | ✅ EnhancedTrajectoryPairDataset with augmentation |
| **Training Pipeline** | `Siamese-Network/train_siamese.py` | ✅ Checkpointing, LR scheduling |
| **Evaluation** | `Siamese-Network/evaluate_siamese.py` | ✅ ROC-AUC, AP, accuracy |
| **MOT Evaluation** | `mot_i24.py` | ✅ MOTA/MOTP/IDF1 across 3 scenarios |
| **LR Baseline** | `Logistic Regression/train_torch_lr.py` | ✅ PyTorch implementation |
| **Config** | `parameters.json` | ✅ cost_function section added |

### Recently Fixed (Feb 3, 2026)

| Issue | Fix | File |
|-------|-----|------|
| **Empty REC_i.json output** | Restored `mp.Manager()` for shared queues/dicts | `pp_lite.py` |
| **Missing direction filter** | Added filter in static_data_reader | `data_feed.py:69-73` |
| **Siamese cost mismatch** | Adjusted thresholds (0.8/1.0 vs 3/4) | `parameters.json` |
| **Malformed JSON output** | Created diagnose_json.py utility | `diagnose_json.py` |
| **Missing MOT metrics** | Added Fgmt/GT and Sw/GT to output | `mot_i24.py` |

### Not Yet Implemented

| Component | Planned File | Notes |
|-----------|--------------|-------|
| ~~Unified feature extractor~~ | `utils/features_stitch.py` | ✅ Now implemented |
| MLP with features | `models/mlp_features.py` | Could use LR features |
| Siamese Transformer | `models/siamese_transformer.py` | Future work |
| TCN | `models/tcn.py` | Future work |
| Experiment benchmark | `experiments/benchmark.py` | Evaluation done manually |

---

## Part 1: Baseline Results (Bhattacharyya - Target to Beat)

### Updated Measurements (Feb 3, 2026)

| Metric | RAW_i | REC_i (Bhatt) | REC_i (LR) |
|--------|-------|---------------|------------|
| Precision | 0.94 | **0.96** | 0.90 |
| Recall | 0.39 | **0.83** | 0.83 |
| MOTA | 0.358 | **0.794** | 0.740 |
| MOTP | 0.582 | 0.645 | 0.645 |
| FP | 3042 | **3939** | 10534 |
| IDsw | 475 | **179** | 182 |
| No. trajs | 789 | 484 | 506 |

### Historical Measurements (reference)

| Metric | RAW_i | REC_i | RAW_ii | REC_ii | RAW_iii | REC_iii |
|--------|-------|-------|--------|--------|---------|---------|
| Precision | 0.71 | **0.90** | 0.87 | 0.88 | 0.76 | 0.76 |
| Recall | 0.56 | **0.83** | 0.55 | 0.79 | 0.46 | 0.67 |
| MOTA | 0.32 | **0.74** | 0.48 | 0.68 | 0.31 | 0.46 |
| MOTP | 0.63 | **0.73** | 0.72 | 0.75 | 0.67 | 0.68 |
| Fgmt/GT | 5.22 | 0.60 | 5.38 | 1.93 | 4.92 | 1.11 |
| Sw/GT | 1.43 | **0.04** | 2.98 | 0.53 | 3.01 | 0.52 |
| No. trajs | 789 | 321 | 411 | 150 | 1250 | 282 |

---

## Part 2: Contrastive Learning - Current Implementation

### What IS Already Implemented

| Component | File:Lines | Details |
|-----------|------------|---------|
| **ContrastiveLoss** | `siamese_model.py:221-261` | Euclidean distance, margin=2.0, pulls similar/pushes dissimilar |
| **CombinedLoss** | `siamese_model.py:264-307` | `(1-α)×BCE + α×Contrastive`, α=0.5 |
| **Hard Negative Mining** | `hard_negative_mining.py` | Finer binning (0.5s/50ft vs 2s/150ft), hardness scoring |
| **Trajectory Masking** | `trajectory_masking.py` | Synthetic positives at camera boundaries, curriculum levels |
| **Endpoint Features** | `stitch_cost_interface.py:157-185` | [time_gap, x_gap, y_gap, velocity_diff] added to similarity head |

### Contrastive Loss Formula
```python
# Similar pairs (label=1): minimize distance
loss_similar = labels * distance²

# Dissimilar pairs (label=0): push apart by margin
loss_dissimilar = (1-labels) * max(0, margin - distance)²

# Combined: (1-α)×BCE + α×Contrastive
total_loss = 0.5 * BCE + 0.5 * ContrastiveLoss
```

### What is NOT Implemented
- **Triplet Loss** (anchor, positive, negative) - only pairwise contrastive
- **Online Hard Negative Mining** - negatives are mined offline before training
- **Cost Calibration** - no mapping between similarity and Bhattacharyya cost scale

---

## Part 3: Root Cause - Why 100% Val Accuracy but Poor Pipeline Performance

The Siamese model achieves **100% validation accuracy** (see `training_history.json`) but produces **worse MOT metrics** than Bhattacharyya baseline.

### Primary Issue: Cost Scale Mismatch

| Cost Function | Output Range | stitch_thresh | Effect |
|---------------|--------------|---------------|--------|
| **Bhattacharyya** | 0 to 10+ | 3 | Correctly filters bad pairs |
| **Siamese** | 0 to ~2 | 3 | **Accepts almost everything!** |

**Siamese cost formula** (`stitch_cost_interface.py:228-233`):
```python
base_cost = 1.0 - similarity  # similarity ∈ [0,1], so base_cost ∈ [0,1]
time_cost = 0.1 * gap         # typically 0.05-0.5
total_cost = base_cost + time_cost  # Total: ~0.1 to ~1.5
```

With `stitch_thresh=3`, even bad pairs with similarity=0.1 get cost=0.9+time ≈ 1.0 < 3, so they're accepted!

### Secondary Issues

| Issue | Impact |
|-------|--------|
| **Distribution shift** | Training: clean pairs with clear labels; Pipeline: noisy edge cases |
| **Missing domain knowledge** | Bhattacharyya uses WLS projection, uncertainty cones (cx,mx,cy,my) |
| **Threshold not calibrated** | `stitch_thresh=3` tuned for Bhattacharyya, not Siamese costs |

---

## Part 4: Proposed Fixes (Priority Order)

### Fix 1: Cost Calibration (Quick - 30 min)
Modify `stitch_cost_interface.py:228-233`:
```python
# Option A: Scale cost to match Bhattacharyya range
total_cost = (base_cost * 5) + time_cost  # Now range ~0.5 to 6

# Option B: Adjust threshold in parameters.json
"stitch_thresh": 0.5,  # Instead of 3
"master_stitch_thresh": 0.7
```

### Fix 2: Add Cost Distribution Analysis (1 hour)
Create script to analyze cost distributions on actual pipeline data:
```python
# Compare Bhattacharyya vs Siamese costs on same fragment pairs
# Find optimal scaling factor and threshold
```

### Fix 3: Triplet Loss Training (2-3 hours)
Add triplet loss to `siamese_model.py`:
```python
class TripletLoss(nn.Module):
    def forward(self, anchor, positive, negative):
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(d_pos - d_neg + margin)
        return loss.mean()
```

### Fix 4: Hybrid Ensemble (2-3 hours)
Combine Siamese + Bhattacharyya costs:
```python
def compute_cost(track1, track2, TIME_WIN, param):
    cost_bhat = bhattacharyya_cost(track1, track2, TIME_WIN, param)
    cost_siam = siamese_cost(track1, track2)
    # Learn optimal weighting
    return alpha * cost_bhat + (1-alpha) * (cost_siam * scale_factor)
```

---

## Part 5: Original Diagnosis (Historical)

The earlier Siamese implementation had these issues (now partially addressed):

| Issue | Location | Impact | Status |
|-------|----------|--------|--------|
| **Poor features** | `siamese_dataset.py:120-155` | Uses only 4 raw features [x,y,v,t] | ✅ Fixed: endpoint features added |
| **Weak negatives** | `siamese_dataset.py:180-249` | Coarse binning (2s/150ft) | ✅ Fixed: 0.5s/50ft binning |
| **Limited data** | ~2,312 fragments → ~2,000 pairs | Deep learning needs more | ✅ Fixed: masking → ~15K pairs |
| **Overparameterized** | 256K params for 2K samples | Overfitting risk | ⚠️ Still a concern |
| **Missing gap info** | No explicit time/space gap | Critical domain knowledge | ✅ Fixed: endpoint features |
| **Cost calibration** | Similarity → cost mapping | Pipeline integration | ❌ NOT FIXED - primary issue |

---

## Part 2: Key File Details

### Core Pipeline Files

| File | Lines | Purpose |
|------|-------|---------|
| **pp_lite.py** | 208 | Main orchestrator - runs single pass without parallel compute. Manages 5 multiprocessing stages: feed → merge → stitch → reconcile → write |
| **merge.py** | 481 | Merges temporally overlapping fragments using Bhattacharyya distance and resamples to 25Hz |
| **min_cost_flow.py** | 142 | Implements online min-cost flow (stitch) stage using MOTGraphSingle for fragment association |
| **reconciliation.py** | 223 | Reconciles trajectories: resampling + CVXOPT optimization for smoothness |
| **data_feed.py** | 132 | Reads raw trajectory data from static collections and distributes to queues |

### Utils Module

| File | Lines | Purpose |
|------|-------|---------|
| **utils_mcf.py** | 356 | MOTGraphSingle class - graph-based fragment tracking. **Key: Line 62-63 initialize cost function, Line 105 uses cost_fn.compute_cost()** |
| **utils_stitcher_cost.py** | 226 | Original Bhattacharyya-based stitch cost function (weighted least squares, uncertainty cone) |
| **stitch_cost_interface.py** | 296 | Strategy pattern abstraction layer - StitchCostFunction ABC with BhattacharyyaCostFunction, SiameseCostFunction, CostFunctionFactory |
| **utils_opt.py** | 679 | Optimization utilities: combine_fragments, resample, opt1/opt2/opt2_l1 reconciliation solvers |
| **misc.py** | 496 | Helper functions: calc_fit, nan_helper, interpolate, add_filter, RANSAC outlier detection |

### Siamese Network Files

| File | Lines | Purpose |
|------|-------|---------|
| **siamese_model.py** | 366 | Twin LSTM encoders + similarity head. Classes: TrajectoryEncoder (BiLSTM), SimilarityHead (MLP), SiameseTrajectoryNetwork, CombinedLoss (BCE + Contrastive) |
| **siamese_dataset.py** | 860 | Dataset loaders: TrajectoryPairDataset (basic), EnhancedTrajectoryPairDataset (with augmentation). Features: collate_fn, gt_ids-based positive/negative pairing |
| **train_siamese.py** | 480 | SiameseTrainer class - training loops, validation, checkpoint management, LR scheduling |
| **evaluate_siamese.py** | 294 | SiameseEvaluator - metrics: ROC-AUC, AP, accuracy, confusion matrix |
| **hard_negative_mining.py** | 567 | HardNegativeMiner + HardNegativeConfig: finer spatial-temporal binning (0.5s, 50ft) |
| **trajectory_masking.py** | 458 | TrajectoryMasker + MaskConfig: creates synthetic positives by masking GT trajectories near camera boundaries |
| **visualize_siamese.py** | 852 | Plots embeddings, similarity distributions, ROC curves, confusion matrices |
| **visualize_results.py** | 546 | Pipeline comparison visualizations across all three scenarios |

---

## Part 3: Data Format

### Fragment/Trajectory Dictionary Structure
```json
{
  "timestamp": [float array],
  "x_position": [float array],
  "y_position": [float array],
  "velocity": [float array],
  "length": [float or float array],
  "width": [float or float array],
  "height": [float or float array],
  "direction": 1 (EB) or -1 (WB),
  "_id": "unique fragment ID",
  "gt_ids": [[{"$oid": "ground truth vehicle ID"}]],
  "first_timestamp": float,
  "last_timestamp": float,
  "starting_x": float,
  "ending_x": float,
  "detection_confidence": [float array],
  "compute_node_id": int
}
```

### Scenario Data Summary

| Scenario | GT File | RAW File | Vehicles | Fragments | Duration |
|----------|---------|----------|----------|-----------|----------|
| i (free-flow) | GT_i.json (24 MB) | RAW_i.json (43 MB) | 313 | 789 | 60s |
| ii (snowy) | GT_ii.json (13 MB) | RAW_ii.json (19 MB) | 99 | 411 | 51s |
| iii (congested) | GT_iii.json (62 MB) | RAW_iii.json (50 MB) | 266 | 1,112 | 50s |

---

## Part 4: Pipeline Flow

```
RAW Fragments
    ↓
[FEED] - data_feed.py (reads GT/RAW JSON)
    ↓
[MERGE] - merge.py (Bhattacharyya distance, resample to 25Hz)
    ↓
[STITCH] - min_cost_flow.py:MOTGraphSingle
    ├─ Initialize: CostFunctionFactory.create(parameters["cost_function"])
    ├─ For each fragment pair:
    │  └─ cost = self.cost_fn.compute_cost(frag1, frag2, TIME_WIN, param)
    │     (either Bhattacharyya or Siamese)
    └─ Graph min-cost flow for association
    ↓
[RECONCILE] - reconciliation.py (CVXOPT smoothness optimization)
    ↓
[WRITE] - write reconciled trajectories
    ↓
[EVALUATE] - mot_i24.py (MOT metrics)
```

---

## Part 5: Configuration

### parameters.json Key Sections

```json
{
  "cost_function": {
    "type": "siamese",  // or "bhattacharyya"
    "checkpoint_path": "Siamese-Network/outputs/best_accuracy.pth",
    "device": "cuda"
  },
  "stitcher_args": {
    "cx": 0.2, "mx": 0.1, "cy": 2, "my": 0.1,
    "stitch_thresh": 3,  // local mode
    "stitch_thresh_master": 4  // master mode
  },
  "reconciliation_args": {
    "lam2_x": 1, "lam2_y": 1,  // smoothness
    "lam3_x": 1, "lam3_y": 1,  // acceleration
    "lam1_x": 1, "lam1_y": 1   // data fidelity
  }
}
```

### Siamese Training Config (outputs/config.json)
```json
{
  "hidden_size": 128,
  "num_layers": 2,
  "bidirectional": true,
  "dropout": 0.3,
  "margin": 2.0,
  "alpha": 0.5,  // Combined BCE + Contrastive loss weight
  "step_size": 15,
  "gamma": 0.5
}
```

---

## Part 6: Running Commands

**Important**: Always activate i24 environment first:
```bash
conda activate i24
```

### Run Full Pipeline
```bash
python pp_lite.py i      # Scenario i (free-flow)
python pp_lite.py ii     # Scenario ii (snowy)
python pp_lite.py iii    # Scenario iii (congested)
```

### MOT Evaluation
```bash
python mot_i24.py i      # Compare RAW vs REC vs GT
python mot_i24.py ii
python mot_i24.py iii
```

### Train Siamese Model
```bash
cd Siamese-Network
python train_siamese.py
```

### Evaluate Model
```bash
cd Siamese-Network
python evaluate_siamese.py
```

### Quick Test (single cost function)
1. Edit `parameters.json` to set desired `cost_function`
2. Run `python pp_lite.py i`
3. If JSON issues, run: `python diagnose_json.py REC_i.json --fix`
4. Run `python mot_i24.py i`

### Diagnose/Fix JSON Issues
```bash
python diagnose_json.py REC_i.json          # Diagnose only
python diagnose_json.py REC_i.json --fix    # Auto-fix common issues
```

---

## Part 7: Feature Engineering for Enhanced LR

Based on current `stitch_cost()`, extract these features:

### Temporal Features
- `time_gap`: t2[0] - t1[-1]
- `duration_ratio`: len(t1) / len(t2)

### Spatial Features
- `x_gap`: x2[0] - x1[-1]
- `y_gap`: y2[0] - y1[-1]
- `x_gap_predicted`: x2[0] - (slope * t2[0] + intercept)  # Using WLS fit

### Kinematic Features
- `velocity_diff`: v2[0] - v1[-1]
- `velocity_ratio`: v2[0] / v1[-1]
- `acceleration_anchor`: (v1[-1] - v1[-n1]) / (t1[-1] - t1[-n1])

### Uncertainty Features (from Bhattacharyya)
- `sigma_x`: cx + mx * gap * |velocity|
- `sigma_y`: cy + my * gap * |velocity|
- `y_variance_track1`: var(y1)
- `y_variance_track2`: var(y2)

### Vehicle Features
- `length_diff`: |mean(length1) - mean(length2)|
- `width_diff`: |mean(width1) - mean(width2)|
- `direction_match`: 1 if same direction, 0 otherwise

### Confidence Features
- `conf_min`: min(min(conf1), min(conf2))
- `conf_mean`: (mean(conf1) + mean(conf2)) / 2

---

## Part 8: Future Improvements

### Merge Stage (`merge.py`)
- Learn merge threshold instead of fixed 0
- Add detection confidence weighting
- Multi-hypothesis merging

### Reconciliation Stage (`reconciliation.py`)
- Learn per-trajectory λ parameters
- Physics-informed neural network for smoothing
- Confidence-weighted optimization

### Additional Models to Implement
- MLP with 47 engineered features
- Siamese Transformer
- Temporal Convolutional Network (TCN)
