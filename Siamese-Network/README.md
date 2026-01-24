# Siamese Neural Network for Trajectory Fragment Association

Deep learning approach to learning similarity metrics for vehicle trajectory fragment association, as proposed in the EEE 598 Final Project.

## Overview

This implementation uses a Siamese Neural Network with LSTM encoders to learn similarity between trajectory fragments. Unlike the logistic regression baseline that uses hand-crafted features, this approach learns representations directly from raw trajectory sequences.

## Architecture

```
Fragment A (variable length)  →  [LSTM Encoder] → Embedding A
                                        ↓              ↓
                                   (shared weights)    → [Similarity Head] → Score [0,1]
                                        ↓              ↓
Fragment B (variable length)  →  [LSTM Encoder] → Embedding B
```

**Components:**
- **Trajectory Encoder**: Bidirectional LSTM (2 layers, 128 hidden units)
- **Embedding Size**: 256 (128 × 2 for bidirectional)
- **Similarity Head**: 3-layer MLP with dropout
- **Loss Function**: Combined BCE + Contrastive Loss

## Files

- `siamese_dataset.py`: Dataset loader for trajectory pairs (raw sequences, not features)
- `siamese_model.py`: Siamese network architecture and loss functions
- `train_siamese.py`: Training script
- `evaluate_siamese.py`: Evaluation and comparison script
- `pipeline_integration.py`: Integration with I24-postprocessing-lite
- `README.md`: This file

## Installation

```bash
# Required packages
pip install torch numpy pandas matplotlib scikit-learn tqdm statsmodels
```

## Usage

### 1. Dataset Preparation

The dataset loader automatically generates positive and negative pairs from the I24-3D dataset:

```python
from siamese_dataset import TrajectoryPairDataset

dataset = TrajectoryPairDataset(
    dataset_names=['i', 'ii', 'iii'],  # All three scenarios
    normalize=True
)
```

**Data Preparation:**
- **Positive pairs**: Sequential fragments from same GT vehicle
- **Negative pairs**: Hard negatives (different vehicles, spatially/temporally close)
- **Features per timestep**: [x_position, y_position, velocity, time_normalized]

### 2. Training

```bash
cd "D:\ASU Academics\Thesis & Research\02_Code\Siamese-Network"
python train_siamese.py
```

**Training Configuration:**
- Batch size: 32
- Learning rate: 0.001 with step decay
- Epochs: 50
- Train/Val split: 80/20
- Loss: α × Contrastive + (1-α) × BCE (α=0.5)

**Outputs** (saved to `outputs/`):
- `best_loss.pth`: Best model by validation loss
- `best_accuracy.pth`: Best model by validation accuracy
- `training_curves.png`: Loss and accuracy plots
- `training_history.json`: Full training history

### 3. Evaluation

```bash
python evaluate_siamese.py
```

**Metrics:**
- ROC-AUC
- Average Precision
- Confusion Matrix
- Accuracy

### 4. Pipeline Integration

Replace the Bhattacharyya distance cost function in the I24-postprocessing-lite pipeline:

```bash
python pipeline_integration.py
```

This modifies `min_cost_flow.py` to use the learned Siamese network for fragment stitching.

## Model Comparison

### Siamese Network vs Logistic Regression

| Metric | Logistic Regression | Siamese Network |
|--------|---------------------|-----------------|
| Features | 47 hand-crafted | End-to-end learned |
| Input | Feature vectors | Raw sequences |
| Architecture | Linear classifier | LSTM + MLP |
| Parameters | ~50 | ~250K |
| Training | Fast (~1 min) | Slower (~2-3 hours) |
| ROC-AUC | 99.98% | TBD |

**Advantages of Siamese Network:**
- Learns representations from raw data
- Captures temporal dependencies
- Generalizes to unseen dynamics
- No manual feature engineering

**Advantages of Logistic Regression:**
- Much faster training
- Highly interpretable
- Already achieving excellent performance (99.98%)
- Lower computational cost

## Key Differences from Logistic Regression

1. **Input**: Raw sequences vs. engineered features
2. **Model**: Deep LSTM network vs. linear classifier
3. **Learning**: End-to-end representation learning vs. feature-based
4. **Computation**: GPU-accelerated vs. CPU-based

## Expected Performance

Based on similar work in the literature:
- **ROC-AUC**: 98-99.5% (comparable to LR baseline)
- **Inference time**: ~5-10ms per pair (GPU)
- **Training time**: 2-3 hours on GPU

## Integration with NCC Algorithm

The trained model produces similarity scores S(fi, fj) ∈ [0,1]:
- **Cost conversion**: C(fi, fj) = 1 - S(fi, fj)
- **Usage**: Replace `stitch_cost()` in `utils/utils_stitcher_cost.py`

## Future Work

1. **Transformer Architecture**: Replace LSTM with attention mechanisms
2. **Triplet Loss**: Use triplet mining for better embedding space
3. **Multi-task Learning**: Joint training with trajectory prediction
4. **Cross-dataset Generalization**: Test on other trajectory datasets

## References

- Wang et al. (2023): "Online Min Cost Circulation for Multi-Object Tracking on Fragments"
- Leal-Taixé et al. (2016): "Learning by Tracking: Siamese CNN for Robust Target Association"
- Milan et al. (2017): "Online Multi-Target Tracking Using Recurrent Neural Networks"

## Contact

Raswanth Prasath (raswanth@asu.edu)
Supervisor: Prof. Yanbing Wang
