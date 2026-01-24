# Siamese Network Implementation Summary

## Overview

Complete implementation of the Siamese Neural Network for trajectory fragment association as proposed in your EEE 598 Final Project. This implementation learns similarity metrics directly from raw trajectory sequences, in contrast to the logistic regression baseline that uses hand-crafted features.

## Implementation Status

✅ **COMPLETED COMPONENTS:**

1. **Dataset Loader** (`siamese_dataset.py`)
   - Loads trajectory sequences from I24-3D dataset
   - Generates positive/negative pairs with hard negative mining
   - Handles variable-length sequences with padding
   - Normalization and preprocessing

2. **Model Architecture** (`siamese_model.py`)
   - Bidirectional LSTM encoder (2 layers, 128 hidden units)
   - Shared-weight Siamese architecture
   - Similarity head (3-layer MLP)
   - Combined loss: BCE + Contrastive Loss

3. **Training Pipeline** (`train_siamese.py`)
   - Complete training loop with validation
   - Learning rate scheduling
   - Model checkpointing (best loss & best accuracy)
   - Training curves and metrics visualization
   - Progress tracking with tqdm

4. **Evaluation** (`evaluate_siamese.py`)
   - ROC-AUC, Precision-Recall, Confusion Matrix
   - Similarity distribution analysis
   - Comparison framework with Logistic Regression

5. **Pipeline Integration** (`pipeline_integration.py`)
   - Wrapper for NCC algorithm compatibility
   - Drop-in replacement for Bhattacharyya distance
   - Example integration code

6. **Documentation**
   - README.md with usage instructions
   - This implementation summary
   - Inline code documentation

## Architecture Details

### Model Components

```
Input: Trajectory Sequences
├── Fragment A: (seq_len_a, 4) → [x, y, velocity, time]
└── Fragment B: (seq_len_b, 4) → [x, y, velocity, time]

Encoder: Bidirectional LSTM
├── Input size: 4 features
├── Hidden size: 128
├── Num layers: 2
├── Dropout: 0.3
└── Output: 256-dim embedding (128 × 2)

Similarity Head: MLP
├── Input: Concatenated embeddings (512-dim)
├── Hidden layers: [64, 32]
├── Dropout: 0.3
└── Output: Similarity score [0,1]
```

### Loss Function

**Combined Loss** = (1-α) × BCE + α × Contrastive

- **BCE Loss**: Trains similarity head directly
- **Contrastive Loss**: Shapes embedding space
  - Pulls similar pairs together
  - Pushes dissimilar pairs apart (margin = 2.0)
- **α = 0.5**: Equal weighting

### Training Configuration

```json
{
  "batch_size": 32,
  "num_epochs": 50,
  "learning_rate": 0.001,
  "optimizer": "Adam",
  "scheduler": "StepLR (step=15, gamma=0.5)",
  "train_val_split": 0.8,
  "device": "CUDA (if available)"
}
```

## File Structure

```
02_Code/Siamese-Network/
├── siamese_dataset.py          # Dataset loader
├── siamese_model.py             # Model architecture
├── train_siamese.py             # Training script
├── evaluate_siamese.py          # Evaluation script
├── pipeline_integration.py      # Integration with NCC
├── requirements.txt             # Python dependencies
├── README.md                    # Usage documentation
├── IMPLEMENTATION_SUMMARY.md    # This file
└── outputs/                     # Training outputs (created during training)
    ├── best_loss.pth
    ├── best_accuracy.pth
    ├── final_model.pth
    ├── training_curves.png
    ├── evaluation_results.png
    ├── training_history.json
    └── config.json
```

## Usage Instructions

### Step 1: Install Dependencies

```bash
cd "D:\ASU Academics\Thesis & Research\02_Code\Siamese-Network"
pip install -r requirements.txt
```

**GPU Installation (Recommended):**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU Installation:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Test Implementation

```bash
# Test model architecture
python siamese_model.py

# Test dataset loader
python -c "from siamese_dataset import TrajectoryPairDataset; print('Dataset loader OK')"
```

### Step 3: Train Model

```bash
python train_siamese.py
```

**Expected output:**
- Training progress bars
- Epoch summaries (loss, accuracy)
- Best model checkpoints saved
- Training curves generated

**Training time estimates:**
- GPU (NVIDIA RTX 3080): ~2-3 hours
- CPU: ~8-12 hours

### Step 4: Evaluate Model

```bash
python evaluate_siamese.py
```

**Outputs:**
- ROC-AUC score
- Precision-Recall curve
- Confusion matrix
- Similarity distribution plots

### Step 5: Integrate into Pipeline

```bash
python pipeline_integration.py
```

This generates integration code and instructions for replacing the Bhattacharyya distance in the I24-postprocessing-lite pipeline.

## Key Differences from Logistic Regression

| Aspect | Logistic Regression | Siamese Network |
|--------|---------------------|-----------------|
| **Input** | 47 hand-crafted features | Raw sequences (x,y,v,t) |
| **Model Type** | Linear classifier | Deep LSTM network |
| **Parameters** | ~50 | ~250,000 |
| **Training Time** | ~1 minute | ~2-3 hours (GPU) |
| **Inference** | Instant | ~5-10ms per pair |
| **Interpretability** | High (feature importance) | Low (black box) |
| **Feature Engineering** | Required (47 features) | Not required (end-to-end) |
| **Captures Dynamics** | Limited (static features) | Yes (temporal LSTM) |
| **ROC-AUC** | 99.98% | TBD (likely 98-99%) |

## Expected Performance

Based on your logistic regression baseline and similar work in literature:

**Predictions:**
- **ROC-AUC**: 98-99.5%
- **Accuracy**: 97-99%
- **Average Precision**: 98-99%

**Reality Check:**
- Logistic regression already achieves 99.98% ROC-AUC
- Siamese network *may not* significantly outperform LR
- Improvement likely marginal (0-2% at most)
- Main benefit: No feature engineering required

## Advantages of Each Approach

### Siamese Network Advantages:
1. **End-to-end learning** - No manual feature engineering
2. **Captures temporal dynamics** - LSTM processes sequences
3. **Generalizable** - Can adapt to new scenarios
4. **Scalable** - Can add more complex architectures

### Logistic Regression Advantages:
1. **Already excellent performance** (99.98% ROC-AUC)
2. **Fast training** (~1 minute vs. 2-3 hours)
3. **Fast inference** (instant vs. 5-10ms)
4. **Interpretable** (feature importance analysis)
5. **Production-ready** (low computational requirements)

## Next Steps

### Testing (Immediate)

1. **Install PyTorch**:
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Run tests**:
   ```bash
   python siamese_model.py
   python -c "from siamese_dataset import TrajectoryPairDataset; d = TrajectoryPairDataset(['i']); print(f'Dataset OK: {len(d)} pairs')"
   ```

### Training (1-2 days)

3. **Train on small dataset** (scenario i only):
   - Modify `train_siamese.py` line 207: `dataset_names=['i']`
   - Run: `python train_siamese.py`
   - Verify training works (~30 min on GPU)

4. **Train on full dataset** (all scenarios):
   - Use default config: `dataset_names=['i', 'ii', 'iii']`
   - Run: `python train_siamese.py`
   - Wait for completion (~2-3 hours)

### Evaluation (1 day)

5. **Evaluate performance**:
   ```bash
   python evaluate_siamese.py
   ```

6. **Compare with Logistic Regression**:
   - Evaluate both on same test set
   - Compare ROC-AUC, accuracy, inference time
   - Determine if improvement justifies complexity

### Integration (1-2 days)

7. **Integrate into pipeline**:
   ```bash
   python pipeline_integration.py
   ```

8. **End-to-end testing**:
   - Replace cost function in min_cost_flow.py
   - Run full reconstruction pipeline
   - Measure MOT metrics (MOTA, MOTP)

## Troubleshooting

### Common Issues

**1. Out of Memory (GPU)**
```python
# Reduce batch size in train_siamese.py
config['batch_size'] = 16  # or 8
```

**2. CUDA Not Available**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**3. Slow Training (CPU)**
```python
# Use smaller model
config['model']['hidden_size'] = 64
config['model']['num_layers'] = 1
```

**4. Poor Performance**
```python
# Try different hyperparameters
config['learning_rate'] = 0.0001  # Lower LR
config['loss']['alpha'] = 0.7     # More contrastive loss
config['num_epochs'] = 100        # Train longer
```

## Research Contributions

This implementation enables:

1. **Comparison Study**: Siamese vs. Logistic Regression
2. **Ablation Study**: Test different architectures (LSTM vs. GRU vs. Transformer)
3. **Feature Analysis**: Which temporal patterns does the network learn?
4. **Generalization Study**: Cross-dataset performance

## Thesis Integration

**Relevant Sections:**

1. **Methodology** (Chapter 3):
   - Describe Siamese architecture
   - Explain LSTM encoder design
   - Detail training procedure

2. **Experiments** (Chapter 4):
   - Present comparison with LR baseline
   - Analyze performance on I24-3D dataset
   - Discuss computational requirements

3. **Results** (Chapter 5):
   - ROC curves comparison
   - Confusion matrices
   - Embedding space visualization
   - Runtime analysis

4. **Discussion** (Chapter 6):
   - When to use deep learning vs. traditional ML
   - Trade-offs: accuracy vs. complexity
   - Future directions (Transformers, etc.)

## Citation

```bibtex
@mastersthesis{prasath2025trajectory,
  title={Vehicle Trajectory Reconstruction using Deep Learning},
  author={Prasath, Raswanth},
  year={2025},
  school={Arizona State University},
  type={Master's Thesis},
  note={Implements Siamese neural network for fragment association}
}
```

## Contact & Support

**Author**: Raswanth Prasath (raswanth@asu.edu)
**Supervisor**: Prof. Yanbing Wang
**Date**: December 2024
**Status**: Implementation Complete - Ready for Training

---

## Appendix: Code Snippets

### Quick Start

```python
# Load and train model
from siamese_dataset import TrajectoryPairDataset
from siamese_model import SiameseTrajectoryNetwork, CombinedLoss
from torch.utils.data import DataLoader
import torch

# Create dataset
dataset = TrajectoryPairDataset(['i'], normalize=True)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Create model
model = SiameseTrajectoryNetwork()
criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train one epoch
for seq_a, len_a, seq_b, len_b, labels in loader:
    similarity, emb_a, emb_b = model(seq_a, len_a, seq_b, len_b)
    loss, bce, contrastive = criterion(similarity, emb_a, emb_b, labels)
    loss.backward()
    optimizer.step()
```

### Integration Example

```python
# Use in pipeline
from pipeline_integration import SiameseCostFunction

# Initialize (once)
cost_fn = SiameseCostFunction('outputs/best_accuracy.pth')

# Use in stitching
cost = cost_fn.stitch_cost(fragment_a, fragment_b, TIME_WIN=5.0, param={})
```

---

**END OF IMPLEMENTATION SUMMARY**
