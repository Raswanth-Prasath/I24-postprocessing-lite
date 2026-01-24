# Running Siamese Network on Sol

Complete instructions for running the Siamese Neural Network training on Sol computing cluster.

## Quick Start

```bash
# 1. Navigate to the directory
cd "path/to/Siamese-Network"

# 2. Make scripts executable
chmod +x setup_sol.sh run_test.sh run_train.sh

# 3. Setup environment (first time only)
bash setup_sol.sh

# 4. Run tests
sbatch run_test.sh

# 5. Start training
sbatch run_train.sh
```

## Detailed Instructions

### Step 1: Upload Files to Sol

Make sure all these files are in your `Siamese-Network/` directory on Sol:

**Core Python Files:**
- `siamese_dataset.py`
- `siamese_model.py`
- `train_siamese.py`
- `evaluate_siamese.py`
- `pipeline_integration.py`

**Configuration:**
- `requirements.txt`

**Shell Scripts:**
- `setup_sol.sh`
- `run_test.sh`
- `run_train.sh`

**Documentation:**
- `README.md`
- `IMPLEMENTATION_SUMMARY.md`
- `SOL_INSTRUCTIONS.md` (this file)

**Data Access:**
Make sure the dataset path in `siamese_dataset.py` line 21 points to your I24-3D data:
```python
self.dataset_dir = Path(r"D:\ASU Academics\Thesis & Research\01_Papers\Datasets\I24-3D")
```

You may need to update this to your Sol path, e.g.:
```python
self.dataset_dir = Path("/home/your_username/Thesis/Datasets/I24-3D")
```

### Step 2: Initial Setup

```bash
# SSH to Sol
ssh your_username@sol.asu.edu

# Navigate to your directory
cd path/to/Siamese-Network

# Make scripts executable
chmod +x setup_sol.sh run_test.sh run_train.sh

# Run setup (installs dependencies)
bash setup_sol.sh
```

**What setup_sol.sh does:**
- Creates `logs/` and `outputs/` directories
- Installs PyTorch with CUDA support
- Installs all other dependencies
- Verifies installation

### Step 3: Test Implementation

Before running the full training, test that everything works:

```bash
# Submit test job
sbatch run_test.sh

# Check job status
squeue -u $USER

# Monitor output (replace JOBID with your job number)
tail -f logs/test_JOBID.out
```

**What run_test.sh does:**
- Tests model architecture
- Tests dataset loader
- Runs 1 quick training epoch
- Verifies everything works

**Expected output:**
```
✓ Model architecture test PASSED
✓ Dataset loader test PASSED
✓ Quick training test PASSED
All tests PASSED!
```

### Step 4: Start Training

Once tests pass, start full training:

```bash
# Submit training job
sbatch run_train.sh

# Check job status
squeue -u $USER

# Monitor training progress
tail -f logs/train_JOBID.out

# Check for errors
tail -f logs/train_JOBID.err
```

**Training details:**
- **Duration**: 2-3 hours (with GPU)
- **Resources**: 1 GPU, 8 CPUs, 32GB RAM
- **Epochs**: 50 (configurable in `train_siamese.py`)
- **Dataset**: All three scenarios (i, ii, iii)

### Step 5: Monitor Progress

**Check job status:**
```bash
# List your jobs
squeue -u $USER

# Job details
scontrol show job JOBID

# Cancel job if needed
scancel JOBID
```

**Watch training progress:**
```bash
# Live output
tail -f logs/train_JOBID.out

# Check errors
tail -f logs/train_JOBID.err

# View last N lines
tail -n 50 logs/train_JOBID.out
```

**Training output shows:**
- Epoch progress bars
- Training and validation metrics
- Best model saves
- Learning rate updates

### Step 6: Check Results

After training completes:

```bash
# View final output
cat logs/train_JOBID.out

# Check saved models
ls -lh outputs/

# View training history
cat outputs/training_history.json
```

**Expected outputs in `outputs/`:**
- `best_loss.pth` - Best model by validation loss
- `best_accuracy.pth` - Best model by validation accuracy
- `final_model.pth` - Final epoch model
- `training_curves.png` - Training plots
- `training_history.json` - Full metrics
- `config.json` - Training configuration

### Step 7: Evaluate Model

```bash
# Submit evaluation job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=siamese_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module purge
module load python/3.9
module load cuda/11.8

python evaluate_siamese.py
EOF

# Monitor evaluation
tail -f logs/eval_*.out
```

**Evaluation outputs:**
- ROC-AUC score
- Accuracy metrics
- Confusion matrix
- Evaluation plots in `outputs/`

## Troubleshooting

### Common Issues

**1. Dataset not found**
```
Error: FileNotFoundError: [Errno 2] No such file or directory: '..../GT_i.json'
```

**Fix:** Update dataset path in `siamese_dataset.py`:
```python
# Line 21
self.dataset_dir = Path("/your/sol/path/to/I24-3D")
```

**2. Out of memory**
```
RuntimeError: CUDA out of memory
```

**Fix:** Reduce batch size in `train_siamese.py`:
```python
# Line 216
config['batch_size'] = 16  # or 8
```

**3. CUDA not available**
```
CUDA available: False
```

**Fix:**
```bash
# Make sure you requested GPU in SLURM script
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load CUDA module
module load cuda/11.8
```

**4. Module not found**
```
ModuleNotFoundError: No module named 'torch'
```

**Fix:**
```bash
# Run setup again
bash setup_sol.sh

# Or install manually
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**5. Job pending forever**
```bash
# Check partition availability
sinfo

# Try different partition
#SBATCH --partition=general
```

**6. Permission denied on scripts**
```bash
chmod +x setup_sol.sh run_test.sh run_train.sh
```

## Configuration Options

### Modify Training Parameters

Edit `train_siamese.py` (around line 206):

```python
config = {
    'dataset_names': ['i', 'ii', 'iii'],  # Which scenarios
    'batch_size': 32,                     # Reduce if OOM
    'num_epochs': 50,                     # More epochs for better training
    'learning_rate': 0.001,               # Lower if not converging
    'model': {
        'hidden_size': 128,               # Smaller = faster, less capacity
        'num_layers': 2,                  # More layers = more complex
        'dropout': 0.3,                   # Higher = more regularization
    }
}
```

### Modify SLURM Resources

Edit `run_train.sh`:

```bash
#SBATCH --time=24:00:00      # Max runtime
#SBATCH --cpus-per-task=8    # Number of CPUs
#SBATCH --mem=32GB           # Memory
#SBATCH --gres=gpu:1         # Number of GPUs
```

## Performance Expectations

### With GPU (recommended):
- **Training time**: 2-3 hours
- **Memory usage**: ~8-12 GB GPU RAM
- **Throughput**: ~50-100 pairs/second

### Without GPU (not recommended):
- **Training time**: 8-12 hours
- **Memory usage**: ~16-24 GB RAM
- **Throughput**: ~5-10 pairs/second

## File Checklist

Before running, verify all files exist:

```bash
# Check core files
ls -1 *.py
# Should show:
# siamese_dataset.py
# siamese_model.py
# train_siamese.py
# evaluate_siamese.py
# pipeline_integration.py

# Check scripts
ls -1 *.sh
# Should show:
# setup_sol.sh
# run_test.sh
# run_train.sh

# Check config
ls -1 *.txt *.md
# Should show:
# requirements.txt
# README.md
# IMPLEMENTATION_SUMMARY.md
# SOL_INSTRUCTIONS.md
```

## Quick Reference

```bash
# Setup (first time)
bash setup_sol.sh

# Test
sbatch run_test.sh

# Train
sbatch run_train.sh

# Check jobs
squeue -u $USER

# Monitor
tail -f logs/train_*.out

# Cancel
scancel JOBID

# Evaluate
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -p gpu -G 1 -t 1:00:00
module load python/3.9 cuda/11.8
python evaluate_siamese.py
EOF
```

## Expected Timeline

1. **Setup**: 5-10 minutes
2. **Testing**: 10-15 minutes
3. **Training**: 2-3 hours (GPU) or 8-12 hours (CPU)
4. **Evaluation**: 5-10 minutes
5. **Total**: ~3-4 hours with GPU

## Results to Expect

After successful training, you should have:

**Metrics:**
- ROC-AUC: ~98-99%
- Accuracy: ~97-99%
- Similar to or slightly below Logistic Regression baseline (99.98%)

**Files:**
- Trained models in `outputs/`
- Training curves showing convergence
- Evaluation plots comparing performance

## Next Steps After Training

1. **Compare with Logistic Regression**
   - Run both on same test set
   - Compare ROC-AUC, accuracy, inference time

2. **Integrate into Pipeline**
   - Use `pipeline_integration.py`
   - Replace cost function in `min_cost_flow.py`

3. **Write Thesis Section**
   - Document architecture
   - Present results
   - Discuss comparison

## Support

If you encounter issues:

1. Check error logs: `cat logs/train_*.err`
2. Review this guide's troubleshooting section
3. Contact me with:
   - Error message
   - Job output (`logs/train_*.out`)
   - Job ID

## Summary

```bash
# Complete workflow
cd path/to/Siamese-Network
chmod +x *.sh
bash setup_sol.sh           # Setup
sbatch run_test.sh          # Test
sbatch run_train.sh         # Train
# Wait 2-3 hours
python evaluate_siamese.py  # Evaluate
```

Good luck with training! 🚀
