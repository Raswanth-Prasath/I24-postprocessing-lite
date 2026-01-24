#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH --mem=256G              # amount of memory for the job
#SBATCH -c 8                    # number of CPU cores
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH -t 10:00:00             # time limit (24 hours)
#SBATCH -p arm                  # partition
#SBATCH -q class                # QOS
#SBATCH -o logs/train_%j.out    # file to save job's STDOUT (%j = JobId)
#SBATCH -e logs/train_%j.err    # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=raswanth@asu.edu  # Your ASU email

echo "========================================================================"
echo "SIAMESE NETWORK TRAINING JOB STARTED"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "========================================================================"

# Python binary path - using i24 conda environment
PYTHON_BIN="/home/raswanth/.conda/envs/i24/bin/python"

# Check if custom Python exists, otherwise use system Python
if [ -f "$PYTHON_BIN" ]; then
    echo "Using custom Python: $PYTHON_BIN"
else
    echo "Custom Python not found, using system Python"
    PYTHON_BIN="python"
fi

echo "Python version:"
$PYTHON_BIN --version

echo -e "\nPyTorch check:"
$PYTHON_BIN -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

echo -e "\nChecking dependencies:"
$PYTHON_BIN -c "import numpy, pandas, matplotlib, sklearn, tqdm; print('✓ All dependencies imported successfully')"

# Create logs and outputs directories
mkdir -p logs outputs
echo -e "\n✓ Created logs/ and outputs/ directories"

echo -e "\n========================================================================"
echo "STARTING TRAINING"
echo "========================================================================"
echo "Training configuration:"
echo "  - Datasets: i, ii, iii (all scenarios)"
echo "  - Batch size: 32"
echo "  - Epochs: 50"
echo "  - Model: Bidirectional LSTM (128 hidden units, 2 layers)"
echo "  - Device: GPU (if available)"
echo "========================================================================"

# Run training
$PYTHON_BIN train_siamese.py

echo -e "\n========================================================================"
echo "TRAINING JOB COMPLETED"
echo "========================================================================"
echo "End Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Check outputs in: ./outputs/"
echo "  - best_accuracy.pth (best model by accuracy)"
echo "  - best_loss.pth (best model by loss)"
echo "  - training_curves.png (training plots)"
echo "  - training_history.json (metrics)"
echo "========================================================================"
