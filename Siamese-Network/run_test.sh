#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH --mem=64G               # amount of memory for the job
#SBATCH -c 4                    # number of CPU cores
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH -t 1:00:00              # time limit (1 hour)
#SBATCH -p arm                  # partition
#SBATCH -q class                # QOS
#SBATCH -o logs/test_%j.out     # file to save job's STDOUT (%j = JobId)
#SBATCH -e logs/test_%j.err     # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=raswanth@asu.edu  # Your ASU email

echo "========================================================================"
echo "SIAMESE NETWORK TESTING JOB STARTED"
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
$PYTHON_BIN -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Create directories
mkdir -p logs outputs
echo -e "\n✓ Created logs/ and outputs/ directories"

# Test 1: Model architecture
echo -e "\n========================================================================"
echo "TEST 1: Model Architecture"
echo "========================================================================"
$PYTHON_BIN siamese_model.py
if [ $? -eq 0 ]; then
    echo "✓ Model architecture test PASSED"
else
    echo "✗ Model architecture test FAILED"
    exit 1
fi

# Test 2: Dataset loader (small test)
echo -e "\n========================================================================"
echo "TEST 2: Dataset Loader"
echo "========================================================================"
$PYTHON_BIN -c "
from siamese_dataset import TrajectoryPairDataset
print('Creating test dataset...')
dataset = TrajectoryPairDataset(
    dataset_names=['i'],
    normalize=True
)
print(f'✓ Dataset loaded: {len(dataset)} pairs')
print('✓ Dataset loader test PASSED')
"
if [ $? -eq 0 ]; then
    echo "✓ Dataset loader test PASSED"
else
    echo "✗ Dataset loader test FAILED"
    exit 1
fi

# Test 3: Quick training test (1 epoch on small dataset)
echo -e "\n========================================================================"
echo "TEST 3: Quick Training Test (1 Epoch)"
echo "========================================================================"
$PYTHON_BIN -c "
import torch
from torch.utils.data import DataLoader, random_split
from siamese_dataset import TrajectoryPairDataset, collate_fn
from siamese_model import SiameseTrajectoryNetwork, CombinedLoss
from pathlib import Path

print('Loading dataset...')
dataset = TrajectoryPairDataset(['i'], normalize=True)

# Use small subset for testing
subset_size = min(100, len(dataset))
subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

loader = DataLoader(subset, batch_size=8, shuffle=True, collate_fn=collate_fn)

print(f'Testing with {len(subset)} samples...')

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = SiameseTrajectoryNetwork(
    input_size=4,
    hidden_size=64,  # Smaller for quick test
    num_layers=1,
    dropout=0.2,
    bidirectional=True,
    similarity_hidden_dim=32
).to(device)

criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train one epoch
model.train()
total_loss = 0
for seq_a, len_a, seq_b, len_b, labels in loader:
    seq_a = seq_a.to(device)
    len_a = len_a.to(device)
    seq_b = seq_b.to(device)
    len_b = len_b.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    similarity, emb_a, emb_b = model(seq_a, len_a, seq_b, len_b)
    loss, bce, contrastive = criterion(similarity, emb_a, emb_b, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

avg_loss = total_loss / len(loader)
print(f'✓ Training test completed. Average loss: {avg_loss:.4f}')
print('✓ Quick training test PASSED')
"

if [ $? -eq 0 ]; then
    echo "✓ Quick training test PASSED"
else
    echo "✗ Quick training test FAILED"
    exit 1
fi

echo -e "\n========================================================================"
echo "ALL TESTS PASSED!"
echo "========================================================================"
echo "End Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "✓ Model architecture works"
echo "✓ Dataset loader works"
echo "✓ Training pipeline works"
echo ""
echo "READY FOR FULL TRAINING!"
echo "Submit training job with: sbatch run_train.sh"
echo "========================================================================"
