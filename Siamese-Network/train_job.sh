#!/bin/bash
#SBATCH --job-name=siamese_train
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules (adjust based on your cluster)
module purge
module load python/3.10
module load cuda/11.8

# Activate conda environment if needed
# source activate your_env

# Navigate to script directory
cd /home/raswanth/I24/I24-postprocessing-lite/Siamese-Network

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
nvidia-smi

# Run training
python train_siamese.py

echo "End time: $(date)"
