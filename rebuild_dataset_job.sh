#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH --mem=10G              # amount of memory for the job
#SBATCH -c 1                    # number of CPU cores
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00              # time limit
#SBATCH -p public                  # partition
#SBATCH -q class                # QOS
#SBATCH -o rebuild_dataset_%j.out     # STDOUT (%j = JobId)
#SBATCH -e rebuild_dataset_%j.err     # STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=raswanth@asu.edu

echo "========================================================================"
echo "REBUILD DATASET JOB STARTED"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "========================================================================"

source ~/.bashrc
conda activate i24

echo "Python version:"
python --version

echo -e "\nStarting dataset rebuild..."

cd /home/raswanth/I24/I24-postprocessing-lite

python "Logistic Regression/rebuild_dataset.py" --output training_dataset_v4.npz --skip-timespace

echo -e "\nEnd Time: $(date)"

