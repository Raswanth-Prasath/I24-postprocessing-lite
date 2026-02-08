#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH --mem=256G              # amount of memory for the job
#SBATCH -c 2                    # number of CPU cores
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00              # time limit (3 scenarios x 80 trials ~3-5h + buffer)
#SBATCH -p public                  # partition
#SBATCH -q class                # QOS
#SBATCH -o optuna_lr_%j.out     # STDOUT (%j = JobId)
#SBATCH -e optuna_lr_%j.err     # STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=raswanth@asu.edu

echo "========================================================================"
echo "OPTUNA LR SEARCH JOB STARTED"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "========================================================================"

source ~/.bashrc
conda activate i24

echo "Python version:"
python --version

echo -e "\nStarting Optuna Bayesian search..."

cd /home/raswanth/I24/I24-postprocessing-lite

python "Logistic Regression/optuna_search_lr.py" \
    --scenarios i,ii,iii \
    --n-model-trials 20 \
    --n-pipeline-trials 80 \
    --n-top-models 5

echo -e "\nEnd Time: $(date)"
