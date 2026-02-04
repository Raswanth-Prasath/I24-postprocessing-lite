#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH --mem=256G              # amount of memory for the job
#SBATCH -c 2                    # number of CPU cores
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00              # time limit
#SBATCH -p arm                  # partition
#SBATCH -q class                # QOS
#SBATCH -o calib_lr_%j.out       # STDOUT (%j = JobId)
#SBATCH -e calib_lr_%j.err       # STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=raswanth@asu.edu

echo "========================================================================"
echo "CALIBRATION SWEEP JOB STARTED"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "========================================================================"

source ~/.bashrc
conda activate i24

echo "Python version:"
python --version

echo -e "\nStarting calibration sweep..."

cd /home/raswanth/I24/I24-postprocessing-lite

# Run the calibration sweep
python "Logistic Regression/calib_lr.py" --scenario i

echo "End Time: $(date)"
