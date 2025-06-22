#!/bin/bash
#SBATCH --job-name=rf_train
#SBATCH --output=rf_train.out
#SBATCH --error=rf_train.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100  # Or rtx3080/v100/h100

module load python/3.8
python project/run_train_rf.py

# Load required modules
module load python

# Activate your virtual environment if you have one
source /home/hpc/iwbn/iwbn111h/.venv/bin/activate

# Run your Python script
python project/run_train_rf.py