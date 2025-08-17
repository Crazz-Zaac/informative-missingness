#!/bin/bash
#SBATCH --job-name=run_exp
#SBATCH --output=slurm_logs/run_exp%j.out  # %j adds job ID
#SBATCH --error=slurm_logs/run_exp%j.err
#SBATCH --time=12:00:00 # Set a time limit (6 hours in this case)
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100  # Or rtx3080/v100/h100
#SBATCH --ntasks=1  
#SBATCH --nodes=1

module load python

# Activate virtual environment 
source /home/hpc/iwbn/iwbn111h/.venv/bin/activate


# Run Python script
python project/run_exp.py