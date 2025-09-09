#!/bin/bash
#SBATCH --job-name=run_exp
#SBATCH --output=slurm_logs/run_exp%j.out  # %j adds job ID
#SBATCH --error=slurm_logs/run_exp%j.err
#SBATCH --time=23:00:00 # Set a time limit (23 hours in this case)
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=a100  # Or rtx3080/v100/h100
#SBATCH --ntasks=1  
#SBATCH --nodes=1
#SBATCH --array=0-4

module load python
module load apptainer

# Activate virtual environment 
source /home/hpc/iwbn/iwbn111h/.venv/bin/activate


# Without apptainer
# python project/run_exp.py

# models=("RandomForest" "GradientBoosting" "LogisticRegression" "XGBoost" "CatBoost")
# MODEL=${models[$SLURM_ARRAY_TASK_ID]}

# echo "Training model: $MODEL"

# apptainer exec --nv -B $PWD:/project model-training.sif \
#     python project/train_model.py --model $MODEL


models=("RandomForest" "GradientBoosting" "LogisticRegression" "XGBoost" "CatBoost")
MODEL=${models[$SLURM_ARRAY_TASK_ID]}

apptainer exec --nv -B $PWD:/project model-training.sif \
    python project/train_model.py --model $MODEL
