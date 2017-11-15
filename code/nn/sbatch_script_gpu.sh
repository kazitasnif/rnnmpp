#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=32
#SBATCH --mem=2g
#SBATCH --time=5-00:15:00     # 5 day and 15 minutes
#SBATCH --output=script_output/mimic3_survived_gpu.stdout
#SBATCH --job-name="mimic_s"
#SBATCH -p gpu

# Print current date
date
# Print name of node
module load cuda/8.0
hostname

scripts/mimic3_survived_run.sh
