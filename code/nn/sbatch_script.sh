#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=5-00:15:00     # 5 day and 15 minutes
#SBATCH --output=script_output/mimic3_survived_value.stdout
#SBATCH --job-name="mimic_s"
#SBATCH -p intel

# Print current date
date
# Print name of node
hostname

scripts/mimic3_survived_run.sh
