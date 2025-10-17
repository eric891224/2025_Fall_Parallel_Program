#!/bin/bash
#SBATCH --job-name=hw2_analysis
#SBATCH --account=ACD114118
#SBATCH --partition=ctest
#SBATCH --nodes=6                    # Request max nodes you need
#SBATCH --ntasks=12                  # Request max tasks you need
#SBATCH --cpus-per-task=32           # Request max CPUs per task
#SBATCH --time=01:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log

# The srun commands inside your script will use subsets of this allocation
bash srun_analysis.sh