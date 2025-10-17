#!/bin/bash
#SBATCH --job-name=hw2_analysis
#SBATCH --account=ACD114118
#SBATCH --nodes=8                    # Request max nodes you need
#SBATCH --ntasks=32                  # Request max tasks you need
#SBATCH --cpus-per-task=24           # Request max CPUs per task
#SBATCH --time=01:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --exclusive                  # Get exclusive access to nodes

# The srun commands inside your script will use subsets of this allocation
bash srun_analysis.sh