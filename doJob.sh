#!/bin/bash
#SBATCH --job-name=CE-learn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1g
#SBATCH --time=8:00:00
#SBATCH --account=egourgou
#SBATCH --partition=standard
#SBATCH --mail-type=NONE

python optimize_params.py run --rootdir $1
