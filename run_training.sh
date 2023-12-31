#!/bin/sh
#SBATCH --job-name=ddpm
#SBATCH -t 24:00:00
#SBATCH -o /work/siddhantgarg_umass_edu/slurm_logs/slurm-%j.out  # %j = job ID
#SBATCH --partition=gypsum-rtx8000
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem 30GB

python unet.py