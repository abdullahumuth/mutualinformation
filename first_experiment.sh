#!/bin/bash
#SBATCH --job-name=mutual_info_first_exp
#SBATCH --output=log_file
#SBATCH --error=log_file
#SBATCH -p pgi-8-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:3

srun julia src/main.jl

