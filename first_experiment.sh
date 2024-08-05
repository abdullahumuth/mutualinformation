#!/bin/bash
#SBATCH --job-name=mutual_info_second
#SBATCH --output=./data/outputs/log_file
#SBATCH --error=./data/outputs/log_file
#SBATCH -p pgi-8-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1

srun julia ./src/main.jl

