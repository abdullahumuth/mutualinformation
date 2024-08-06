#!/bin/bash
#SBATCH --job-name=mutual_info_time_series
#SBATCH --output=./data/outputs/output%j.%N.out
#SBATCH --error=./data/outputs/error%j.%N.err
#SBATCH -p pgi-8-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1

srun julia ./src/main.jl

