#!/bin/bash
#SBATCH --job-name=mut_inf_exp3
#SBATCH --output=./data/outputs/output%j.%N.out
#SBATCH --error=./data/outputs/error%j.%N.err
#SBATCH -p pgi-8-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1

srun julia ./src/experiment4.jl

