#!/bin/bash
#SBATCH --job-name=smk_mathtask
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail
cd /scratch/u6g/zhouyihe.u6g/verl_v0.4.x
apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc 
