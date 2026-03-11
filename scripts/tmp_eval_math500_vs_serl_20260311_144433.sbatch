#!/bin/bash
#SBATCH --job-name=cmp_math500_serl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH -p gpu
#SBATCH -o slurm_logs/%x_%j.out

set -euo pipefail
REPO=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x
cd ${REPO}
mkdir -p slurm_logs

DATASET_TAG=${DATASET_TAG:?}
VAL_FILE=${VAL_FILE:?}
EXP_NAME=cmp_${DATASET_TAG}_${SLURM_JOB_ID}

apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc 
