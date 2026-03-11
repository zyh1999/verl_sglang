#!/usr/bin/env bash
#SBATCH --job-name=grpo_math_qwen25_3b_default_adam_8h
#SBATCH --partition=workq
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err

set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME=math_grpo_qwen25_3b_sdpa_default_adam_s42_2gpu_8h_${TS}
REPO_IN_CONT=/mnt/home/verl_v0.4.x
mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

echo "[slurm] job=${SLURM_JOB_ID} host=$(hostname) start=$(date)"
apptainer exec --nv \
  -B /scratch/u6g/zhouyihe.u6g:/mnt/home \
  -B /dev/shm:/dev/shm \
  /scratch/u6g/zhouyihe.u6g/sglang.sif \
  bash -lc "set -euo pipefail; cd ${REPO_IN_CONT}; export PYTHON_BIN=${REPO_IN_CONT}/.verl/bin/python; export PROJECT_NAME=verl_adam; export MODEL_PATH=Qwen/Qwen2.5-3B-Instruct; export EXP_NAME=${EXP_NAME}; export OUT_DIR=${REPO_IN_CONT}/outputs/verl_adam/${EXP_NAME}; export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}; export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}; export NGPUS_PER_NODE=2; export NNODES=1; export SEED=42; bash examples/RL_math/run_qwen2.5-3b_math_grpo.sh"
echo "[slurm] end=$(date)"
