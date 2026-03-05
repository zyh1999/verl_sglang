#!/usr/bin/env bash
#SBATCH --job-name=grpo_math_orig_adamw_precond
#SBATCH --partition=workq
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err
set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME=qwen25_3b_math_orig_adamw_precond_pe3_fa2_${TS}
REPO_IN_CONT=/mnt/home/verl_v0.4.x
mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

echo "[slurm] job=${SLURM_JOB_ID} host=$(hostname) start=$(date)"
apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc "set -euo pipefail; cd ${REPO_IN_CONT}; export PYTHON_BIN=${REPO_IN_CONT}/.verl/bin/python; export PROJECT_NAME=verl_sharpness; export EXP_NAME=${EXP_NAME}; export OUT_DIR=${REPO_IN_CONT}/outputs/verl_sharpness/${EXP_NAME}; export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}; export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}; export NGPUS_PER_NODE=2; export NNODES=1; export MODEL_PATH=/mnt/home/verl_cache/prewarm_qwen25_3b/.hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1; export OFFLOAD=True; export GPU_MEM_UTIL=0.2; export MAX_RESPONSE_LENGTH=1024; bash examples/RL_math/run_qwen2.5-3b_math_grpo.sh /mnt/home/verl_cache/prewarm_qwen25_3b/.hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.adamw_precond ++actor_rollout_ref.actor.optim.optimizer=AdamWPrecond ++actor_rollout_ref.actor.optim.betas=[0.92,0.92] ++actor_rollout_ref.actor.optim.override_optimizer_config.log_precond_stats=True ++actor_rollout_ref.actor.optim.override_optimizer_config.precond_stat_prefix=actor"
echo "[slurm] end=$(date)"
