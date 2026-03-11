#!/usr/bin/env bash
#SBATCH --job-name=grpo_math_orig_adamw_precond_qwen25_0p5b_micro1_rsp512
#SBATCH --partition=workq
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err
set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME=math_grpo_qwen25_3b_sdpa_amp_bf16_mfp32_precond_sharpness_lr1em6_pe3_s42_4gpu_${TS}
REPO_IN_CONT=/mnt/home/verl_v0.4.x
mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

echo "[slurm] job=${SLURM_JOB_ID} host=$(hostname) start=$(date)"
apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc "set -euo pipefail; cd ${REPO_IN_CONT}; export PYTHON_BIN=${REPO_IN_CONT}/.verl/bin/python; export PROJECT_NAME=verl_sharpness; export EXP_NAME=${EXP_NAME}; export OUT_DIR=${REPO_IN_CONT}/outputs/verl_sharpness/${EXP_NAME}; export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}; export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}; export NGPUS_PER_NODE=4; export NNODES=1; export MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct; export OFFLOAD=True; export GPU_MEM_UTIL=0.2; export MAX_RESPONSE_LENGTH=512; bash examples/RL_math/run_qwen2.5-3b_math_grpo.sh Qwen/Qwen2.5-0.5B-Instruct actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.adamw_precond ++actor_rollout_ref.actor.optim.optimizer=AdamWPrecond ++actor_rollout_ref.actor.optim.betas=[0.92,0.92] ++actor_rollout_ref.actor.optim.override_optimizer_config.log_precond_stats=True ++actor_rollout_ref.actor.optim.override_optimizer_config.precond_stat_prefix=actor +actor_rollout_ref.actor.precond_sharpness_interval=5 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.model.enable_gradient_checkpointing=False actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
echo "[slurm] end=$(date)"
