#!/usr/bin/env bash
#SBATCH --job-name=grpo_q25_math_signum_pe1_sdpa
#SBATCH --partition=workq
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err
set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
# 按你的命名要求：signum + lr + pe1 + sdpa
EXP_NAME=math_grpo_qwen25_3b_sdpa_amp_bf16_mfp32_signum_lr1e-6_pe1_${TS}
REPO_IN_CONT=/mnt/home/verl_v0.4.x
apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc "set -euo pipefail; cd ${REPO_IN_CONT}; export PYTHON_BIN=${REPO_IN_CONT}/.verl/bin/python; export PROJECT_NAME=adam_verl; export EXP_NAME=${EXP_NAME}; export OUT_DIR=${REPO_IN_CONT}/outputs/adam_verl/${EXP_NAME}; export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}; export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}; export NGPUS_PER_NODE=2; export NNODES=1; export PPO_EPOCHS=1; bash examples/RL_math/run_qwen2.5-3b_math_grpo.sh Qwen/Qwen2.5-3B-Instruct actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.sign_sgd actor_rollout_ref.actor.optim.optimizer=Signum actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.override_optimizer_config.momentum=0.9 actor_rollout_ref.model.override_config.attn_implementation=sdpa"
