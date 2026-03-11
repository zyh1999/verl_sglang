#!/usr/bin/env bash
#SBATCH --job-name=grpo_qwen3_1p7b_mainenv_hf_smoke
#SBATCH --partition=workq
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err
set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME=math_grpo_qwen3_1p7b_mainenv_hf_smoke_n2v2_s42_2gpu_1h_${TS}
REPO_IN_CONT=/mnt/home/verl_v0.4.x
MODEL=/mnt/home/verl_cache/2536592/.hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc "set -euo pipefail; cd ${REPO_IN_CONT}; export PYTHON_BIN=${REPO_IN_CONT}/.verl/bin/python; export PROJECT_NAME=verl_sharpness; export EXP_NAME=${EXP_NAME}; export OUT_DIR=${REPO_IN_CONT}/outputs/verl_sharpness/${EXP_NAME}; export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}; export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}; export NGPUS_PER_NODE=2; export NNODES=1; export MODEL_PATH=${MODEL}; export OFFLOAD=False; export MAX_RESPONSE_LENGTH=1024; export PPO_EPOCHS=1; export N_RESP_PER_PROMPT=2; export VAL_N=2; bash examples/RL_math/run_qwen2.5-3b_math_grpo.sh ${MODEL} actor_rollout_ref.rollout.name=hf actor_rollout_ref.rollout.mode=sync actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 actor_rollout_ref.ref.fsdp_config.dtype=bfloat16"
