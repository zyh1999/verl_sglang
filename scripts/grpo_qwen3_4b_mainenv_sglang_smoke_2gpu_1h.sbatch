#!/usr/bin/env bash
#SBATCH --job-name=grpo_qwen3_4b_mainenv_sglang_smoke
#SBATCH --partition=workq
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err
set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME=math_grpo_qwen3_4b_mainenv_sglang_smoke_n2v2_s42_2gpu_1h_${TS}
REPO_IN_CONT=/mnt/home/verl_v0.4.x
MODEL=Qwen/Qwen3-4B

apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc "set -euo pipefail; cd ${REPO_IN_CONT}; export PYTHON_BIN=${REPO_IN_CONT}/.verl/bin/python; export PROJECT_NAME=verl_sharpness; export EXP_NAME=${EXP_NAME}; export OUT_DIR=${REPO_IN_CONT}/outputs/verl_sharpness/${EXP_NAME}; export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}; export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}; export HOME=/mnt/home/verl_cache/${SLURM_JOB_ID}/home; export XDG_CACHE_HOME=/mnt/home/verl_cache/${SLURM_JOB_ID}/xdg_cache; mkdir -p ${HOME} ${XDG_CACHE_HOME} ${RAY_TMPDIR}; export NGPUS_PER_NODE=2; export NNODES=1; export MODEL_PATH=${MODEL}; export OFFLOAD=False; export MAX_RESPONSE_LENGTH=1024; export PPO_EPOCHS=1; export N_RESP_PER_PROMPT=2; export VAL_N=2; bash examples/RL_math/run_qwen2.5-3b_math_grpo.sh ${MODEL} actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 actor_rollout_ref.ref.fsdp_config.dtype=bfloat16 data.max_prompt_length=1024 data.max_response_length=1024"
