#!/usr/bin/env bash
#SBATCH --job-name=smoke_qwen25_3b_vllm
#SBATCH --partition=workq
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err
set -euo pipefail
TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME=smoke_qwen25_3b_vllm_${TS}
REPO_IN_CONT=/mnt/home/verl_v0.4.x
apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc "set -euo pipefail; cd ${REPO_IN_CONT}; export PYTHON_BIN=${REPO_IN_CONT}/.verl_vllm/bin/python; export PROJECT_NAME=verl_math_cot; export EXP_NAME=${EXP_NAME}; export OUT_DIR=${REPO_IN_CONT}/outputs/verl_math_cot/${EXP_NAME}; export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}; export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}; export NGPUS_PER_NODE=2; export NNODES=1; export MATH_TRAIN_INPUT=${REPO_IN_CONT}/data/numinamath_cot_35k/train.parquet; export MAX_RESPONSE_LENGTH=512; export ROLLOUT_NAME=vllm; bash examples/RL_math/run_qwen2.5-3b_math_grpo.sh Qwen/Qwen2.5-3B-Instruct actor_rollout_ref.model.override_config.attn_implementation=sdpa actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.val_kwargs.n=2 trainer.total_epochs=1 trainer.test_freq=5 trainer.save_freq=200"
