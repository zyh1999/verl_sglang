#!/bin/bash
#SBATCH --job-name=cmp_math500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail
cd /scratch/u6g/zhouyihe.u6g/verl_v0.4.x
EXP_NAME=cmp_math500_${SLURM_JOB_ID}
apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc "set -euo pipefail; cd /mnt/home/verl_v0.4.x; export PYTHON_BIN=/mnt/home/verl_v0.4.x/.verl/bin/python; export PROJECT_NAME=verl_eval_cmp; export EXP_NAME=${EXP_NAME}; export OUT_DIR=/mnt/home/verl_v0.4.x/outputs/verl_eval_cmp/${EXP_NAME}; export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}; export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}; export HOME=/mnt/home/verl_cache/${SLURM_JOB_ID}/home; export XDG_CACHE_HOME=/mnt/home/verl_cache/${SLURM_JOB_ID}/xdg_cache; mkdir -p /mnt/home/verl_cache/${SLURM_JOB_ID}/home /mnt/home/verl_cache/${SLURM_JOB_ID}/xdg_cache /mnt/home/raytmp/${SLURM_JOB_ID}; export NGPUS_PER_NODE=2; export NNODES=1; export MODEL_PATH=Qwen/Qwen2.5-3B-Instruct; export OFFLOAD=False; export MAX_RESPONSE_LENGTH=1024; export PPO_EPOCHS=1; export N_RESP_PER_PROMPT=2; export VAL_N=2; export TEST_FREQ=1; export TOTAL_EPOCHS=1; export VAL_SUBSET_RATIO=1.0; export TRAIN_PROMPT_BSZ=32; export TRAIN_PROMPT_MINI_BSZ=8; export MICRO_BATCH_SIZE_PER_GPU=4; export TEST_FILES=[""]; bash examples/RL_math/run_qwen2.5-3b_math_grpo.sh Qwen/Qwen2.5-3B-Instruct actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.mode=sync"
