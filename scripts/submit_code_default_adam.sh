#!/usr/bin/env bash
set -euo pipefail
PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-10:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"
REPO_IN_CONT="${REPO_IN_CONT:-/mnt/home/verl_v0.4.x}"
PYTHON_BIN_IN_CONT="${PYTHON_BIN_IN_CONT:-${REPO_IN_CONT}/.verl/bin/python}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_example_code}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_code/run_qwen2.5-3b_code_grpo_no_IS.sh}"
host_ts="$(date +%Y%m%d_%H%M%S)"
job_name="grpo_code_default_adam"
exp_name="qwen25_code_default_adam_${host_ts}"
out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"
mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs
wrap_cmd=$(cat <<WRAP
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd ; export PYTHON_BIN=; export PROJECT_NAME=; export MODEL_PATH=; export EXP_NAME=; export OUT_DIR=; export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}; export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}; export NGPUS_PER_NODE=; export NNODES=1; bash "
echo "[slurm] end=\$(date)"
WRAP
)
sbatch --parsable \
  --job-name="${job_name}" \
  --partition="${PARTITION}" \
  --gpus="${GPUS}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --output="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out" \
  --error="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err" \
  --wrap "${wrap_cmd}"
