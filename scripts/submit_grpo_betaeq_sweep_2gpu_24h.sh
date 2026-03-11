#!/usr/bin/env bash
set -euo pipefail

# Submit GRPO beta1=beta2 sweep (single seed each) on Slurm with 2 GPUs for 24 hours.
#
# Usage (on host):
#   bash verl_v0.4.x/scripts/submit_grpo_betaeq_sweep_2gpu_24h.sh
#
# Optional env overrides:
#   PARTITION=workq
#   GPUS=2
#   TIME=1-00:00:00
#   MEM=240G
#   CPUS=32
#   SEED=42
#   EXCLUDE=nid010230
#   PROJECT_NAME=verl_grpo_example_math
#   MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
#   RUN_SCRIPT=examples/RL_math/run_qwen2.5-3b_math_grpo_no_clip.sh
#   IMG=/scratch/u6g/zhouyihe.u6g/sglang.sif
#   BIND=/scratch/u6g/zhouyihe.u6g:/mnt/home
#
# Notes:
# - Uses Apptainer and binds /scratch/... -> /mnt/home in the container.
# - Isolates OUT_DIR/EXP_NAME per beta to avoid checkpoint/log collisions.

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-1-00:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
EXCLUDE="${EXCLUDE:-}"

SEED="${SEED:-42}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_example_math}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo_no_clip.sh}"

IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"

# Datasets filtering uses multiprocessing (SemLock). Too many workers can hit system limits
# (shown as OSError: [Errno 28] No space left on device). Keep it moderate but still parallel.
FILTER_WORKERS="${FILTER_WORKERS:-8}"

REPO_IN_CONT="/mnt/home/verl_v0.4.x"
PYTHON_BIN_IN_CONT="${PYTHON_BIN_IN_CONT:-${REPO_IN_CONT}/.verl/bin/python}"

if [[ ! -f "${IMG}" ]]; then
  echo "[error] IMG not found: ${IMG}" >&2
  exit 1
fi

host_ts="$(date +%Y%m%d_%H%M%S)"
betas=(0.95 0.97 0.99 0.995 0.999)

echo "============================================================"
echo "[beta-sweep] partition=${PARTITION} gpus=${GPUS} time=${TIME} mem=${MEM} cpus=${CPUS}"
echo "[beta-sweep] seed=${SEED} project=${PROJECT_NAME} model=${MODEL_PATH}"
echo "[beta-sweep] img=${IMG}"
echo "[beta-sweep] bind=${BIND}"
echo "[beta-sweep] run_script=${RUN_SCRIPT}"
echo "============================================================"

mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

submitted=()
for beta in "${betas[@]}"; do
  beta_tag="$(echo "${beta}" | sed 's/\./p/g')"
  job_name="grpo_b1eqb2_${beta_tag}_s${SEED}"
  exp_name="math_grpo_no_clip_b1eqb2_${beta_tag}_s${SEED}_${host_ts}"
  out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"

  # Hydra list syntax: [x,y]
  betas_override="actor_rollout_ref.actor.optim.betas=[${beta},${beta}]"

  # Seeds: data loader + validation subset + fsdp engine seed (actor/ref)
  seed_overrides=(
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "critic.data_loader_seed=${SEED}"
    "trainer.val_subset_seed=${SEED}"
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
    "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
  )

  hydra_args=(
    "${betas_override}"
    "${seed_overrides[@]}"
    "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
  )
  # Shell-escape hydra args so we can pass them through `bash -lc` safely.
  hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"

  wrap_cmd=$(
    cat <<EOF
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
echo "[cfg] beta=${beta} seed=${SEED} exp_name=${exp_name}"
# IMPORTANT:
# - Ray uses AF_UNIX sockets, whose path length must be <= 107 bytes.
# - Ensure /dev/shm is correctly available inside the container for multiprocessing SemLock.
# So we must keep RAY_TMPDIR (and thus Ray session socket paths) short.
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd '${REPO_IN_CONT}'; export PYTHON_BIN='${PYTHON_BIN_IN_CONT}'; export PROJECT_NAME='${PROJECT_NAME}'; export MODEL_PATH='${MODEL_PATH}'; export EXP_NAME='${exp_name}'; export OUT_DIR='${out_dir}'; export CACHE_BASE='/mnt/home/verl_cache/${job_name}'; export RAY_TMPDIR='/mnt/home/raytmp/${job_name}'; export NGPUS_PER_NODE='${GPUS}'; export NNODES=1; bash '${RUN_SCRIPT}'${hydra_args_escaped}"
echo "[slurm] end=\$(date)"
EOF
  )

  # Submit one job per beta
  jid="$(
    sbatch --parsable \
      --job-name="${job_name}" \
      --partition="${PARTITION}" \
      --gpus="${GPUS}" \
      --cpus-per-task="${CPUS}" \
      --mem="${MEM}" \
      --time="${TIME}" \
      ${EXCLUDE:+--exclude="${EXCLUDE}"} \
      --output="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out" \
      --error="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err" \
      --wrap "${wrap_cmd}"
  )"

  submitted+=("${jid}:${job_name}:${beta}")
  echo "[submitted] job_id=${jid} name=${job_name} beta=${beta}"
done

echo "============================================================"
echo "[done] submitted ${#submitted[@]} jobs"
printf '%s\n' "${submitted[@]}"
echo "Tips:"
echo "  squeue -u \$USER -n grpo_b1eqb2_*"
echo "  tail -f /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/grpo_b1eqb2_*_<JOBID>.out"
echo "============================================================"

