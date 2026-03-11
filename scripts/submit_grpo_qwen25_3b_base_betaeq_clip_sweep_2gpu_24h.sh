#!/usr/bin/env bash
set -euo pipefail

# Submit GRPO beta1=beta2 sweep (single seed each) for a given model (e.g. Llama/Qwen) + CLIP
# on Slurm with 2 GPUs for 24 hours.
#
# Usage (on host):
#   bash verl_v0.4.x/scripts/submit_grpo_qwen25_3b_base_betaeq_clip_sweep_2gpu_24h.sh
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
#   IMG=/scratch/u6g/zhouyihe.u6g/sglang.sif
#   BIND=/scratch/u6g/zhouyihe.u6g:/mnt/home
#   SHM_BIND=/dev/shm:/dev/shm
#   FILTER_WORKERS=8
#   MODEL_PATH_HOST=/scratch/... (host path to local snapshot)
#   MODEL_PATH_IN_CONT=/mnt/home/... (container path)
#   MODEL_TAG=llama32_3b_instruct (used in exp_name/job_name; default inferred from model dir name)
#   BETAS="0.92" (space-separated list; overrides default sweep betas)
#   ACTOR_LRS="1e-6 2e-6" (space-separated list; if set, submits one job per lr per beta)
#
# Notes:
# - We keep Ray tmp dir SHORT (use SLURM_JOB_ID) to avoid AF_UNIX socket path length limit (107 bytes).
# - We bind /dev/shm into the container for HuggingFace datasets multiprocessing SemLock.

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-1-00:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
EXCLUDE="${EXCLUDE:-}"

SEED="${SEED:-42}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_example_math}"

IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"

FILTER_WORKERS="${FILTER_WORKERS:-8}"

REPO_IN_CONT="/mnt/home/verl_v0.4.x"
PYTHON_BIN_IN_CONT="${PYTHON_BIN_IN_CONT:-${REPO_IN_CONT}/.verl/bin/python}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo.sh}"

if [[ ! -f "${IMG}" ]]; then
  echo "[error] IMG not found: ${IMG}" >&2
  exit 1
fi

# Resolve local model path (prefer explicit overrides; otherwise auto-detect newest HF snapshot).
MODEL_PATH_HOST="${MODEL_PATH_HOST:-}"
MODEL_PATH_IN_CONT="${MODEL_PATH_IN_CONT:-}"
if [[ -z "${MODEL_PATH_HOST}" && -z "${MODEL_PATH_IN_CONT}" ]]; then
  snap_root="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/.hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots"
  newest_snap="$(ls -1t "${snap_root}" 2>/dev/null | head -n 1 || true)"
  if [[ -n "${newest_snap}" ]]; then
    MODEL_PATH_HOST="${snap_root}/${newest_snap}"
  fi
fi
if [[ -z "${MODEL_PATH_IN_CONT}" ]]; then
  if [[ -n "${MODEL_PATH_HOST}" ]]; then
    MODEL_PATH_IN_CONT="${MODEL_PATH_HOST/\/scratch\/u6g\/zhouyihe.u6g/\/mnt\/home}"
  else
    echo "[error] cannot determine model path. Set MODEL_PATH_HOST or MODEL_PATH_IN_CONT." >&2
    exit 1
  fi
fi

# Build a short, filesystem- and W&B-friendly model tag for exp/job names.
MODEL_TAG="${MODEL_TAG:-}"
if [[ -z "${MODEL_TAG}" ]]; then
  # Prefer host path base name if available, otherwise container path base name.
  _model_base="${MODEL_PATH_HOST:-${MODEL_PATH_IN_CONT}}"
  _model_base="${_model_base##*/}"
  # Normalize: lower + replace non [a-z0-9._-] with underscore.
  MODEL_TAG="$(echo "${_model_base}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/_/g')"
  MODEL_TAG="${MODEL_TAG:-model}"
fi

mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs
host_ts="$(date +%Y%m%d_%H%M%S)"

# Requested betas
# - Default sweep uses 5 values below
# - Override by setting env: BETAS="0.92" or BETAS="0.90 0.92 0.95"
if [[ -n "${BETAS:-}" ]]; then
  read -r -a betas <<< "${BETAS}"
else
  betas=(0.90 0.92 0.95 0.97 0.99)
fi

# Actor learning rates (optional sweep dimension)
ACTOR_LRS="${ACTOR_LRS:-}"
if [[ -n "${ACTOR_LRS}" ]]; then
  read -r -a actor_lrs <<< "${ACTOR_LRS}"
else
  actor_lrs=("default")
fi

echo "============================================================"
echo "[beta-sweep][clip][${MODEL_TAG}] partition=${PARTITION} gpus=${GPUS} time=${TIME} mem=${MEM} cpus=${CPUS}"
echo "[beta-sweep] seed=${SEED} project=${PROJECT_NAME}"
echo "[beta-sweep] img=${IMG}"
echo "[beta-sweep] bind=${BIND}"
echo "[beta-sweep] run_script=${RUN_SCRIPT}"
echo "[beta-sweep] model_path_in_cont=${MODEL_PATH_IN_CONT}"
echo "[beta-sweep] actor_lrs=${ACTOR_LRS:-<default>}"
echo "============================================================"

submitted=()
for beta in "${betas[@]}"; do
  for lr in "${actor_lrs[@]}"; do
    beta_tag="$(echo "${beta}" | sed 's/\./p/g')"

    lr_tag=""
    lr_override=""
    if [[ "${lr}" != "default" ]]; then
      # Normalize lr string for names: 1e-6 -> 1em6, 2.5e-6 -> 2p5em6
      lr_tag="$(echo "${lr}" | sed -E 's/\\./p/g; s/-/m/g; s/\\+/p/g')"
      lr_tag="_lr${lr_tag}"
      lr_override="actor_rollout_ref.actor.optim.lr=${lr}"
    fi

    job_name="grpo_${MODEL_TAG}_clip_b1eqb2_${beta_tag}${lr_tag}_s${SEED}"
    exp_name="math_grpo_${MODEL_TAG}_clip_b1eqb2_${beta_tag}${lr_tag}_s${SEED}_${host_ts}"
    out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"

    # Hydra list syntax: [x,y]
    betas_override="actor_rollout_ref.actor.optim.betas=[${beta},${beta}]"

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
    if [[ -n "${lr_override}" ]]; then
      hydra_args+=("${lr_override}")
    fi
    hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"

    wrap_cmd=$(
      cat <<EOF
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
echo "[cfg] beta=${beta} actor_lr=${lr} seed=${SEED} exp_name=${exp_name}"
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd '${REPO_IN_CONT}'; export PYTHON_BIN='${PYTHON_BIN_IN_CONT}'; export PROJECT_NAME='${PROJECT_NAME}'; export EXP_NAME='${exp_name}'; export OUT_DIR='${out_dir}'; export CACHE_BASE='/mnt/home/verl_cache/\${SLURM_JOB_ID}'; export RAY_TMPDIR='/mnt/home/raytmp/\${SLURM_JOB_ID}'; export NGPUS_PER_NODE='${GPUS}'; export NNODES=1; bash '${RUN_SCRIPT}' '${MODEL_PATH_IN_CONT}'${hydra_args_escaped}"
echo "[slurm] end=\$(date)"
EOF
    )

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

    submitted+=("${jid}:${job_name}:${beta}:${lr}")
    echo "[submitted] job_id=${jid} name=${job_name} beta=${beta} actor_lr=${lr}"
  done
done

echo "============================================================"
echo "[done] submitted ${#submitted[@]} jobs"
printf '%s\n' "${submitted[@]}"
echo "Tips:"
echo "  squeue -u \$USER -n grpo_${MODEL_TAG}_clip_b1eqb2_*"
echo "============================================================"

