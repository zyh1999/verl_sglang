#!/usr/bin/env bash
set -euo pipefail

# Submit 2 jobs (clip / no_clip) for Qwen2.5-3B base (non-instruct) on Math GRPO.
#
# Usage:
#   bash verl_v0.4.x/scripts/submit_grpo_qwen25_3b_base_clip_noclip_2gpu_12h.sh
#
# Optional env overrides:
#   PARTITION=workq
#   GPUS=2
#   TIME=12:00:00
#   MEM=240G
#   CPUS=32
#   SEED=42
#   EXCLUDE=nid010230
#   PROJECT_NAME=verl_grpo_example_math
#   IMG=/scratch/u6g/zhouyihe.u6g/sglang.sif
#   BIND=/scratch/u6g/zhouyihe.u6g:/mnt/home
#   SHM_BIND=/dev/shm:/dev/shm
#   FILTER_WORKERS=8
#   MODEL_PATH_HOST=/scratch/... (host path)
#   MODEL_PATH_IN_CONT=/mnt/home/... (container path)

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-12:00:00}"
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

if [[ ! -f "${IMG}" ]]; then
  echo "[error] IMG not found: ${IMG}" >&2
  exit 1
fi

# Resolve local model path (prefer explicit overrides; otherwise auto-detect newest HF snapshot).
MODEL_PATH_HOST="${MODEL_PATH_HOST:-}"
MODEL_PATH_IN_CONT="${MODEL_PATH_IN_CONT:-}"
if [[ -z "${MODEL_PATH_HOST}" && -z "${MODEL_PATH_IN_CONT}" ]]; then
  snap_root="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/.hf/hub/models--Qwen--Qwen2.5-3B/snapshots"
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

mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs
host_ts="$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "[submit] partition=${PARTITION} gpus=${GPUS} time=${TIME} mem=${MEM} cpus=${CPUS}"
echo "[submit] seed=${SEED} project=${PROJECT_NAME}"
echo "[submit] img=${IMG}"
echo "[submit] bind=${BIND}"
echo "[submit] model_path_in_cont=${MODEL_PATH_IN_CONT}"
echo "============================================================"

submit_one () {
  local mode="$1"       # clip | no_clip
  local run_script="$2" # examples/RL_math/*.sh

  local job_name="grpo_qwen25_3b_base_${mode}_s${SEED}"
  local exp_name="math_grpo_qwen25_3b_base_${mode}_s${SEED}_${host_ts}"
  local out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"

  local seed_overrides=(
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "critic.data_loader_seed=${SEED}"
    "trainer.val_subset_seed=${SEED}"
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
    "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
  )

  local hydra_args=(
    "${seed_overrides[@]}"
    "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
  )
  local hydra_args_escaped
  hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"

  local wrap_cmd
  wrap_cmd=$(
    cat <<EOF
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
echo "[cfg] mode=${mode} seed=${SEED} exp_name=${exp_name}"
# Ray 的 AF_UNIX socket path 长度上限是 107 字节；因此这里强制用非常短的目录名（用 job_id）
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd '${REPO_IN_CONT}'; export PYTHON_BIN='${PYTHON_BIN_IN_CONT}'; export PROJECT_NAME='${PROJECT_NAME}'; export EXP_NAME='${exp_name}'; export OUT_DIR='${out_dir}'; export CACHE_BASE='/mnt/home/verl_cache/\${SLURM_JOB_ID}'; export RAY_TMPDIR='/mnt/home/raytmp/\${SLURM_JOB_ID}'; export NGPUS_PER_NODE='${GPUS}'; export NNODES=1; bash '${run_script}' '${MODEL_PATH_IN_CONT}'${hydra_args_escaped}"
echo "[slurm] end=\$(date)"
EOF
  )

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
}

jid_clip="$(submit_one "clip" "examples/RL_math/run_qwen2.5-3b_math_grpo.sh")"
echo "[submitted] clip job_id=${jid_clip}"

jid_noclip="$(submit_one "no_clip" "examples/RL_math/run_qwen2.5-3b_math_grpo_no_clip.sh")"
echo "[submitted] no_clip job_id=${jid_noclip}"

echo "============================================================"
echo "[done] submitted 2 jobs"
echo "  clip    : ${jid_clip}"
echo "  no_clip : ${jid_noclip}"
echo "Tips:"
echo "  squeue -u \$USER -n grpo_qwen25_3b_base_*"
echo "============================================================"

