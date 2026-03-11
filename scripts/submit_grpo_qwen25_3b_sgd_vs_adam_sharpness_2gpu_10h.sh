#!/usr/bin/env bash
set -euo pipefail

# Submit two Qwen2.5-3B GRPO runs for sharpness study:
# 1) SGD   lr=1e-1
# 2) AdamW lr=1e-6 (default AdamW betas from config)
#
# Shared setup:
# - 2 GPU, 10h
# - wandb project_name: LLM-sharpness
# - critical sharpness logging enabled

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-10:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
EXCLUDE="${EXCLUDE:-}"

SEED="${SEED:-42}"
PROJECT_NAME="${PROJECT_NAME:-LLM-sharpness}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo.sh}"

IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"

FILTER_WORKERS="${FILTER_WORKERS:-8}"
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-fp32}"
FSDP_TRAIN_DTYPE="${FSDP_TRAIN_DTYPE:-bfloat16}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
CRITICAL_INTERVAL="${CRITICAL_INTERVAL:-10}"
RAY_BASE_DIR="${RAY_BASE_DIR:-/mnt/home/raytmp}"
CACHE_BASE_DIR="${CACHE_BASE_DIR:-/mnt/home/verl_cache}"

REPO_IN_CONT="/mnt/home/verl_v0.4.x"
PYTHON_BIN_IN_CONT="${PYTHON_BIN_IN_CONT:-${REPO_IN_CONT}/.verl/bin/python}"

if [[ ! -f "${IMG}" ]]; then
  echo "[error] IMG not found: ${IMG}" >&2
  exit 1
fi

mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

host_ts="$(date +%Y%m%d_%H%M%S)"

seed_overrides=(
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "critic.data_loader_seed=${SEED}"
  "trainer.val_subset_seed=${SEED}"
  "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
  "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
)

submit_one() {
  local opt_tag="$1"      # sgd / adamw
  local lr="$2"           # e.g. 1e-1
  local opt_name="$3"     # SGD / AdamW
  local opt_impl="$4"     # torch.optim

  local lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/m}"
  local mp_tag="qwen25_3b_${opt_tag}_lr${lr_tag}_sdpa_amp_bf16_mfp32_pe${PPO_EPOCHS}_sharp"
  local job_name="grpo_${mp_tag}_s${SEED}"
  local exp_name="math_grpo_${mp_tag}_s${SEED}_${host_ts}"
  local out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"

  local hydra_args=(
    "actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.fsdp_config.model_dtype=${FSDP_MODEL_DTYPE}"
    "actor_rollout_ref.actor.fsdp_config.dtype=${FSDP_TRAIN_DTYPE}"
    "actor_rollout_ref.actor.fsdp_config.use_orig_params=True"
    "actor_rollout_ref.ref.fsdp_config.model_dtype=${FSDP_MODEL_DTYPE}"
    "actor_rollout_ref.ref.fsdp_config.dtype=${FSDP_TRAIN_DTYPE}"
    "actor_rollout_ref.ref.fsdp_config.use_orig_params=True"
    "actor_rollout_ref.actor.optim.optimizer=${opt_name}"
    "actor_rollout_ref.actor.optim.optimizer_impl=${opt_impl}"
    "actor_rollout_ref.actor.optim.lr=${lr}"
    "+actor_rollout_ref.actor.log_critical_sharpness=True"
    "+actor_rollout_ref.actor.critical_sharpness_interval=${CRITICAL_INTERVAL}"
    "${seed_overrides[@]}"
    "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
  )
  local hydra_args_escaped
  hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"

  echo "------------------------------------------------------------"
  echo "[submit] opt=${opt_name} lr=${lr} model=${MODEL_PATH}"
  echo "[submit] job_name=${job_name}"

  local wrap_cmd
  wrap_cmd=$(
    cat <<EOF
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
echo "[cfg] model=qwen25_3b opt=${opt_name} lr=${lr} exp_name=${exp_name}"
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd '${REPO_IN_CONT}'; export PYTHON_BIN='${PYTHON_BIN_IN_CONT}'; export PROJECT_NAME='${PROJECT_NAME}'; export MODEL_PATH='${MODEL_PATH}'; export EXP_NAME='${exp_name}'; export OUT_DIR='${out_dir}'; export CACHE_BASE=\"${CACHE_BASE_DIR}/\${SLURM_JOB_ID}\"; export RAY_TMPDIR=\"${RAY_BASE_DIR}/\${SLURM_JOB_ID}\"; mkdir -p \"\\\${CACHE_BASE}\" \"\\\${RAY_TMPDIR}\"; export NGPUS_PER_NODE='${GPUS}'; export NNODES=1; bash '${RUN_SCRIPT}'${hydra_args_escaped}"
echo "[slurm] end=\$(date)"
EOF
  )

  local jid
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
  echo "[submitted] job_id=${jid} name=${job_name}"
}

submit_one "sgd" "1e-1" "SGD" "torch.optim"
submit_one "adamw" "1e-6" "AdamW" "torch.optim"

echo "------------------------------------------------------------"
echo "[done] submitted 2 jobs. project_name=${PROJECT_NAME}"
echo "Tip: squeue -u \$USER -o '%.18i %.70j %.8T %.10M %.10l %R' | awk '/qwen25_3b_.*_sharp/'"
echo "------------------------------------------------------------"
