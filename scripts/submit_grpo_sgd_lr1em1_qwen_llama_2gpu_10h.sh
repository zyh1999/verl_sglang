#!/usr/bin/env bash
set -euo pipefail

# Submit 2 one-off SGD runs (no momentum), lr=1e-1:
# - Qwen2.5-3B-Instruct
# - Llama-3.2-3B-Instruct
#
# Shared setup:
# - 2 GPU, 10h
# - ppo_epochs=1
# - sdpa
# - params fp32 + compute bf16

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-10:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
EXCLUDE="${EXCLUDE:-}"

SEED="${SEED:-42}"
PROJECT_NAME="${PROJECT_NAME:-verl_adam}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo.sh}"
LR="${LR:-1e-1}"

IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"

FILTER_WORKERS="${FILTER_WORKERS:-8}"
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-fp32}"
FSDP_TRAIN_DTYPE="${FSDP_TRAIN_DTYPE:-bfloat16}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
RUN_QWEN="${RUN_QWEN:-1}"
RUN_LLAMA="${RUN_LLAMA:-1}"
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
lr_tag="${LR//./p}"
lr_tag="${lr_tag//-/m}"

seed_overrides=(
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "critic.data_loader_seed=${SEED}"
  "trainer.val_subset_seed=${SEED}"
  "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
  "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
)

submit_one() {
  local model_tag="$1"
  local model_path="$2"

  local mp_tag="sdpa_amp_bf16_mfp32_SGD_nomom_lr${lr_tag}_pe${PPO_EPOCHS}_${model_tag}"
  local job_name="grpo_${model_tag}_${mp_tag}_s${SEED}"
  local exp_name="math_grpo_${model_tag}_${mp_tag}_s${SEED}_${host_ts}"
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

    "actor_rollout_ref.actor.optim.optimizer=SGD"
    "actor_rollout_ref.actor.optim.optimizer_impl=torch.optim"
    "actor_rollout_ref.actor.optim.lr=${LR}"

    "${seed_overrides[@]}"
    "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
  )
  local hydra_args_escaped
  hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"

  echo "------------------------------------------------------------"
  echo "[submit] model=${model_tag} opt=SGD(no-momentum) lr=${LR} ppo_epochs=${PPO_EPOCHS} attn=${ATTN_IMPL} m=${FSDP_MODEL_DTYPE} d=${FSDP_TRAIN_DTYPE}"
  echo "[submit] job_name=${job_name}"

  local wrap_cmd
  wrap_cmd=$(
    cat <<EOF
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
echo "[cfg] model=${model_tag} opt=SGD(no-momentum) lr=${LR} exp_name=${exp_name}"
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd '${REPO_IN_CONT}'; export PYTHON_BIN='${PYTHON_BIN_IN_CONT}'; export PROJECT_NAME='${PROJECT_NAME}'; export EXP_NAME='${exp_name}'; export OUT_DIR='${out_dir}'; export CACHE_BASE=\"${CACHE_BASE_DIR}/\${SLURM_JOB_ID}\"; export RAY_TMPDIR=\"${RAY_BASE_DIR}/\${SLURM_JOB_ID}\"; mkdir -p \"\\\${CACHE_BASE}\" \"\\\${RAY_TMPDIR}\"; echo \"[runtime] CACHE_BASE=\\\${CACHE_BASE}\"; echo \"[runtime] RAY_TMPDIR=\\\${RAY_TMPDIR}\"; export NGPUS_PER_NODE='${GPUS}'; export NNODES=1; bash '${RUN_SCRIPT}' '${model_path}'${hydra_args_escaped}"
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
  echo "[submitted] job_id=${jid}"
}

if [[ "${RUN_QWEN}" == "1" ]]; then
  submit_one "qwen25_3b" "Qwen/Qwen2.5-3B-Instruct"
fi
if [[ "${RUN_LLAMA}" == "1" ]]; then
  submit_one "llama32_3b" "/mnt/home/models/Llama-3.2-3B-Instruct"
fi

echo "------------------------------------------------------------"
echo "[done] submit finished (RUN_QWEN=${RUN_QWEN}, RUN_LLAMA=${RUN_LLAMA})"
echo "Tip: squeue -u \$USER -o '%.18i %.70j %.8T %.10M %.10l %R' | awk '/SGD_nomom_lr1em1/'"
echo "------------------------------------------------------------"

