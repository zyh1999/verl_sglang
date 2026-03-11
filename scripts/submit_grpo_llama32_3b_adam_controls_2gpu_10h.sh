#!/usr/bin/env bash
set -euo pipefail

# Submit two AdamW control runs for Llama-3.2-3B-Instruct:
# 1) default AdamW betas (0.9, 0.999)
# 2) beta1=beta2=0.92
#
# Fixed setup:
# - 2 GPU, 10h
# - ppo_epochs=1
# - sdpa
# - params fp32 + compute bf16

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-15:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
EXCLUDE="${EXCLUDE:-}"

SEED="${SEED:-42}"
PROJECT_NAME="${PROJECT_NAME:-verl_adam}"
MODEL_PATH_IN_CONT="${MODEL_PATH_IN_CONT:-/mnt/home/models/Llama-3.2-3B-Instruct}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo.sh}"
ACTOR_LR="${ACTOR_LR:-1e-6}"

IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"

FILTER_WORKERS="${FILTER_WORKERS:-8}"
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-fp32}"
FSDP_TRAIN_DTYPE="${FSDP_TRAIN_DTYPE:-bfloat16}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
# Resume from ckpt: set RESUME_FROM_CKPT=true to inherit ckpt + wandb run
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-false}"
EXP_NAME_DEFAULTB="${EXP_NAME_DEFAULTB:-math_grpo_llama32_3b_sdpa_amp_bf16_mfp32_AdamW_defaultb_llama32_lr1em6_pe1_s42_20260227_232722}"
EXP_NAME_B1EQB2="${EXP_NAME_B1EQB2:-math_grpo_llama32_3b_sdpa_amp_bf16_mfp32_AdamW_b1eqb2_0p92_llama32_lr1em6_pe1_s42_20260227_232722}"
WANDB_RUN_ID_DEFAULTB="${WANDB_RUN_ID_DEFAULTB:-hpu6n6t2}"
WANDB_RUN_ID_B1EQB2="${WANDB_RUN_ID_B1EQB2:-r4121rma}"
WANDB_RESUME_MODE="${WANDB_RESUME_MODE:-allow}"
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
lr_tag="${ACTOR_LR//./p}"
lr_tag="${lr_tag//-/m}"

seed_overrides=(
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "critic.data_loader_seed=${SEED}"
  "trainer.val_subset_seed=${SEED}"
  "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
  "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
)

submit_one() {
  local tag="$1"
  local beta1="$2"
  local beta2="$3"
  local resume_exp_name=""
  local wandb_run_id=""

  if [[ "${RESUME_FROM_CKPT}" == "true" ]]; then
    if [[ "${tag}" == "AdamW_defaultb" ]]; then
      resume_exp_name="${EXP_NAME_DEFAULTB}"
      wandb_run_id="${WANDB_RUN_ID_DEFAULTB}"
    elif [[ "${tag}" == "AdamW_b1eqb2_0p92" ]]; then
      resume_exp_name="${EXP_NAME_B1EQB2}"
      wandb_run_id="${WANDB_RUN_ID_B1EQB2}"
    fi
  fi

  local mp_tag="sdpa_amp_bf16_mfp32_${tag}_llama32_lr${lr_tag}_pe${PPO_EPOCHS}"
  local job_name="grpo_llama32_3b_${mp_tag}_s${SEED}"
  local exp_name
  if [[ -n "${resume_exp_name}" ]]; then
    exp_name="${resume_exp_name}"
  else
    exp_name="math_grpo_llama32_3b_${mp_tag}_s${SEED}_${host_ts}"
  fi
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

    "actor_rollout_ref.actor.optim.optimizer=AdamW"
    "actor_rollout_ref.actor.optim.optimizer_impl=torch.optim"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.optim.betas=[${beta1},${beta2}]"

    "${seed_overrides[@]}"
    "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
  )
  local hydra_args_escaped
  hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"

  local wandb_exports=""
  if [[ -n "${wandb_run_id}" ]]; then
    wandb_exports="export WANDB_RUN_ID='${wandb_run_id}'; "
    [[ -n "${WANDB_RESUME_MODE}" ]] && wandb_exports+="export WANDB_RESUME='${WANDB_RESUME_MODE}'; "
  fi

  echo "------------------------------------------------------------"
  echo "[submit] model=llama32_3b opt=AdamW tag=${tag} lr=${ACTOR_LR} betas=[${beta1},${beta2}]"
  echo "[submit] job_name=${job_name}"

  local wrap_cmd
  wrap_cmd=$(
    cat <<EOF
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
echo "[cfg] model=llama32_3b opt=AdamW tag=${tag} lr=${ACTOR_LR} betas=[${beta1},${beta2}] exp_name=${exp_name}"
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd '${REPO_IN_CONT}'; export PYTHON_BIN='${PYTHON_BIN_IN_CONT}'; export PROJECT_NAME='${PROJECT_NAME}'; export EXP_NAME='${exp_name}'; export OUT_DIR='${out_dir}'; export CACHE_BASE=\"${CACHE_BASE_DIR}/\${SLURM_JOB_ID}\"; export RAY_TMPDIR=\"${RAY_BASE_DIR}/\${SLURM_JOB_ID}\"; mkdir -p \"\\\${CACHE_BASE}\" \"\\\${RAY_TMPDIR}\"; echo \"[runtime] CACHE_BASE=\\\${CACHE_BASE}\"; echo \"[runtime] RAY_TMPDIR=\\\${RAY_TMPDIR}\"; export NGPUS_PER_NODE='${GPUS}'; export NNODES=1; ${wandb_exports}bash '${RUN_SCRIPT}' '${MODEL_PATH_IN_CONT}'${hydra_args_escaped}"
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

submit_one "AdamW_defaultb" "0.9" "0.999"
submit_one "AdamW_b1eqb2_0p92" "0.92" "0.92"

echo "------------------------------------------------------------"
echo "[done] submitted 2 control jobs (ACTOR_LR=${ACTOR_LR})"
echo "Tip: squeue -u \$USER -o '%.18i %.70j %.8T %.10M %.10l %R' | awk '/AdamW_.*llama32/'"
echo "------------------------------------------------------------"

