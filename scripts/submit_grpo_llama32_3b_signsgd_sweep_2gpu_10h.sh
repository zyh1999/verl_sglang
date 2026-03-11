#!/usr/bin/env bash
set -euo pipefail

# Sweep SignSGD (NO momentum) for Llama-3.2-3B-Instruct:
# - 5 learning rates: 1e-5, 3e-6, 1e-6, 5e-7, 1e-7
# - 2 GPUs, 10h
# - sdpa + params fp32 + compute bf16 + ppo_epochs=1

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-10:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
EXCLUDE="${EXCLUDE:-}"

SEED="${SEED:-42}"
PROJECT_NAME="${PROJECT_NAME:-verl_adam}"
MODEL_PATH_IN_CONT="${MODEL_PATH_IN_CONT:-/mnt/home/models/Llama-3.2-3B-Instruct}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo.sh}"

IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"

FILTER_WORKERS="${FILTER_WORKERS:-8}"
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-fp32}"
FSDP_TRAIN_DTYPE="${FSDP_TRAIN_DTYPE:-bfloat16}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

REPO_IN_CONT="/mnt/home/verl_v0.4.x"
PYTHON_BIN_IN_CONT="${PYTHON_BIN_IN_CONT:-${REPO_IN_CONT}/.verl/bin/python}"

if [[ ! -f "${IMG}" ]]; then
  echo "[error] IMG not found: ${IMG}" >&2
  exit 1
fi

lrs=(1e-5 3e-6 1e-6 5e-7 1e-7)

mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

for lr in "${lrs[@]}"; do
  host_ts="$(date +%Y%m%d_%H%M%S)"
  lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/m}"
  mp_tag="sdpa_amp_bf16_mfp32_SignSGD_llama32_lr${lr_tag}_pe${PPO_EPOCHS}"

  job_name="grpo_llama32_3b_${mp_tag}_s${SEED}"
  exp_name="math_grpo_llama32_3b_${mp_tag}_s${SEED}_${host_ts}"
  out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"

  seed_overrides=(
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "critic.data_loader_seed=${SEED}"
    "trainer.val_subset_seed=${SEED}"
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
    "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
  )

  hydra_args=(
    "actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.fsdp_config.model_dtype=${FSDP_MODEL_DTYPE}"
    "actor_rollout_ref.actor.fsdp_config.dtype=${FSDP_TRAIN_DTYPE}"
    "actor_rollout_ref.actor.fsdp_config.use_orig_params=True"
    "actor_rollout_ref.ref.fsdp_config.model_dtype=${FSDP_MODEL_DTYPE}"
    "actor_rollout_ref.ref.fsdp_config.dtype=${FSDP_TRAIN_DTYPE}"
    "actor_rollout_ref.ref.fsdp_config.use_orig_params=True"

    "actor_rollout_ref.actor.optim.optimizer=SignSGD"
    "actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.sign_sgd"
    "actor_rollout_ref.actor.optim.lr=${lr}"

    "${seed_overrides[@]}"
    "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
  )
  hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"

  echo "------------------------------------------------------------"
  echo "[submit] model=llama32_3b opt=SignSGD lr=${lr} ppo_epochs=${PPO_EPOCHS} attn=${ATTN_IMPL} m=${FSDP_MODEL_DTYPE} d=${FSDP_TRAIN_DTYPE}"
  echo "[submit] job_name=${job_name}"

  wrap_cmd=$(
    cat <<EOF
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
echo "[cfg] model=llama32_3b opt=SignSGD lr=${lr} exp_name=${exp_name}"
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
  echo "[submitted] job_id=${jid}"
done

echo "------------------------------------------------------------"
echo "[done] submitted ${#lrs[@]} jobs"
echo "Tip: squeue -u \$USER -o '%.18i %.70j %.8T %.10M %.10l %R' | awk '/SignSGD_llama32/'"
echo "------------------------------------------------------------"

