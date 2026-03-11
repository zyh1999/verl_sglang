#!/usr/bin/env bash
set -euo pipefail

# Submit one GRPO job: mixed precision + original AdamW (no betas override).
# - 2 GPU, 10h
# - keep default optimizer betas from config (typically AdamW defaults), i.e. NOT beta1=beta2

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-10:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
EXCLUDE="${EXCLUDE:-}"

SEED="${SEED:-42}"
PROJECT_NAME="${PROJECT_NAME:-verl_adam}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo.sh}"

IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"

FILTER_WORKERS="${FILTER_WORKERS:-8}"
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-fp32}"
FSDP_TRAIN_DTYPE="${FSDP_TRAIN_DTYPE:-bfloat16}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
BETA1="${BETA1:-0.92}"
BETA2="${BETA2:-0.92}"
ATTN_IMPL="${ATTN_IMPL:-}"
MP_TAG="${MP_TAG:-amp_bf16_m${FSDP_MODEL_DTYPE}_b1eqb2_0p92_pe${PPO_EPOCHS}_adamw}"
MP_TAG="${MP_TAG//./p}"

REPO_IN_CONT="/mnt/home/verl_v0.4.x"
PYTHON_BIN_IN_CONT="${PYTHON_BIN_IN_CONT:-${REPO_IN_CONT}/.verl/bin/python}"

if [[ ! -f "${IMG}" ]]; then
  echo "[error] IMG not found: ${IMG}" >&2
  exit 1
fi

host_ts="$(date +%Y%m%d_%H%M%S)"
job_name="grpo_qwen25_3b_${MP_TAG}_s${SEED}"
exp_name="math_grpo_qwen25_3b_${MP_TAG}_s${SEED}_${host_ts}"
out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"

seed_overrides=(
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "critic.data_loader_seed=${SEED}"
  "trainer.val_subset_seed=${SEED}"
  "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
  "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
)

hydra_args=(
  "actor_rollout_ref.actor.log_adam_snr=True"
  "actor_rollout_ref.actor.optim.betas=[${BETA1},${BETA2}]"
  "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=${FSDP_MODEL_DTYPE}"
  "actor_rollout_ref.actor.fsdp_config.dtype=${FSDP_TRAIN_DTYPE}"
  "actor_rollout_ref.actor.fsdp_config.use_orig_params=True"
  "actor_rollout_ref.ref.fsdp_config.model_dtype=${FSDP_MODEL_DTYPE}"
  "actor_rollout_ref.ref.fsdp_config.dtype=${FSDP_TRAIN_DTYPE}"
  "actor_rollout_ref.ref.fsdp_config.use_orig_params=True"
  "${seed_overrides[@]}"
  "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
)
if [[ -n "${ATTN_IMPL}" ]]; then
  hydra_args+=("actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}")
fi
hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"

echo "============================================================"
echo "[qwen-orig-adam-mixed] partition=${PARTITION} gpus=${GPUS} time=${TIME} mem=${MEM} cpus=${CPUS}"
echo "[qwen-orig-adam-mixed] seed=${SEED} project=${PROJECT_NAME}"
echo "[qwen-orig-adam-mixed] model=${MODEL_PATH}"
echo "[qwen-orig-adam-mixed] optimizer=AdamW betas=[${BETA1},${BETA2}] log_adam_snr=True"
echo "[qwen-orig-adam-mixed] ppo_epochs=${PPO_EPOCHS}"
echo "[qwen-orig-adam-mixed] mixed_precision_tag=${MP_TAG} model_dtype=${FSDP_MODEL_DTYPE} train_dtype=${FSDP_TRAIN_DTYPE}"
echo "[qwen-orig-adam-mixed] img=${IMG}"
echo "[qwen-orig-adam-mixed] run_script=${RUN_SCRIPT}"
echo "============================================================"

mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

wrap_cmd=$(
  cat <<EOF
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
echo "[cfg] orig_adam_mixed seed=${SEED} exp_name=${exp_name}"
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd '${REPO_IN_CONT}'; export PYTHON_BIN='${PYTHON_BIN_IN_CONT}'; export PROJECT_NAME='${PROJECT_NAME}'; export MODEL_PATH='${MODEL_PATH}'; export EXP_NAME='${exp_name}'; export OUT_DIR='${out_dir}'; export FSDP_MODEL_DTYPE='${FSDP_MODEL_DTYPE}'; export CACHE_BASE='/mnt/home/verl_cache/\${SLURM_JOB_ID}'; export RAY_TMPDIR='/mnt/home/raytmp/\${SLURM_JOB_ID}'; export NGPUS_PER_NODE='${GPUS}'; export NNODES=1; bash '${RUN_SCRIPT}'${hydra_args_escaped}"
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

echo "============================================================"
echo "[submitted] job_id=${jid} name=${job_name}"
echo "Tips:"
echo "  squeue -u \$USER -n ${job_name}"
echo "  tail -f /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/${job_name}_${jid}.out"
echo "============================================================"
