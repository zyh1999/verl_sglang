#!/usr/bin/env bash
set -euo pipefail
PARTITION="${PARTITION:-workq}"; GPUS="${GPUS:-2}"; TIME="${TIME:-15:00:00}"; MEM="${MEM:-240G}"; CPUS="${CPUS:-32}"
SEED="${SEED:-42}"; PROJECT_NAME="${PROJECT_NAME:-verl_adam}"
MODEL_PATH_IN_CONT="${MODEL_PATH_IN_CONT:-/mnt/home/models/Llama-3.2-3B-Instruct}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo.sh}"
IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"; BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"; SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"
FILTER_WORKERS="${FILTER_WORKERS:-8}"; FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-fp32}"; FSDP_TRAIN_DTYPE="${FSDP_TRAIN_DTYPE:-bfloat16}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"; ATTN_IMPL="${ATTN_IMPL:-sdpa}"; MOMENTUM="${MOMENTUM:-0.9}"
REPO_IN_CONT="/mnt/home/verl_v0.4.x"; PYTHON_BIN_IN_CONT="${PYTHON_BIN_IN_CONT:-${REPO_IN_CONT}/.verl/bin/python}"
[[ -f "${IMG}" ]] || { echo "[error] IMG not found: ${IMG}" >&2; exit 1; }
lrs=(5e-7 4e-7 3e-7 2e-7 1e-7)
mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs
for lr in "${lrs[@]}"; do
  host_ts="$(date +%Y%m%d_%H%M%S)"; lr_tag="${lr//./p}"; lr_tag="${lr_tag//-/m}"
  mp_tag="sdpa_amp_bf16_mfp32_SignSGD_Momentum_llama32_lr${lr_tag}_mom${MOMENTUM}_pe${PPO_EPOCHS}"
  job_name="grpo_llama32_3b_${mp_tag}_s${SEED}"; exp_name="math_grpo_llama32_3b_${mp_tag}_s${SEED}_${host_ts}"; out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"
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
    "actor_rollout_ref.actor.optim.override_optimizer_config.momentum=${MOMENTUM}"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "critic.data_loader_seed=${SEED}"
    "trainer.val_subset_seed=${SEED}"
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
    "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
    "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
  )
  hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"
  wrap_cmd="set -euo pipefail; echo [cfg] lr=${lr} mom=${MOMENTUM} exp=${exp_name}; apptainer exec --nv -B  -B   bash -lc \"set -euo pipefail; cd ; export PYTHON_BIN=; export PROJECT_NAME=; export EXP_NAME=; export OUT_DIR=; export CACHE_BASE=/mnt/home/verl_cache; export RAY_TMPDIR=/mnt/home/raytmp; export NGPUS_PER_NODE=; export NNODES=1; bash  ${hydra_args_escaped}\""
  jid=$(sbatch --parsable --job-name="${job_name}" --partition="${PARTITION}" --gpus="${GPUS}" --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${TIME}" --output="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out" --error="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.err" --wrap "${wrap_cmd}")
  echo "[submitted] lr=${lr} job_id=${jid}"
done
echo "[done] submitted ${#lrs[@]} jobs"
