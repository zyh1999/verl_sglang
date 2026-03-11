#!/usr/bin/env bash
set -euo pipefail

# Submit GRPO actor learning-rate sweep for Llama-3.2-3B-Instruct (CLIP enabled)
# under two beta settings:
#  - beta1=beta2=0.92
#  - original AdamW betas: [0.9, 0.999]
#
# Key requirement: override ppo_epochs=1 via Hydra args (do NOT edit run_*.sh).
# Total jobs = 2 * (#ACTOR_LRS) = 10 by default.
#
# Usage (on host):
#   bash verl_v0.4.x/scripts/submit_grpo_llama32_3b_instruct_clip_lr_sweep_2betas_2gpu_10h_ppo1.sh
#
# Optional env overrides:
#   PARTITION=workq
#   GPUS=2
#   TIME=10:00:00
#   MEM=240G
#   CPUS=32
#   SEED=42
#   EXCLUDE=nid010230
#   PROJECT_NAME=verl_grpo_example_math
#   MODEL_PATH_IN_CONT=/mnt/home/models/Llama-3.2-3B-Instruct
#   RUN_SCRIPT=examples/RL_math/run_qwen2.5-3b_math_grpo.sh
#   IMG=/scratch/u6g/zhouyihe.u6g/sglang.sif
#   BIND=/scratch/u6g/zhouyihe.u6g:/mnt/home
#   FILTER_WORKERS=8
#   ACTOR_LRS="3e-7 6e-7 2e-6 3e-6 1e-5"
#   PPO_EPOCHS=1
#
# Notes:
# - Uses short SLURM_JOB_ID-based cache dirs to avoid Ray AF_UNIX socket path limit.
# - Binds /dev/shm for HF datasets multiprocessing SemLock.

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-10:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
EXCLUDE="${EXCLUDE:-}"

SEED="${SEED:-42}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_example_math}"

MODEL_PATH_IN_CONT="${MODEL_PATH_IN_CONT:-/mnt/home/models/Llama-3.2-3B-Instruct}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo.sh}"

IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"

FILTER_WORKERS="${FILTER_WORKERS:-8}"
ACTOR_LRS="${ACTOR_LRS:-3e-7 6e-7 2e-6 3e-6 1e-5}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"

REPO_IN_CONT="/mnt/home/verl_v0.4.x"
PYTHON_BIN_IN_CONT="${PYTHON_BIN_IN_CONT:-${REPO_IN_CONT}/.verl/bin/python}"

if [[ ! -f "${IMG}" ]]; then
  echo "[error] IMG not found: ${IMG}" >&2
  exit 1
fi

host_ts="$(date +%Y%m%d_%H%M%S)"

# two beta settings
beta_pairs=(
  "0.92 0.92"
  "0.9 0.999"
)

actor_lrs_array=(${ACTOR_LRS})

echo "============================================================"
echo "[llama-clip-2betas-lr-sweep] partition=${PARTITION} gpus=${GPUS} time=${TIME} mem=${MEM} cpus=${CPUS}"
echo "[llama-clip-2betas-lr-sweep] seed=${SEED} project=${PROJECT_NAME}"
echo "[llama-clip-2betas-lr-sweep] model_in_cont=${MODEL_PATH_IN_CONT}"
echo "[llama-clip-2betas-lr-sweep] override: actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
echo "[llama-clip-2betas-lr-sweep] actor_lrs=${ACTOR_LRS}"
echo "[llama-clip-2betas-lr-sweep] img=${IMG}"
echo "[llama-clip-2betas-lr-sweep] bind=${BIND}"
echo "[llama-clip-2betas-lr-sweep] run_script=${RUN_SCRIPT}"
echo "============================================================"

mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

submitted=()

for pair in "${beta_pairs[@]}"; do
  read -r beta1 beta2 <<<"${pair}"

  beta1_tag="$(echo "${beta1}" | sed 's/\./p/g')"
  beta2_tag="$(echo "${beta2}" | sed 's/\./p/g')"
  if [[ "${beta1}" == "${beta2}" ]]; then
    beta_name="b1eqb2_${beta1_tag}"
  else
    beta_name="b1_${beta1_tag}_b2_${beta2_tag}"
  fi

  for actor_lr in "${actor_lrs_array[@]}"; do
    lr_tag="$(echo "${actor_lr}" | sed 's/\\./p/g' | sed 's/e-/em/g')"
    job_name="grpo_llama32_3b_instruct_clip_${beta_name}_ppo${PPO_EPOCHS}_lr${lr_tag}_s${SEED}"
    exp_name="math_grpo_llama32_3b_instruct_clip_${beta_name}_ppo${PPO_EPOCHS}_lr${lr_tag}_s${SEED}_${host_ts}"
    out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"

    seed_overrides=(
      "actor_rollout_ref.actor.data_loader_seed=${SEED}"
      "critic.data_loader_seed=${SEED}"
      "trainer.val_subset_seed=${SEED}"
      "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
      "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
    )

    # IMPORTANT: override PPO epochs via Hydra, do not rely on the run script default.
    hydra_args=(
      "actor_rollout_ref.actor.optim.betas=[${beta1},${beta2}]"
      "actor_rollout_ref.actor.optim.lr=${actor_lr}"
      "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
      "${seed_overrides[@]}"
      "data.filter_overlong_prompts_workers=${FILTER_WORKERS}"
    )
    hydra_args_escaped="$(printf " %q" "${hydra_args[@]}")"

    wrap_cmd=$(
      cat <<EOF
set -euo pipefail
echo "[slurm] job=\${SLURM_JOB_ID} host=\$(hostname) start=\$(date)"
echo "[cfg] betas=[${beta1},${beta2}] actor_lr=${actor_lr} seed=${SEED} ppo_epochs=${PPO_EPOCHS} exp_name=${exp_name}"
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd '${REPO_IN_CONT}'; export PYTHON_BIN='${PYTHON_BIN_IN_CONT}'; export PROJECT_NAME='${PROJECT_NAME}'; export EXP_NAME='${exp_name}'; export OUT_DIR='${out_dir}'; export CACHE_BASE='/mnt/home/verl_cache/\${SLURM_JOB_ID}'; export RAY_TMPDIR='/mnt/home/raytmp/\${SLURM_JOB_ID}'; export NGPUS_PER_NODE='${GPUS}'; export NNODES=1; export PPO_EPOCHS='${PPO_EPOCHS}'; bash '${RUN_SCRIPT}' '${MODEL_PATH_IN_CONT}'${hydra_args_escaped}"
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

    submitted+=("${jid}:${job_name}:betas=[${beta1},${beta2}]:lr=${actor_lr}:ppo=${PPO_EPOCHS}")
    echo "[submitted] job_id=${jid} name=${job_name} betas=[${beta1},${beta2}] lr=${actor_lr} ppo_epochs=${PPO_EPOCHS}"
  done
done

echo "============================================================"
echo "[done] submitted ${#submitted[@]} jobs"
printf '%s\n' "${submitted[@]}"
echo "Tips:"
echo "  squeue -u \$USER -o '%.18i %.70j %.8T %.10M %.10l %R' | egrep 'grpo_llama32_3b_instruct_clip_.*_ppo'"
echo "============================================================"

