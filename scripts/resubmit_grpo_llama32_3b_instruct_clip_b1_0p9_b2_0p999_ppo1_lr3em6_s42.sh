#!/usr/bin/env bash
set -euo pipefail

# Resubmit the single failed config:
#   betas=[0.9,0.999], actor_lr=3e-6, ppo_epochs=1, seed=42, 2GPU/10h
#
# Failure root cause in previous run: torch.distributed DistNetworkError (EADDRINUSE).
# Mitigation:
# - exclude the previous bad node by default
# - set MASTER_PORT deterministically from SLURM_JOB_ID to avoid port collisions
#
# Usage:
#   bash verl_v0.4.x/scripts/resubmit_grpo_llama32_3b_instruct_clip_b1_0p9_b2_0p999_ppo1_lr3em6_s42.sh
#
# Optional env overrides:
#   PARTITION=workq GPUS=2 TIME=10:00:00 MEM=240G CPUS=32
#   EXCLUDE=nid011278
#   PROJECT_NAME=verl_grpo_example_math
#   MODEL_PATH_IN_CONT=/mnt/home/models/Llama-3.2-3B-Instruct
#   RUN_SCRIPT=examples/RL_math/run_qwen2.5-3b_math_grpo.sh
#   IMG=/scratch/u6g/zhouyihe.u6g/sglang.sif
#   BIND=/scratch/u6g/zhouyihe.u6g:/mnt/home
#   FILTER_WORKERS=8 PPO_EPOCHS=1

PARTITION="${PARTITION:-workq}"
GPUS="${GPUS:-2}"
TIME="${TIME:-10:00:00}"
MEM="${MEM:-240G}"
CPUS="${CPUS:-32}"
EXCLUDE="${EXCLUDE:-nid011278}"

SEED="${SEED:-42}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_example_math}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"

MODEL_PATH_IN_CONT="${MODEL_PATH_IN_CONT:-/mnt/home/models/Llama-3.2-3B-Instruct}"
RUN_SCRIPT="${RUN_SCRIPT:-examples/RL_math/run_qwen2.5-3b_math_grpo.sh}"

IMG="${IMG:-/scratch/u6g/zhouyihe.u6g/sglang.sif}"
BIND="${BIND:-/scratch/u6g/zhouyihe.u6g:/mnt/home}"
SHM_BIND="${SHM_BIND:-/dev/shm:/dev/shm}"

FILTER_WORKERS="${FILTER_WORKERS:-8}"

REPO_IN_CONT="/mnt/home/verl_v0.4.x"
PYTHON_BIN_IN_CONT="${PYTHON_BIN_IN_CONT:-${REPO_IN_CONT}/.verl/bin/python}"

beta1="0.9"
beta2="0.999"
actor_lr="3e-6"

host_ts="$(date +%Y%m%d_%H%M%S)"

beta_name="b1_0p9_b2_0p999"
lr_tag="3em6"

job_name="grpo_llama32_3b_instruct_clip_${beta_name}_ppo${PPO_EPOCHS}_lr${lr_tag}_s${SEED}"
exp_name="math_grpo_llama32_3b_instruct_clip_${beta_name}_ppo${PPO_EPOCHS}_lr${lr_tag}_s${SEED}_${host_ts}"
out_dir="${REPO_IN_CONT}/outputs/${PROJECT_NAME}/${exp_name}"

mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

seed_overrides=(
  "actor_rollout_ref.actor.data_loader_seed=${SEED}"
  "critic.data_loader_seed=${SEED}"
  "trainer.val_subset_seed=${SEED}"
  "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
  "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
)

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
apptainer exec --nv -B "${BIND}" -B "${SHM_BIND}" "${IMG}" bash -lc "set -euo pipefail; cd '${REPO_IN_CONT}'; export PYTHON_BIN='${PYTHON_BIN_IN_CONT}'; export PROJECT_NAME='${PROJECT_NAME}'; export EXP_NAME='${exp_name}'; export OUT_DIR='${out_dir}'; export CACHE_BASE='/mnt/home/verl_cache/\${SLURM_JOB_ID}'; export RAY_TMPDIR='/mnt/home/raytmp/\${SLURM_JOB_ID}'; export NGPUS_PER_NODE='${GPUS}'; export NNODES=1; export PPO_EPOCHS='${PPO_EPOCHS}'; export MASTER_PORT=\$((10000 + \${SLURM_JOB_ID} % 50000)); bash '${RUN_SCRIPT}' '${MODEL_PATH_IN_CONT}'${hydra_args_escaped}"
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

echo "[resubmitted] job_id=${jid} name=${job_name} exp_name=${exp_name} exclude=${EXCLUDE}"

