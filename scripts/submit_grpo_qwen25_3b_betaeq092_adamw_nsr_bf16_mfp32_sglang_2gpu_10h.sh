#!/usr/bin/env bash
set -euo pipefail

PARTITION="workq"
GPUS="2"
TIME="10:00:00"
MEM="240G"
CPUS="32"
EXCLUDE=""

SEED="42"
PROJECT_NAME="verl_adam"
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
RUN_SCRIPT="examples/RL_math/run_qwen2.5-3b_math_grpo.sh"

IMG="/scratch/u6g/zhouyihe.u6g/sglang.sif"
BIND="/scratch/u6g/zhouyihe.u6g:/mnt/home"
SHM_BIND="/dev/shm:/dev/shm"

FILTER_WORKERS="8"
FSDP_MODEL_DTYPE="fp32"
FSDP_TRAIN_DTYPE="bfloat16"
ROLLOUT_DTYPE="bfloat16"
ROLLOUT_NAME="sglang"
ATTN_IMPL="flash_attention_2"
MP_TAG="amp_bf16_mfp32__attn_"
MP_TAG=""

REPO_IN_CONT="/mnt/home/verl_v0.4.x"
PYTHON_BIN_IN_CONT="/.verl/bin/python"

host_ts="20260303_102132"
beta1="0.92"
beta2="0.92"
beta_name="b1eqb2_0p92"

job_name="grpo_qwen25_3b___adamw_nsr_s"
exp_name="math_grpo_qwen25_3b___adamw_nsr_s_"
out_dir="/outputs//"

seed_overrides=(
  "actor_rollout_ref.actor.data_loader_seed="
  "critic.data_loader_seed="
  "trainer.val_subset_seed="
  "actor_rollout_ref.actor.fsdp_config.seed="
  "actor_rollout_ref.ref.fsdp_config.seed="
)

hydra_args=(
  "actor_rollout_ref.model.override_config.attn_implementation="
  "actor_rollout_ref.rollout.name="
  "actor_rollout_ref.rollout.dtype="
  "actor_rollout_ref.actor.fsdp_config.model_dtype="
  "actor_rollout_ref.actor.fsdp_config.dtype="
  "actor_rollout_ref.actor.fsdp_config.use_orig_params=True"
  "actor_rollout_ref.ref.fsdp_config.model_dtype="
  "actor_rollout_ref.ref.fsdp_config.dtype="
  "actor_rollout_ref.ref.fsdp_config.use_orig_params=True"
  "actor_rollout_ref.actor.optim.optimizer=AdamW_NSR"
  "actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.adamw_nsr"
  "actor_rollout_ref.actor.optim.betas=[,]"
  "actor_rollout_ref.actor.log_adam_snr=True"
  "actor_rollout_ref.actor.ppo_epochs=1"
  ""
  "data.filter_overlong_prompts_workers="
)
hydra_args_escaped=" \"\""

mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs

wrap_cmd=set -euo pipefail
echo \"[slurm] job=${SLURM_JOB_ID} host=$(hostname) start=$(date)\"
echo \"[cfg] bf16_mfp32_adamw_nsr seed= exp_name=\"
apptainer exec --nv -B \"\" -B \"\" \"\" bash -lc \"set -euo pipefail; cd ''; export PYTHON_BIN=''; export PROJECT_NAME=''; export MODEL_PATH=''; export EXP_NAME=''; export OUT_DIR=''; export CACHE_BASE='/mnt/home/verl_cache/\'; export RAY_TMPDIR='/mnt/home/raytmp/\'; export NGPUS_PER_NODE=''; export NNODES=1; bash ''\"
echo \"[slurm] end=$(date)\"

jid=

echo "[submitted] job_id= name="
