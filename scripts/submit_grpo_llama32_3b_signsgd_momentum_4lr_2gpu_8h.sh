#!/usr/bin/env bash
set -euo pipefail
PARTITION=workq
GPUS=2
TIME=08:00:00
MEM=240G
CPUS=32
SEED=42
PROJECT_NAME=verl_adam
MODEL_PATH_IN_CONT=/mnt/home/models/Llama-3.2-3B-Instruct
RUN_SCRIPT=examples/RL_math/run_qwen2.5-3b_math_grpo.sh
IMG=/scratch/u6g/zhouyihe.u6g/sglang.sif
BIND=/scratch/u6g/zhouyihe.u6g:/mnt/home
SHM_BIND=/dev/shm:/dev/shm
REPO_IN_CONT=/mnt/home/verl_v0.4.x
PYTHON_BIN_IN_CONT=/mnt/home/verl_v0.4.x/.verl/bin/python
lrs=(3e-6 1e-6 5e-7 1e-7)
mkdir -p /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs
for lr in ; do
  ts=20260305_180833
  lr_tag=; lr_tag=
  mp_tag=sdpa_amp_bf16_mfp32_SignSGD_Momentum_llama32_lr_pe1
  job_name=grpo_llama32_3b__s
  exp_name=math_grpo_llama32_3b__s_
  out_dir=/outputs//
  hydra_args=(
    actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp32
    actor_rollout_ref.ref.fsdp_config.dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True
    actor_rollout_ref.actor.optim.optimizer=SignSGD
    actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.sign_sgd
    actor_rollout_ref.actor.optim.lr=
    actor_rollout_ref.actor.optim.override_optimizer_config.momentum=0.9
    actor_rollout_ref.actor.data_loader_seed=
    critic.data_loader_seed=
    trainer.val_subset_seed=
    actor_rollout_ref.actor.fsdp_config.seed=
    actor_rollout_ref.ref.fsdp_config.seed=
    data.filter_overlong_prompts_workers=8
  )
  hydra_args_escaped= ''
  wrap_cmd=set -euo pipefail
apptainer exec --nv -B "" -B "" "" bash -lc "set -euo pipefail; cd ''; export PYTHON_BIN=''; export PROJECT_NAME=''; export EXP_NAME=''; export OUT_DIR=''; export CACHE_BASE='/mnt/home/verl_cache/${SLURM_JOB_ID}'; export RAY_TMPDIR='/mnt/home/raytmp/${SLURM_JOB_ID}'; export NGPUS_PER_NODE=''; export NNODES=1; bash '' ''"
  jid=
  echo submitted lr=
done
