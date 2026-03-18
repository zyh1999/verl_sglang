#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 默认使用当前 PATH 上的 python（在容器里建议激活 venv 后运行，例如 source .verl/bin/activate）
# 如需指定，运行前设置：PYTHON_BIN=/path/to/python
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

# SGLang/flashinfer JIT 需要较新的 nvcc 才能处理 Hopper (compute_90a)
CUDA_HOME="${CUDA_HOME:-/opt/apps/libs/nvidia-cuda/toolkit/12.4.1}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# ============================================================
# verl GRPO Example: Qwen2.5-3B-Instruct (Math)
# - 除模型本身配置外，尽量保持第二份脚本写法
# - 入口保持 verl：python -m verl.trainer.main_ppo
# ============================================================

unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES

nnodes="${NNODES:-1}"

# 数据（默认用仓库内的 ./data，不放在 $HOME 下）
data_root="${DATA_ROOT:-${ROOT_DIR}/data}"
# gsm8k_train_path="${GSM8K_TRAIN_PATH:-$data_root/gsm8k/train.parquet}"
# gsm8k_test_path="${GSM8K_TEST_PATH:-$data_root/gsm8k/test.parquet}"
# 训练用的 “math7500”：使用 SeRL 提供的 7.5k GT（data/math_task/train.parquet）
math_train_path="${MATH_TRAIN_PATH:-$data_root/math_task/train.parquet}"
math_test_path="${MATH_TEST_PATH:-$data_root/math_task/test.parquet}"
# math500_test_path="${MATH500_TEST_PATH:-$data_root/math_task/test.parquet}"
# math_hard_test_path="${MATH_HARD_TEST_PATH:-$data_root/math_task_hard/test.parquet}"
aime2024_test_path="${AIME2024_TEST_PATH:-$data_root/math_task_aime2024/test.parquet}"
aime2025_test_path="${AIME2025_TEST_PATH:-$data_root/math_task_aime2025/test.parquet}"
# gpqa_test_path="${GPQA_TEST_PATH:-$data_root/math_task_gpqa/test.parquet}"

build_hydra_list() {
  local out="["
  local sep=""
  local item
  for item in "$@"; do
    out="${out}${sep}'${item}'"
    sep=","
  done
  out="${out}]"
  printf '%s' "$out"
}

# 训练集：默认只跑 math_task/train.parquet（约 7.5k）
# 可用环境变量 TRAIN_FILES 覆盖
train_files="${TRAIN_FILES:-['$math_train_path']}"

if [[ ! -f "${math_train_path}" && -z "${TRAIN_FILES:-}" ]]; then
  echo "Missing train file: ${math_train_path}" >&2
  exit 1
fi

# 测试集：默认只传入实际存在的文件；可用环境变量 TEST_FILES 覆盖
if [[ -n "${TEST_FILES:-}" ]]; then
  test_files="${TEST_FILES}"
else
  val_candidates=(
    "${math_test_path}"
    # "${math500_test_path}"
    # "${math_hard_test_path}"
    # "${aime2024_test_path}"
    # "${aime2025_test_path}"
    # "${gpqa_test_path}"
  )
  existing_val_files=()
  for path in "${val_candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      existing_val_files+=("${path}")
    else
      echo "Skipping missing val file: ${path}" >&2
    fi
  done
  if [[ ${#existing_val_files[@]} -eq 0 ]]; then
    echo "No validation files found under ${data_root}" >&2
    exit 1
  fi
  test_files="$(build_hydra_list "${existing_val_files[@]}")"
fi

# 模型（与第一份统一）
if [[ $# -gt 0 && "${1}" != -* ]]; then
  MODEL_PATH="${1}"
  shift
else
  MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
fi

# 长度配置
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-2048}"

# batch 配置
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-16}"
micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-16}"
ppo_epochs="${PPO_EPOCHS:-3}"

project_name="${PROJECT_NAME:-verl_new}"
# 默认 run 名
exp_name="${EXP_NAME:-qwen2.5_3b_train_gsm8k+math_val_math500+math_hard_grpo_epochs_${ppo_epochs}}"

# Algorithm
adv_estimator="${ADV_ESTIMATOR:-grpo}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"

# Validation
val_n="${VAL_N:-16}"
val_do_sample="${VAL_DO_SAMPLE:-True}"
val_temperature="${VAL_TEMPERATURE:-1.0}"
val_top_p="${VAL_TOP_P:-1.0}"
val_top_k="${VAL_TOP_K:--1}"
val_subset_ratio="${VAL_SUBSET_RATIO:-1.0}"
val_subset_seed="${VAL_SUBSET_SEED:-42}"
val_subset_resample_each_eval="${VAL_SUBSET_RESAMPLE_EACH_EVAL:-False}"

# KL config
use_kl_in_reward="${USE_KL_IN_REWARD:-False}"
kl_coef="${KL_COEF:-0.0}"
use_kl_loss="${USE_KL_LOSS:-True}"
kl_loss_coef="${KL_LOSS_COEF:-0.001}"
kl_loss_type="${KL_LOSS_TYPE:-low_var_kl}"

# clip（verl PPO/GRPO 仍会用到 actor 的 clip）
clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.2}"
clip_ratio_c="${CLIP_RATIO_C:-3.0}"
loss_agg_mode="${LOSS_AGG_MODE:-token-mean}"

use_importance_sampling="${USE_IMPORTANCE_SAMPLING:-True}"

# 性能相关参数
sp_size="${SP_SIZE:-1}"
gen_tp="${GEN_TP:-1}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}"
offload="${OFFLOAD:-False}"
ref_offload="${REF_OFFLOAD:-False}"
gpu_mem_util="${GPU_MEM_UTIL:-0.35}"
rollout_enforce_eager="${ROLLOUT_ENFORCE_EAGER:-True}"
free_cache_engine="${FREE_CACHE_ENGINE:-False}"
sglang_skip_server_warmup="${SGLANG_SKIP_SERVER_WARMUP:-False}"

rollout_name="${ROLLOUT_NAME:-vllm}"

if [[ "${rollout_name}" == "hf" ]]; then
  echo "Unsupported rollout backend for this script: hf (async rollout only supports sglang or vllm)" >&2
  exit 1
fi

rollout_extra_args=()
if [[ "${rollout_name}" == "sglang" ]]; then
  rollout_extra_args+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.skip_server_warmup=${sglang_skip_server_warmup}")
fi

# 日志/输出
out_dir="${OUT_DIR:-${ROOT_DIR}/outputs/${project_name}/${exp_name}}"

mkdir -p "${out_dir}"

echo "============================================================"
echo "[verl][GRPO] Qwen2.5-3B"
echo "project_name=${project_name}"
echo "exp_name=${exp_name}"
echo "model=${MODEL_PATH}"
echo "train_files=${train_files}"
echo "max_prompt_length=${max_prompt_length}, max_response_length=${max_response_length}"
echo "train_bsz=${train_prompt_bsz}, mini_bsz=${train_prompt_mini_bsz}, micro_bsz/gpu=${micro_batch_size_per_gpu}"
echo "ppo_epochs=${ppo_epochs}"
echo "n=${n_resp_per_prompt}, temp=${temperature}, top_p=${top_p}, top_k=${top_k}"
# echo "rollout=${rollout_name}, gpu_mem_util=${gpu_mem_util}, enforce_eager=${rollout_enforce_eager}, free_cache_engine=${free_cache_engine}, offload=${offload}, ref_offload=${ref_offload}"
echo "val_n=${val_n}, val_do_sample=${val_do_sample}, val_temp=${val_temperature}, val_top_p=${val_top_p}, val_top_k=${val_top_k}"
echo "val_subset_ratio=${val_subset_ratio}"
echo "val_subset_seed=${val_subset_seed}, val_subset_resample_each_eval=${val_subset_resample_each_eval}"
echo "use_importance_sampling=${use_importance_sampling}"
echo "============================================================"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${adv_estimator}" \
  data.train_files="${train_files}" \
  data.val_files="${test_files}" \
  data.train_batch_size="${train_prompt_bsz}" \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.actor.ppo_epochs="${ppo_epochs}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_batch_size_per_gpu}" \
  actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
  actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
  actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}" \
  algorithm.use_kl_in_reward="${use_kl_in_reward}" \
  algorithm.kl_ctrl.kl_coef="${kl_coef}" \
  actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
  actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
  actor_rollout_ref.actor.clip_ratio_c="${clip_ratio_c}" \
  actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
  actor_rollout_ref.actor.use_importance_sampling="${use_importance_sampling}" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${ref_offload}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
  actor_rollout_ref.rollout.name="${rollout_name}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_mem_util}" \
  actor_rollout_ref.rollout.enforce_eager="${rollout_enforce_eager}" \
  actor_rollout_ref.rollout.free_cache_engine="${free_cache_engine}" \
  actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
  actor_rollout_ref.rollout.temperature="${temperature}" \
  actor_rollout_ref.rollout.top_p="${top_p}" \
  actor_rollout_ref.rollout.top_k="${top_k}" \
  actor_rollout_ref.rollout.val_kwargs.n="${val_n}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${val_do_sample}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
  trainer.val_subset_ratio="${val_subset_ratio}" \
  trainer.val_subset_seed="${val_subset_seed}" \
  trainer.val_subset_resample_each_eval="${val_subset_resample_each_eval}" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE:-1}" \
  trainer.nnodes="${nnodes}" \
  trainer.save_freq="${SAVE_FREQ:-100}" \
  trainer.test_freq="${TEST_FREQ:-20}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-4}" \
  trainer.resume_mode="disable" \
  trainer.resume_from_path=null \
  trainer.default_local_dir="${out_dir}" \
  "${rollout_extra_args[@]}" \
  "$@" 2>&1 | tee "${out_dir}/${project_name}_${exp_name}_grpo.log"