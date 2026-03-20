#!/usr/bin/env bash
set -xeuo pipefail

# Force conda env/pkg lookup on scratch first (reduce HOME pressure)
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/h99859yz/conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/h99859yz/conda/pkgs}"

# ============================================================
# verl GRPO Example: Qwen2.5-3B-Instruct (Math) - NO-IS + NO CLIP ABLATION
# - 基于 examples/RL_math/run_qwen2.5-3b_math_grpo_no_IS.sh
# - 仅增加 disable_clip 开关（默认 True）：actor_rollout_ref.actor.disable_clip
# ============================================================

unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 默认使用当前 PATH 上的 python（在容器里建议激活 venv 后运行，例如 source .verl/bin/activate）
# 如需指定，运行前设置：PYTHON_BIN=/path/to/python
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

# 统一把运行时产生的缓存/临时文件写到「当前目录」(默认你在仓库根目录 verl_v0.4.x 下运行 bash)。
# 可通过环境变量覆盖（例如 CACHE_BASE=/mnt/home）。
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/h99859yz}"
CACHE_BASE="${CACHE_BASE:-${SCRATCH_BASE}/verl_new/cache}"
mkdir -p "${CACHE_BASE}"
# 统一把 cache base 规范成绝对路径（一些工具要求 cache 目录必须是绝对路径）
CACHE_BASE="$(cd "${CACHE_BASE}" && pwd)"
export CACHE_BASE
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_BASE}/.hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export RAY_TMPDIR="${RAY_TMPDIR:-${CACHE_BASE}/ray}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${RAY_TMPDIR}"

nnodes="${NNODES:-1}"

# 数据（默认用仓库内的 ./data，不放在 $HOME 下）
data_root="${DATA_ROOT:-${ROOT_DIR}/data}"
gsm8k_train_path="${GSM8K_TRAIN_PATH:-$data_root/gsm8k/train.parquet}"
gsm8k_test_path="${GSM8K_TEST_PATH:-$data_root/gsm8k/test.parquet}"
# 训练用的 “math7500”：使用 SeRL 提供的 7.5k GT（data/math_task/train.parquet）
math_train_path="${MATH_TRAIN_PATH:-$data_root/math_task/train.parquet}"
math_test_path="${MATH_TEST_PATH:-$data_root/math_task/test.parquet}"
math500_test_path="${MATH500_TEST_PATH:-$data_root/math_task/test.parquet}"
math_hard_test_path="${MATH_HARD_TEST_PATH:-$data_root/math_task_hard/test.parquet}"
aime2024_test_path="${AIME2024_TEST_PATH:-$data_root/math_task_aime2024/test.parquet}"
aime2025_test_path="${AIME2025_TEST_PATH:-$data_root/math_task_aime2025/test.parquet}"
gpqa_test_path="${GPQA_TEST_PATH:-$data_root/math_task_gpqa/test.parquet}"

# 训练集：默认跑 GSM8K + Math
train_files="${TRAIN_FILES:-['$math_train_path']}"
# 测试集：同时跑 Math500 + math_hard，并在日志里按 data_source 分开汇报
test_files="${TEST_FILES:-['$math500_test_path','$math_hard_test_path','$aime2024_test_path','$aime2025_test_path','$gpqa_test_path']}"

# 模型（默认：Qwen2.5 3B）
# 支持把模型作为第一个位置参数传入：
#   bash run_qwen2.5-3b_math_grpo_no_IS_no_clip.sh <MODEL_PATH_OR_NAME> [hydra_overrides...]
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
optimizer_name="${OPTIMIZER:-AdamW}"

project_name="${PROJECT_NAME:-verl_new}"
exp_name="${EXP_NAME:-qwen2.5_3b_no_IS_train_gsm8k+math_val_math500+math_hard_grpo_epochs_${ppo_epochs}_no_clip}"

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

# clip（参数仍保留，但在 disable_clip=True 时会被 policy loss 忽略）
clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.2}"
clip_ratio_c="${CLIP_RATIO_C:-3.0}"
loss_agg_mode="${LOSS_AGG_MODE:-token-mean}"

# no-IS：对齐原 no_IS 脚本默认 False
use_importance_sampling="${USE_IMPORTANCE_SAMPLING:-False}"

# no-clip 开关（默认 True；需要对照实验可设 False）
disable_clip="${DISABLE_CLIP:-True}"

# 性能相关参数
sp_size="${SP_SIZE:-1}"
gen_tp="${GEN_TP:-1}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}"
offload="${OFFLOAD:-False}"
ENGINE="${ENGINE:-sglang}"

# 日志/输出
out_dir="${OUT_DIR:-${SCRATCH_BASE}/verl_new/ckpt/${project_name}/${exp_name}}"
mkdir -p "${out_dir}"

echo "============================================================"
echo "[verl][GRPO] Qwen2.5-3B (Math) - NO-IS + NO CLIP"
echo "project_name=${project_name}"
echo "exp_name=${exp_name}"
echo "model=${MODEL_PATH}"
echo "train_files=${train_files}"
echo "max_prompt_length=${max_prompt_length}, max_response_length=${max_response_length}"
echo "train_bsz=${train_prompt_bsz}, mini_bsz=${train_prompt_mini_bsz}, micro_bsz/gpu=${micro_batch_size_per_gpu}"
echo "optimizer=${optimizer_name}"
echo "ppo_epochs=${ppo_epochs}"
echo "n=${n_resp_per_prompt}, temp=${temperature}, top_p=${top_p}, top_k=${top_k}"
echo "engine=${ENGINE}, gen_tp=${gen_tp}"
echo "val_n=${val_n}, val_do_sample=${val_do_sample}, val_temp=${val_temperature}, val_top_p=${val_top_p}, val_top_k=${val_top_k}"
echo "val_subset_ratio=${val_subset_ratio}"
echo "val_subset_seed=${val_subset_seed}, val_subset_resample_each_eval=${val_subset_resample_each_eval}"
echo "use_importance_sampling=${use_importance_sampling}"
echo "disable_clip=${disable_clip}"
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
  actor_rollout_ref.actor.fsdp_config.dtype=bf16 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.actor.ppo_epochs="${ppo_epochs}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_batch_size_per_gpu}" \
  actor_rollout_ref.actor.optim.optimizer="${optimizer_name}" \
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
  +actor_rollout_ref.actor.disable_clip="${disable_clip}" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
  actor_rollout_ref.rollout.name="${ENGINE}" \
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
  trainer.n_gpus_per_node="${NGPUS_PER_NODE:-2}" \
  trainer.nnodes="${nnodes}" \
  trainer.save_freq="${SAVE_FREQ:-100}" \
  trainer.test_freq="${TEST_FREQ:-20}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-4}" \
  trainer.resume_mode="disable" \
  trainer.resume_from_path=null \
  trainer.default_local_dir="${out_dir}" \
  "$@" 2>&1 | tee "${out_dir}/${project_name}_${exp_name}_grpo.log"
