#!/usr/bin/env bash
set -xeuo pipefail

# Force conda env/pkg lookup on scratch first (reduce HOME pressure)
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/scratch/h99859yz/conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/h99859yz/conda/pkgs}"

# ============================================================
# verl GRPO Example: Qwen2.5 3B (Code)
# - 基于 examples/grpo_trainer/run_qwen2.5-3b_math_grpo.sh 的脚本风格
# - 使用 function-based RM：data_source in {codecontests, apps, codeforces, taco}
# - ground_truth 应包含测试用例（通常为 JSON string / dict，至少含 inputs/outputs）
# - 可选：设置 SANDBOX_FUSION_URL 走远程沙箱判题（更安全）
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
CACHE_BASE="${CACHE_BASE:-.}"
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

# 数据（默认用仓库内的 ./data；也可通过环境变量覆盖）
data_root="${DATA_ROOT:-${ROOT_DIR}/data}"
code_train_path="${CODE_TRAIN_PATH:-$data_root/code/train.parquet}"
code_test_path="${CODE_TEST_PATH:-$data_root/code/test.parquet}"

train_files="${TRAIN_FILES:-['$code_train_path']}"
test_files="${TEST_FILES:-['$code_test_path']}"

# 模型（可换成任意 HF CausalLM chat 模型；优先推荐 coder 版本）
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}"

# 长度配置（code 往往更长）
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-2048}"

# batch 配置（code 判题更重，默认稍保守；可用环境变量覆盖）
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-16}"
micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-16}"
ppo_epochs="${PPO_EPOCHS:-3}"

project_name="${PROJECT_NAME:-verl_grpo_example_code}"
exp_name="${EXP_NAME:-qwen2.5_coder_3b_code_no_IS_grpo_epochs_${ppo_epochs}}"

# Algorithm
adv_estimator="${ADV_ESTIMATOR:-grpo}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-4}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"

# Validation（用于 pass@k / best@k 风格的评估；code 任务一般也能用）
val_n="${VAL_N:-8}"
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

# 是否使用重要性采样（PPO 默认 True；no-IS 脚本会设 False）
use_importance_sampling="${USE_IMPORTANCE_SAMPLING:-False}"

# 性能相关参数（可按机器情况覆盖）
sp_size="${SP_SIZE:-1}"
gen_tp="${GEN_TP:-1}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}"
offload="${OFFLOAD:-False}"
# rollout 引擎可用显存占比（控制 KV cache 预留；过大可能导致训练/rollout 切换时 OOM）
gpu_mem_util="${GPU_MEMORY_UTILIZATION:-0.40}"

# rollout inference engine: sglang | vllm | hf
ENGINE="${ENGINE:-sglang}"

# 可选：远程沙箱判题（更安全，避免本地执行不可信代码）
# - SANDBOX_FUSION_URL 为空则走本地 prime_code 判题
sandbox_fusion_url="${SANDBOX_FUSION_URL:-}"
sandbox_fusion_max_concurrent="${SANDBOX_FUSION_MAX_CONCURRENT:-64}"

# 日志/输出
out_dir="${OUT_DIR:-./outputs/${project_name}/${exp_name}}"
mkdir -p "${out_dir}"

extra_args=()
if [[ -n "${sandbox_fusion_url}" ]]; then
  extra_args+=("reward_model.sandbox_fusion.url=${sandbox_fusion_url}")
  extra_args+=("reward_model.sandbox_fusion.max_concurrent=${sandbox_fusion_max_concurrent}")
else
  extra_args+=("reward_model.sandbox_fusion.url=null")
fi

echo "============================================================"
echo "[verl][GRPO][no-IS] Qwen2.5 3B (Code)"
echo "project_name=${project_name}"
echo "exp_name=${exp_name}"
echo "model=${MODEL_PATH}"
echo "train_files=${train_files}"
echo "max_prompt_length=${max_prompt_length}, max_response_length=${max_response_length}"
echo "train_bsz=${train_prompt_bsz}, mini_bsz=${train_prompt_mini_bsz}, micro_bsz/gpu=${micro_batch_size_per_gpu}"
echo "ppo_epochs=${ppo_epochs}"
echo "n=${n_resp_per_prompt}, temp=${temperature}, top_p=${top_p}, top_k=${top_k}"
echo "engine=${ENGINE}, gen_tp=${gen_tp}"
echo "gpu_memory_utilization=${gpu_mem_util}"
echo "val_n=${val_n}, val_do_sample=${val_do_sample}, val_temp=${val_temperature}, val_top_p=${val_top_p}, val_top_k=${val_top_k}"
echo "val_subset_ratio=${val_subset_ratio}"
echo "val_subset_seed=${val_subset_seed}, val_subset_resample_each_eval=${val_subset_resample_each_eval}"
echo "use_importance_sampling=${use_importance_sampling}"
echo "sandbox_fusion_url=${sandbox_fusion_url:-<local prime_code>}"
echo "============================================================"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${adv_estimator}" \
  data.train_files="${train_files}" \
  data.val_files="${test_files}" \
  data.train_batch_size="${train_prompt_bsz}" \
  data.dataloader_num_workers=2 \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
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
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}" \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
  +ray_kwargs.ray_init.include_dashboard=False \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_mem_util}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
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
  trainer.default_local_dir="${out_dir}" \
  "${extra_args[@]}" \
  "$@" 2>&1 | tee "${out_dir}/${project_name}_${exp_name}_grpo_no_IS.log"


