# Copyright 2025 Yihe Zhou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility to evaluate PPO actor objective on a mini-batch."""

from collections.abc import Callable
from typing import Any

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.seqlen_balancing import prepare_dynamic_batch


def evaluate_ppo_actor_objective(
    *,
    mini_batch: DataProto,
    config: Any,
    temperature: float,
    pad_token_id: int,
    on_policy: bool,
    ulysses_sequence_parallel_size: int,
    forward_micro_batch_fn: Callable[..., dict[str, Any]],
    return_tensor: bool = False,
):
    """Evaluate the same PPO actor objective used for update.

    If return_tensor=True, returns a differentiable scalar Tensor for HVP usage.
    Otherwise returns Python float (legacy behavior).
    """
    if config.use_dynamic_bsz:
        max_token_len = config.ppo_max_token_len_per_gpu * ulysses_sequence_parallel_size
        micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
        grad_accum = None
    else:
        grad_accum = config.ppo_mini_batch_size // config.ppo_micro_batch_size_per_gpu
        micro_batches = mini_batch.split(config.ppo_micro_batch_size_per_gpu)

    entropy_coeff = config.entropy_coeff
    loss_agg_mode = config.loss_agg_mode
    calculate_entropy = config.calculate_entropy or (entropy_coeff != 0)
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    policy_loss_fn = get_policy_loss_fn(loss_mode)

    total_objective = None
    for micro_batch in micro_batches:
        micro_batch = micro_batch.to(get_device_id())
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
        response_mask = model_inputs["response_mask"]
        advantages = model_inputs["advantages"]

        outputs = forward_micro_batch_fn(model_inputs, temperature=temperature, calculate_entropy=calculate_entropy)
        log_prob = outputs["log_probs"]
        entropy = outputs["entropys"] if calculate_entropy else None

        if hasattr(config, "use_rollout_log_probs") and config.use_rollout_log_probs:
            old_log_prob = model_inputs["old_log_probs"]
        else:
            old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

        rollout_is_weights = model_inputs.get("rollout_is_weights", None)
        pg_loss, _ = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )

        policy_loss = pg_loss
        if calculate_entropy and entropy is not None:
            entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            if entropy_coeff != 0:
                policy_loss -= entropy_agg * entropy_coeff

        if config.use_kl_loss:
            ref_log_prob = model_inputs["ref_log_prob"]
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            policy_loss = policy_loss + kl_loss * config.kl_loss_coef

        if config.use_dynamic_bsz:
            loss_scale_factor = response_mask.shape[0] / config.ppo_mini_batch_size
        else:
            loss_scale_factor = 1 / grad_accum
        contrib = policy_loss * loss_scale_factor
        total_objective = contrib if total_objective is None else (total_objective + contrib)

    if total_objective is None:
        # no micro-batch case, keep type stable
        total_objective = policy_loss.new_tensor(0.0)

    if return_tensor:
        return total_objective
    return float(total_objective.detach().item())
