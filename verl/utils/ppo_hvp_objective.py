"""Differentiable PPO actor objective for HVP/preconditioned-sharpness.

This path is intentionally decoupled from regular logging/eval helpers and avoids
forward kernels that may break higher-order gradients.
"""

from collections.abc import Callable
from typing import Any

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits_v2


def evaluate_ppo_actor_objective_for_hvp(
    *,
    mini_batch: DataProto,
    actor_module: torch.nn.Module,
    config: Any,
    temperature: float,
    pad_token_id: int,
    ulysses_sequence_parallel_size: int,
) -> torch.Tensor:
    """Return differentiable scalar PPO objective tensor for HVP.

    Notes:
    - Uses vanilla logits->logprob path (logprobs_from_logits_v2) to keep grad graph.
    - Avoids fused CE / custom kernels in this HVP path.
    """
    if config.use_dynamic_bsz:
        max_token_len = config.ppo_max_token_len_per_gpu * ulysses_sequence_parallel_size
        micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
        grad_accum = None
    else:
        grad_accum = config.ppo_mini_batch_size // config.ppo_micro_batch_size_per_gpu
        micro_batches = mini_batch.split(config.ppo_micro_batch_size_per_gpu)

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    policy_loss_fn = get_policy_loss_fn(loss_mode)

    total_objective = None
    for mb in micro_batches:
        mb = mb.to(get_device_id())
        b = mb.batch

        input_ids = b["input_ids"]
        attention_mask = b["attention_mask"]
        position_ids = b["position_ids"]
        responses = b["responses"]
        response_mask = b["response_mask"]
        advantages = b["advantages"]
        old_log_prob = b["old_log_probs"]

        out = actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            pad_token_id=pad_token_id,
        )
        logits = out.logits
        response_length = responses.size(-1)
        logits = logits[:, -response_length - 1 : -1, :]
        logits = logits / temperature

        log_prob = logprobs_from_logits_v2(logits, responses)

        rollout_is_weights = b.get("rollout_is_weights", None)
        pg_loss, _ = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )

        policy_loss = pg_loss
        if config.use_kl_loss:
            ref_log_prob = b["ref_log_prob"]
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)
            policy_loss = policy_loss + kl_loss * config.kl_loss_coef

        if config.use_dynamic_bsz:
            loss_scale_factor = response_mask.shape[0] / config.ppo_mini_batch_size
        else:
            loss_scale_factor = 1 / grad_accum

        contrib = policy_loss * loss_scale_factor
        total_objective = contrib if total_objective is None else (total_objective + contrib)

    if total_objective is None:
        dev = get_device_id()
        total_objective = torch.tensor(0.0, device=dev)
    return total_objective
