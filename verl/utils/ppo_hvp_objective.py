"""Differentiable PPO actor objective for HVP/preconditioned-sharpness.

This path is intentionally decoupled from regular logging/eval helpers and avoids
forward kernels that may break higher-order gradients.
"""

from collections.abc import Callable
from typing import Any
import os

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits_v2


def _unwrap_module(m: torch.nn.Module) -> torch.nn.Module:
    cur = m
    # unwrap common wrappers (DDP/FSDP style .module chains)
    for _ in range(6):
        nxt = getattr(cur, "module", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def _find_last_transformer_block(actor_module: torch.nn.Module) -> torch.nn.Module | None:
    base = _unwrap_module(actor_module)

    # Common HF layouts
    candidates = []
    model = getattr(base, "model", None)
    if model is not None:
        layers = getattr(model, "layers", None)
        if layers is not None:
            candidates.append(layers)
    transformer = getattr(base, "transformer", None)
    if transformer is not None:
        h = getattr(transformer, "h", None)
        if h is not None:
            candidates.append(h)
    layers2 = getattr(base, "layers", None)
    if layers2 is not None:
        candidates.append(layers2)

    for seq in candidates:
        try:
            if len(seq) > 0:
                return seq[-1]
        except Exception:
            continue
    return None


def _fmt_cuda_mem(tag: str) -> str:
    if not torch.cuda.is_available():
        return f"{tag} cuda=na"
    d = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(d) / (1024**3)
    reserv = torch.cuda.memory_reserved(d) / (1024**3)
    peak = torch.cuda.max_memory_allocated(d) / (1024**3)
    return f"{tag} cuda_alloc={alloc:.3f}GiB reserved={reserv:.3f}GiB peak={peak:.3f}GiB"


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
    - Supports token subsampling via config.hvp_token_stride (e.g., 4 => use 1/4 tokens).
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
    hvp_token_stride = int(getattr(config, "hvp_token_stride", 1))
    hvp_token_stride = max(hvp_token_stride, 1)

    # Optional hard graph-cut before the last transformer block for HVP path.
    # This ensures higher-order graph stays local to the last block.
    hvp_detach_last_block_input = bool(getattr(config, "hvp_detach_last_block_input", False))
    hvp_mem_debug = bool(getattr(config, "hvp_mem_debug", False) or (os.getenv("HVP_MEM_DEBUG", "0") == "1"))

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

        hook = None
        if hvp_detach_last_block_input:
            last_block = _find_last_transformer_block(actor_module)
            if last_block is not None:
                def _pre_hook(_module, args):
                    if not args:
                        return args
                    x0 = args[0]
                    if isinstance(x0, torch.Tensor):
                        x0 = x0.detach().requires_grad_(True)
                        return (x0,) + tuple(args[1:])
                    return args

                hook = last_block.register_forward_pre_hook(_pre_hook)

        try:
            if hvp_mem_debug and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                print(f"[hvp_mem] {_fmt_cuda_mem('before_forward')}", flush=True)
            out = actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                pad_token_id=pad_token_id,
            )
            if hvp_mem_debug:
                print(f"[hvp_mem] {_fmt_cuda_mem('after_forward')}", flush=True)
        finally:
            if hook is not None:
                hook.remove()
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
