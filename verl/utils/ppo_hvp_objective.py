from typing import Any
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits_v2


def _unwrap_module(m: torch.nn.Module) -> torch.nn.Module:
    cur = m
    for _ in range(6):
        nxt = getattr(cur, 'module', None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def _find_last_transformer_block(actor_module: torch.nn.Module):
    base = _unwrap_module(actor_module)
    model = getattr(base, 'model', None)
    if model is not None and getattr(model, 'layers', None) is not None and len(model.layers) > 0:
        return model.layers[-1]
    transformer = getattr(base, 'transformer', None)
    if transformer is not None and getattr(transformer, 'h', None) is not None and len(transformer.h) > 0:
        return transformer.h[-1]
    layers2 = getattr(base, 'layers', None)
    if layers2 is not None and len(layers2) > 0:
        return layers2[-1]
    return None


def evaluate_ppo_actor_objective_for_hvp(*, mini_batch: DataProto, actor_module: torch.nn.Module, config: Any, temperature: float, pad_token_id: int, ulysses_sequence_parallel_size: int) -> torch.Tensor:
    if config.use_dynamic_bsz:
        max_token_len = config.ppo_max_token_len_per_gpu * ulysses_sequence_parallel_size
        micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
        grad_accum = None
    else:
        grad_accum = config.ppo_mini_batch_size // config.ppo_micro_batch_size_per_gpu
        micro_batches = mini_batch.split(config.ppo_micro_batch_size_per_gpu)

    policy_loss_fn = get_policy_loss_fn(config.policy_loss.get('loss_mode', 'vanilla'))
    hvp_token_stride = max(int(getattr(config, 'hvp_token_stride', 1)), 1)
    hvp_detach_last_block_input = bool(getattr(config, 'hvp_detach_last_block_input', False))
    hvp_local_graph_mode = str(getattr(config, 'hvp_local_graph_mode', '')).lower()

    total_objective = None
    for mb in micro_batches:
        mb = mb.to(get_device_id())
        b = mb.batch
        input_ids = b['input_ids']
        attention_mask = b['attention_mask']
        position_ids = b['position_ids']
        responses = b['responses']
        response_mask = b['response_mask']
        advantages = b['advantages']
        old_log_prob = b['old_log_probs']

        hook = None
        if hvp_detach_last_block_input and hvp_local_graph_mode != 'lm_head_only':
            last_block = _find_last_transformer_block(actor_module)
            if last_block is not None:
                def _pre_hook(_module, args):
                    if not args:
                        return args
                    x0 = args[0]
                    if isinstance(x0, torch.Tensor):
                        x0 = x0.detach()
                        return (x0,) + tuple(args[1:])
                    return args
                hook = last_block.register_forward_pre_hook(_pre_hook)

        try:
            if hvp_local_graph_mode == 'lm_head_only':
                base = _unwrap_module(actor_module)
                trunk = getattr(base, 'model', None)
                if trunk is None:
                    raise RuntimeError('hvp_local_graph_mode=lm_head_only requires actor_module.model')
                # align input dtypes to trunk parameter dtype (avoid Float vs BFloat16 mismatch)
                trunk_dtype = torch.bfloat16
                if trunk_dtype is not None:
                    # For FA2 path keep attention_mask as bool mask.
                    if attention_mask.dtype != torch.bool:
                        attention_mask = attention_mask.to(torch.bool)
                with torch.no_grad():
                    if (trunk_dtype is not None) and torch.cuda.is_available() and trunk_dtype.is_floating_point:
                        with torch.autocast(device_type='cuda', dtype=trunk_dtype):
                            out_ng = trunk(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                use_cache=False,
                                output_hidden_states=True,
                            )
                    else:
                        out_ng = trunk(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                            output_hidden_states=True,
                        )
                    hidden = out_ng.hidden_states[-1]
                hidden = hidden.detach().to(torch.bfloat16)
                lm_head = getattr(base, 'lm_head', None)
                if lm_head is None:
                    raise RuntimeError('lm_head missing for hvp_local_graph_mode=lm_head_only')
                for _p in lm_head.parameters():
                    _p.requires_grad_(True)
                target_dtype = lm_head.weight.dtype if hasattr(lm_head, 'weight') else hidden.dtype
                if hidden.dtype != target_dtype:
                    hidden = hidden.to(target_dtype)
                # IMPORTANT: do NOT materialize full-vocab logits here in lm_head_only mode.
            else:
                out = actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    pad_token_id=pad_token_id,
                )
                logits = out.logits
        finally:
            if hook is not None:
                hook.remove()

        response_length = responses.size(-1)
        if hvp_local_graph_mode == 'lm_head_only':
            hidden_resp = hidden[:, -response_length - 1 : -1, :]
        else:
            logits = logits[:, -response_length - 1 : -1, :] / temperature

        if hvp_token_stride > 1:
            sl = slice(None, None, hvp_token_stride)
            responses_hvp = responses[:, sl]
            old_log_prob_hvp = old_log_prob[:, sl]
            advantages_hvp = advantages[:, sl]
            response_mask_hvp = response_mask[:, sl]
            rollout_is_weights_hvp = b.get('rollout_is_weights', None)
            if rollout_is_weights_hvp is not None and rollout_is_weights_hvp.ndim >= 2:
                rollout_is_weights_hvp = rollout_is_weights_hvp[:, sl]
            ref_log_prob_hvp = b['ref_log_prob'][:, sl] if config.use_kl_loss else None
            if hvp_local_graph_mode == 'lm_head_only':
                hidden_hvp = hidden_resp[:, sl, :]
            else:
                logits_hvp = logits[:, sl, :]
        else:
            responses_hvp = responses
            old_log_prob_hvp = old_log_prob
            advantages_hvp = advantages
            response_mask_hvp = response_mask
            rollout_is_weights_hvp = b.get('rollout_is_weights', None)
            ref_log_prob_hvp = b['ref_log_prob'] if config.use_kl_loss else None
            if hvp_local_graph_mode == 'lm_head_only':
                hidden_hvp = hidden_resp
            else:
                logits_hvp = logits

        if hvp_local_graph_mode == 'lm_head_only':
            sampled_k = max(int(getattr(config, 'hvp_sampled_softmax_k', 0)), 0)
            if sampled_k > 0:
                B, T, _ = hidden_hvp.shape
                vocab_size = lm_head.weight.size(0)
                neg_ids = torch.randint(0, vocab_size, (sampled_k,), device=hidden_hvp.device)
                cand_ids = torch.cat([
                    responses_hvp.unsqueeze(-1),
                    neg_ids.view(1, 1, -1).expand(B, T, -1),
                ], dim=-1)
                W = lm_head.weight[cand_ids]
                logits_cand = (hidden_hvp.unsqueeze(-2) * W).sum(-1)
                if getattr(lm_head, 'bias', None) is not None:
                    logits_cand = logits_cand + lm_head.bias[cand_ids]
                logits_cand = logits_cand / temperature
                log_prob = torch.log_softmax(logits_cand, dim=-1)[..., 0]
            else:
                logits_hvp = lm_head(hidden_hvp) / temperature
                log_prob = logprobs_from_logits_v2(logits_hvp, responses_hvp)
        else:
            log_prob = logprobs_from_logits_v2(logits_hvp, responses_hvp)
        pg_loss, _ = policy_loss_fn(
            old_log_prob=old_log_prob_hvp,
            log_prob=log_prob,
            advantages=advantages_hvp,
            response_mask=response_mask_hvp,
            loss_agg_mode=config.loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights_hvp,
        )

        policy_loss = pg_loss
        if config.use_kl_loss:
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob_hvp, kl_penalty=config.kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask_hvp, loss_agg_mode=config.loss_agg_mode)
            policy_loss = policy_loss + kl_loss * config.kl_loss_coef

        loss_scale_factor = (response_mask_hvp.shape[0] / config.ppo_mini_batch_size) if config.use_dynamic_bsz else (1 / grad_accum)
        contrib = policy_loss * loss_scale_factor
        total_objective = contrib if total_objective is None else (total_objective + contrib)

    if total_objective is None:
        total_objective = torch.tensor(0.0, device=get_device_id())
    return total_objective
