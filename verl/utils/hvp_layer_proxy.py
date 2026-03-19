from __future__ import annotations

import torch
import torch.nn.functional as F


def _unwrap_module(m: torch.nn.Module) -> torch.nn.Module:
    cur = m
    for _ in range(6):
        nxt = getattr(cur, "module", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def _find_last_down_proj(actor_module: torch.nn.Module) -> torch.nn.Module:
    base = _unwrap_module(actor_module)
    model = getattr(base, "model", None)
    if model is not None and getattr(model, "layers", None) is not None and len(model.layers) > 0:
        last = model.layers[-1]
    else:
        raise RuntimeError("cannot find model.layers[-1] for ffn down_proj hvp proxy")

    mlp = getattr(last, "mlp", None)
    if mlp is None:
        raise RuntimeError("last transformer block has no mlp")

    down_proj = getattr(mlp, "down_proj", None)
    if down_proj is None:
        raise RuntimeError("last transformer block mlp has no down_proj")
    return down_proj


def build_last_down_proj_proxy_logprob(
    *,
    actor_module: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    pad_token_id: int,
    response_length: int,
    hvp_token_stride: int,
    temperature: float,
    chunk_tokens: int = 64,
) -> torch.Tensor:
    down_proj = _find_last_down_proj(actor_module)

    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(_module, args):
        if not args:
            return args
        x0 = args[0]
        if isinstance(x0, torch.Tensor):
            captured["x"] = x0.detach()

    hook = down_proj.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            _ = actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                pad_token_id=pad_token_id,
            )
    finally:
        hook.remove()

    x = captured.get("x", None)
    if x is None:
        raise RuntimeError("failed to capture last down_proj input activation")

    x = x[:, -response_length - 1 : -1, :]
    if hvp_token_stride > 1:
        x = x[:, ::hvp_token_stride, :]

    outs = []
    T = x.size(1)
    if chunk_tokens <= 0:
        chunk_tokens = T

    for s in range(0, T, chunk_tokens):
        e = min(s + chunk_tokens, T)
        xc = x[:, s:e, :]
        target_dtype = down_proj.weight.dtype
        if xc.dtype != target_dtype:
            xc = xc.to(target_dtype)
        y = F.linear(xc, down_proj.weight, down_proj.bias)
        proxy = 0.5 * y.float().pow(2).mean(dim=-1)
        if temperature != 1.0:
            proxy = proxy / float(temperature)
        outs.append(proxy)

    return torch.cat(outs, dim=1)
