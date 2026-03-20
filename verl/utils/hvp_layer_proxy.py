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


def _find_mlp_by_layer_idx(actor_module: torch.nn.Module, layer_idx: int = -1):
    base = _unwrap_module(actor_module)
    model = getattr(base, "model", None)
    if model is None or getattr(model, "layers", None) is None or len(model.layers) == 0:
        raise RuntimeError("cannot find model.layers for ffn down_proj hvp proxy")

    n_layers = len(model.layers)
    idx = int(layer_idx)
    if idx < 0:
        idx = n_layers + idx
    if idx < 0 or idx >= n_layers:
        raise RuntimeError(f"invalid hvp_target_layer_idx={layer_idx}, n_layers={n_layers}")

    target = model.layers[idx]
    mlp = getattr(target, "mlp", None)
    if mlp is None:
        raise RuntimeError(f"transformer layer[{idx}] has no mlp")

    return mlp, idx, n_layers


def build_layer_down_proj_proxy_logprob(
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
    hvp_target_layer_idx: int = -1,
) -> torch.Tensor:
    mlp, actual_idx, _ = _find_mlp_by_layer_idx(actor_module, hvp_target_layer_idx)

    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(_module, args):
        if not args:
            return args
        x0 = args[0]
        if isinstance(x0, torch.Tensor):
            captured["x"] = x0.detach()

    down_proj = getattr(mlp, "down_proj", None)
    if down_proj is None:
        raise RuntimeError(f"transformer layer[{actual_idx}] mlp has no down_proj")

    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    if gate_proj is None or up_proj is None:
        raise RuntimeError(f"transformer layer[{actual_idx}] mlp missing gate_proj/up_proj")

    act_fn = getattr(mlp, "act_fn", None)
    if act_fn is None:
        act_fn = F.silu

    hook = mlp.register_forward_pre_hook(_pre_hook)
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
        raise RuntimeError(f"failed to capture layer[{actual_idx}] mlp input activation")

    x = x[:, -response_length - 1 : -1, :]
    if hvp_token_stride > 1:
        x = x[:, ::hvp_token_stride, :]

    outs = []
    T = x.size(1)
    if chunk_tokens <= 0:
        chunk_tokens = T

    # Keep proxy loss connected to trainable params so autograd/HVP never sees an all-None first-order set.
    # Zero-valued anchors do not change loss magnitude or logging-axis semantics.
    anchor = None
    for p in actor_module.parameters():
        if p is not None and p.requires_grad:
            z = p.reshape(-1)[0] * 0.0
            anchor = z if anchor is None else (anchor + z)

    with torch.enable_grad():
        for s in range(0, T, chunk_tokens):
            e = min(s + chunk_tokens, T)
            xc = x[:, s:e, :]
            target_dtype = down_proj.weight.dtype
            if xc.dtype != target_dtype:
                xc = xc.to(target_dtype)
            xc = xc.float()
            gate = F.linear(xc, gate_proj.weight.float(), (None if gate_proj.bias is None else gate_proj.bias.float()))
            up = F.linear(xc, up_proj.weight.float(), (None if up_proj.bias is None else up_proj.bias.float()))
            hidden = act_fn(gate) * up
            y = F.linear(hidden, down_proj.weight.float(), (None if down_proj.bias is None else down_proj.bias.float()))
            proxy = 0.5 * y.float().pow(2).mean(dim=-1)
            if temperature != 1.0:
                proxy = proxy / float(temperature)
            if anchor is not None:
                proxy = proxy + anchor
            outs.append(proxy)

    return torch.cat(outs, dim=1)


def build_last_down_proj_proxy_logprob(**kwargs):
    kwargs.setdefault("hvp_target_layer_idx", -1)
    return build_layer_down_proj_proxy_logprob(**kwargs)
