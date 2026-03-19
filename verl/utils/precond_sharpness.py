from __future__ import annotations

from typing import Callable
import os
import re
import traceback

import torch


def _fmt_cuda_mem(tag: str) -> str:
    if not torch.cuda.is_available():
        return f"{tag} cuda=na"
    d = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(d) / (1024**3)
    reserv = torch.cuda.memory_reserved(d) / (1024**3)
    peak = torch.cuda.max_memory_allocated(d) / (1024**3)
    return f"{tag} cuda_alloc={alloc:.3f}GiB reserved={reserv:.3f}GiB peak={peak:.3f}GiB"


__all__ = [
    "should_compute_precond_sharpness",
    "estimate_precond_sharpness",
    "aggregate_rollout_precond_sharpness",
]


def should_compute_precond_sharpness(*, enabled: bool, rollout_step: int, interval: int) -> bool:
    if not enabled:
        return False
    interval = max(int(interval), 1)
    rollout_step = int(rollout_step)
    # Debug-friendly behavior: always record at first rollout step,
    # then follow interval cadence.
    return rollout_step == 1 or (rollout_step > 0 and (rollout_step % interval == 0))


def _dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))


def _norm(vs: list[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum((v * v).sum() for v in vs) + 1e-16)


def _scale(vs: list[torch.Tensor], s: torch.Tensor) -> list[torch.Tensor]:
    return [v / (s + 1e-16) for v in vs]


def _detach(vs: list[torch.Tensor]) -> list[torch.Tensor]:
    return [v.detach() for v in vs]


def _split_blocks(params: list[torch.nn.Parameter], num_blocks: int) -> list[tuple[str, list[torch.nn.Parameter]]]:
    n = len(params)
    if n == 0:
        return []
    num_blocks = max(1, min(int(num_blocks), n))
    base = n // num_blocks
    rem = n % num_blocks
    blocks: list[tuple[str, list[torch.nn.Parameter]]] = []
    i = 0
    for b in range(num_blocks):
        take = base + (1 if b < rem else 0)
        blk = params[i : i + take]
        i += take
        if blk:
            blocks.append((f"uniform_{b}", blk))
    return blocks


@torch.no_grad()
def _build_adam_diag(optimizer: torch.optim.Optimizer, block_params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    group_of = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            group_of[id(p)] = group

    denom = []
    for p in block_params:
        g = group_of.get(id(p), None)
        eps = (g or {}).get("eps", 1e-8)
        beta2 = (g or {}).get("betas", (0.9, 0.999))[1]
        amsgrad = (g or {}).get("amsgrad", False)
        st = optimizer.state.get(p, {})
        step = int(st.get("step", 0))
        if amsgrad and ("max_exp_avg_sq" in st):
            vbuf = st["max_exp_avg_sq"]
        else:
            vbuf = st.get("exp_avg_sq", None)
        if vbuf is None:
            d = torch.full_like(p, float(eps))
        else:
            bc2 = 1.0 - (beta2 ** max(step, 1))
            if bc2 <= 1e-16:
                bc2 = 1.0
            v_hat = vbuf / bc2
            d = torch.sqrt(v_hat).to(dtype=p.dtype, device=p.device) + float(eps)
        denom.append(d)
    return denom


def _split_blocks_by_transformer_layer(named_params: list[tuple[str, torch.nn.Parameter]]) -> list[tuple[str, list[torch.nn.Parameter]]]:
    layer_to_params: dict[str, list[torch.nn.Parameter]] = {}
    other: list[torch.nn.Parameter] = []

    patterns = [
        re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)"),
        re.compile(r"(?:^|\.)transformer\.h\.(\d+)(?:\.|$)"),
        re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    ]

    for name, p in named_params:
        key = None
        for pat in patterns:
            m = pat.search(name)
            if m is not None:
                key = f"layer_{int(m.group(1))}"
                break
        if key is None:
            other.append(p)
        else:
            layer_to_params.setdefault(key, []).append(p)

    blocks = [(k, layer_to_params[k]) for k in sorted(layer_to_params.keys(), key=lambda x: int(x.split("_")[1]))]
    if other:
        blocks.append(("other", other))
    return [(k, v) for k, v in blocks if v]




def _coerce_loss_tensor(loss_obj) -> torch.Tensor:
    """Coerce common loss container outputs into a scalar tensor loss."""
    if isinstance(loss_obj, torch.Tensor):
        return loss_obj

    if isinstance(loss_obj, dict):
        for k in ("loss", "total_loss", "actor_loss", "pg_loss"):
            v = loss_obj.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        for v in loss_obj.values():
            if isinstance(v, torch.Tensor):
                return v

    if isinstance(loss_obj, (tuple, list)):
        for v in loss_obj:
            if isinstance(v, torch.Tensor):
                return v

    if isinstance(loss_obj, (float, int)):
        # Not differentiable; return tensor for consistent error handling below.
        return torch.tensor(float(loss_obj))

    raise TypeError(f"Unsupported loss return type: {type(loss_obj)}")


def _hvp_block(
    *,
    evaluate_loss_fn: Callable[[], torch.Tensor],
    block_params: list[torch.nn.Parameter],
    vec: list[torch.Tensor],
) -> list[torch.Tensor]:
    mem_debug = (os.getenv("HVP_MEM_DEBUG", "0") == "1")
    if mem_debug and torch.cuda.is_available():
        print(f"[hvp_mem] {_fmt_cuda_mem('pre_hvp_loss')}", flush=True)
    loss_obj = evaluate_loss_fn()
    loss = _coerce_loss_tensor(loss_obj)
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"Loss must be Tensor after coercion, got {type(loss)}")
    if not loss.requires_grad:
        raise TypeError(
            f"Loss tensor requires_grad=False (type={type(loss_obj)}). "
            "Please return a differentiable tensor loss from evaluate_loss_fn()."
        )
    g = torch.autograd.grad(loss, block_params, create_graph=True, allow_unused=True)
    if mem_debug:
        print(f"[hvp_mem] {_fmt_cuda_mem('post_first_grad')}", flush=True)
    used = [gi for gi in g if gi is not None]
    if len(used) == 0:
        shapes = [tuple(p.shape) for p in block_params[:4]]
        names = []
        try:
            named = list(getattr(optimizer, "_named_params_cache", []))
            idset = {id(p) for p in block_params}
            names = [n for (n, p) in named if id(p) in idset][:8]
        except Exception:
            names = []
        req = [bool(getattr(p, "requires_grad", False)) for p in block_params[:8]]
        print("[precond_sharpness][debug] all_none_first_grad: n_params=%s sample_shapes=%s sample_requires_grad=%s sample_names=%s" % (len(block_params), shapes, req, names), flush=True)
        raise RuntimeError("all first-order grads are None in HVP path")
    dot = sum(
        ((gi if gi is not None else torch.zeros_like(p)).flatten().dot(v.flatten()))
        for gi, v, p in zip(g, vec, block_params)
    )
    if not dot.requires_grad:
        non_none = sum(1 for gi in g if gi is not None)
        req = sum(1 for gi in g if (gi is not None and gi.requires_grad))
        p0 = block_params[0] if block_params else None
        p0_type = type(p0).__name__ if p0 is not None else 'None'
        if p0 is not None:
            print(f"[precond_sharpness][debug] dot.no_grad: non_none_g={non_none}/{len(g)} req_g={req} p0_type={p0_type} p0_req={getattr(p0,'requires_grad',None)}", flush=True)
        raise RuntimeError("dot does not require grad in HVP path")
    hv = torch.autograd.grad(dot, block_params, retain_graph=False, allow_unused=True)
    if mem_debug:
        print(f"[hvp_mem] {_fmt_cuda_mem('post_hvp_grad')}", flush=True)
    return [(hi if hi is not None else torch.zeros_like(p)) for hi, p in zip(hv, block_params)]


def _power_iter_precond_block(
    *,
    evaluate_loss_fn: Callable[[], torch.Tensor],
    block_params: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    n_power_iter: int,
    tol: float,
    init_v: list[torch.Tensor] | None = None,
    sign_align: bool = True,
    jitter: float = 0.0,
) -> tuple[float, list[torch.Tensor]]:
    denom = _build_adam_diag(optimizer, block_params)
    denom_sqrt = [torch.sqrt(d) for d in denom]

    def apply_A(v: list[torch.Tensor]) -> list[torch.Tensor]:
        x = [vi / ds for vi, ds in zip(v, denom_sqrt)]
        hv = _hvp_block(evaluate_loss_fn=evaluate_loss_fn, block_params=block_params, vec=x)
        z = [hvi / ds for hvi, ds in zip(hv, denom_sqrt)]
        return z

    if init_v is not None and len(init_v) == len(block_params) and all(iv.shape == p.shape for iv, p in zip(init_v, block_params)):
        v = [iv.to(device=p.device, dtype=p.dtype) for iv, p in zip(init_v, block_params)]
        if jitter > 0:
            v = [vi + jitter * torch.randn_like(vi) for vi in v]
    else:
        v = [torch.randn_like(p) for p in block_params]

    v = _scale(v, _norm(v))
    lam = None
    for _ in range(max(int(n_power_iter), 1)):
        Av = apply_A(v)
        if sign_align and float(_dot(v, Av).detach().item()) < 0:
            Av = [-x for x in Av]
        rq = _dot(v, Av)
        new_lam = float(rq.detach().item())
        nrm = _norm(Av)
        if float(nrm.detach().item()) == 0.0:
            return 0.0, _detach(v)
        v = _scale(Av, nrm)
        if lam is not None and abs(new_lam - lam) <= tol * (abs(lam) + 1e-12):
            lam = new_lam
            break
        lam = new_lam
    return float(lam if lam is not None else 0.0), _detach(v)


def estimate_precond_sharpness(
    *,
    params: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    evaluate_loss_fn: Callable[[], torch.Tensor],
    num_blocks: int = 8,
    n_power_iter: int = 5,
    tol: float = 1e-3,
    named_params: list[tuple[str, torch.nn.Parameter]] | None = None,
    init_v_blocks: dict[str, list[torch.Tensor]] | None = None,
    sign_align: bool = True,
    jitter: float = 0.0,
) -> tuple[dict[str, float], dict[str, list[torch.Tensor]]]:
    if not params:
        return {}, {}

    # debug mode: compute only one block to reduce peak memory.
    if named_params:
        all_blocks = _split_blocks_by_transformer_layer(named_params)
        if all_blocks:
            blocks = [all_blocks[0]]
        else:
            blocks = [("full_head", params[:1])]
    else:
        blocks = [("full_head", params[:1])]

    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        lams: list[float] = []
        v_out: dict[str, list[torch.Tensor]] = {}
        for key, blk in blocks:
            init_v = init_v_blocks.get(key) if init_v_blocks else None
            lam, vj = _power_iter_precond_block(
                evaluate_loss_fn=evaluate_loss_fn,
                block_params=blk,
                optimizer=optimizer,
                n_power_iter=n_power_iter,
                tol=tol,
                init_v=init_v,
                sign_align=sign_align,
                jitter=jitter,
            )
            if os.getenv("HVP_DEBUG_LAM", "0") == "1":
                print(f"[precond_sharpness][lam_raw] block={key} lam={lam}", flush=True)
            lams.append(float(lam))
            v_out[key] = vj

        if not lams:
            return {}, v_out

        best = max(lams)
        mean = float(sum(lams) / len(lams))
        var = float(sum((x - mean) ** 2 for x in lams) / len(lams))
        return {
            "actor/precond_sharpness": float(best),
            "actor/precond_sharpness_mean": mean,
            "actor/precond_sharpness_std": float(var**0.5),
            "actor/precond_sharpness_num_blocks": float(len(lams)),
            "actor/precond_sharpness_block_mode": 1.0,
        }, v_out
    except Exception as e:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        except Exception:
            rank = 0
        if rank == 0:
            print(f"[precond_sharpness][error] {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
        return {}, {}
    finally:
        torch.set_grad_enabled(prev)


def aggregate_rollout_precond_sharpness(samples: list[dict[str, float]], *, rollout_step: int | None = None) -> dict[str, float]:
    if not samples:
        return {}
    vals = [float(m["actor/precond_sharpness"]) for m in samples if "actor/precond_sharpness" in m]
    if not vals:
        return {}
    mean = float(sum(vals) / len(vals))
    var = float(sum((x - mean) ** 2 for x in vals) / len(vals))
    out = {
        "actor/precond_sharpness": mean,
        "actor/precond_sharpness_std": float(var**0.5),
        "actor/precond_sharpness_num_minibatches": float(len(vals)),
    }
    if rollout_step is not None:
        out["actor/precond_sharpness_rollout_step"] = float(rollout_step)
    return out
