import math
from typing import Any

import torch


def _get_step_value(step: Any) -> int:
    if step is None:
        return 0
    if isinstance(step, torch.Tensor):
        return int(step.item())
    return int(step)


def _compute_param_snr(
    exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, step: int, beta1: float, beta2: float, eps: float
) -> torch.Tensor:
    if step <= 0:
        return torch.empty(0, device=exp_avg.device, dtype=torch.float32)

    beta1_correction = 1.0 - math.pow(beta1, step)
    beta2_correction = 1.0 - math.pow(beta2, step)
    m_hat = exp_avg.float() / beta1_correction
    v_hat = exp_avg_sq.float() / beta2_correction
    noise_var = torch.clamp(v_hat - m_hat.square(), min=0.0)
    return m_hat.abs() / torch.sqrt(noise_var + eps)


def compute_adam_snr_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    values = []
    for group in optimizer.param_groups:
        eps = float(group.get("eps", 1e-8))
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        if beta1 <= 0 or beta1 >= 1 or beta2 <= 0 or beta2 >= 1:
            continue
        for param in group["params"]:
            state = optimizer.state.get(param, {})
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            step = _get_step_value(state.get("step"))
            if exp_avg is None or exp_avg_sq is None or step <= 0:
                continue
            snr = _compute_param_snr(exp_avg, exp_avg_sq, step, beta1, beta2, eps)
            if snr.numel() > 0:
                values.append(snr.reshape(-1).cpu())

    if not values:
        return {}

    flat = torch.cat(values)
    return {
        "adam_nsr/mean": float(flat.mean().item()),
        "adam_nsr/std": float(flat.std(unbiased=False).item()),
        "adam_nsr/max": float(flat.max().item()),
        "adam_nsr/min": float(flat.min().item()),
        "adam_nsr/n": float(flat.numel()),
    }
