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
"""
Critical sharpness estimation utilities.

Implements the line-search procedure in
"A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs":
    eta_c = inf {eta > 0 | L(theta - eta * Delta_theta) > L(theta)}
    lambda_c = 2 / eta_c
"""

from collections.abc import Callable

import torch
import torch.distributed as dist

__all__ = ["estimate_critical_sharpness"]


@torch.no_grad()
def _set_params_from_snapshots(params, snapshots):
    for p, s in zip(params, snapshots, strict=False):
        p.copy_(s.to(p.dtype))


@torch.no_grad()
def _set_params_along_direction(params, theta0, direction, eta: float):
    for p, p0, d in zip(params, theta0, direction, strict=False):
        p.copy_((p0.to(p.dtype) - eta * d.to(p.dtype)))


@torch.no_grad()
def estimate_critical_sharpness(
    *,
    params: list[torch.nn.Parameter],
    theta_before: list[torch.Tensor],
    theta_after: list[torch.Tensor],
    evaluate_loss_fn: Callable[[], float],
    eta_init: float = 1.0,
    max_expand_steps: int = 40,
    max_binary_steps: int = 20,
    binary_tol: float = 1e-2,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Estimate critical learning rate and critical sharpness along update direction."""
    if len(params) != len(theta_before) or len(params) != len(theta_after):
        return {}

    direction = [b - a for b, a in zip(theta_before, theta_after, strict=False)]  # Delta_theta
    direction_sq_norm = sum(torch.sum(d.float() * d.float()).item() for d in direction)
    # All-reduce so all ranks use the same global norm; avoids collective mismatch when
    # FSDP shards params (each rank has different local direction_sq_norm).
    if dist.is_initialized():
        buf = torch.tensor([direction_sq_norm], device=params[0].device, dtype=torch.float64)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        direction_sq_norm = buf.item()
    if direction_sq_norm <= eps:
        return {}

    try:
        _set_params_from_snapshots(params, theta_before)
        base_loss = float(evaluate_loss_fn())

        eta = max(float(eta_init), float(eps))
        _set_params_along_direction(params, theta_before, direction, eta)
        loss_eta = float(evaluate_loss_fn())

        lower = None
        upper = None
        if loss_eta > base_loss:
            upper = eta
            for _ in range(max_expand_steps):
                eta = eta / 2.0
                if eta <= eps:
                    lower = 0.0
                    break
                _set_params_along_direction(params, theta_before, direction, eta)
                loss_eta = float(evaluate_loss_fn())
                if loss_eta <= base_loss:
                    lower = eta
                    break
            if lower is None:
                lower = 0.0
        else:
            lower = eta
            for _ in range(max_expand_steps):
                eta = eta * 2.0
                _set_params_along_direction(params, theta_before, direction, eta)
                loss_eta = float(evaluate_loss_fn())
                if loss_eta > base_loss:
                    upper = eta
                    break

        if upper is None:
            return {}

        for _ in range(max_binary_steps):
            if upper <= eps:
                break
            if lower > 0 and abs(1.0 - lower / upper) <= binary_tol:
                break
            mid = 0.5 * (lower + upper)
            _set_params_along_direction(params, theta_before, direction, mid)
            loss_mid = float(evaluate_loss_fn())
            if loss_mid > base_loss:
                upper = mid
            else:
                lower = mid

        eta_c = 0.5 * (lower + upper)
        lambda_c = 2.0 / max(eta_c, eps)
        return {
            "actor/critical_lr": float(eta_c),
            "actor/critical_sharpness": float(lambda_c),
            "actor/critical_loss_base": float(base_loss),
        }
    finally:
        # Keep actor weights at post-update state for normal training flow.
        _set_params_from_snapshots(params, theta_after)
