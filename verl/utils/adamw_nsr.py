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
AdamW with Noise-to-Signal Ratio (NSR) state representation.

Based on "In Search of Adam's Secret Sauce" (Orvieto & Gower, NeurIPS 2025).

Instead of the standard Adam states (m, v) where:
    m_t = β m_{t-1} + (1-β) g_t          (first moment)
    v_t = β v_{t-1} + (1-β) g_t²         (second moment)

This optimizer stores (m, r) where r = NSR² = (v - m²) / m², and uses
the equivalent recursion:
    u_t = (g_t - m_{t-1}) / m_{t-1}       (innovation)
    m_t = m_{t-1} * (1 + (1-β) u_t)
    r_t = (β r_{t-1} + β(1-β) u_t²) / (1 + (1-β) u_t)²

The Adam update then decomposes as:
    step = sign(m̂) / sqrt(1 + NSR²) = sign(m̂) / sqrt(1 + r)

Key property: r has a much narrower dynamic range than v, making it a
better candidate for future low-precision optimization.

Requires β₁ = β₂ (equal betas).
"""

import math

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["AdamW_NSR"]


class AdamW_NSR(Optimizer):
    """AdamW optimizer with NSR (Noise-to-Signal Ratio) state representation.

    Mathematically equivalent to standard AdamW when β₁ = β₂.
    Stores (m, r) instead of (m, v), where r = (v - m²) / m² ≈ NSR².

    Args:
        params: Iterable of parameters or dicts defining parameter groups.
        lr (float): Learning rate. Default: 1e-3.
        betas (tuple): Must satisfy β₁ = β₂. Default: (0.92, 0.92).
        eps (float): Epsilon for the Adam denominator. Default: 1e-8.
        weight_decay (float): Decoupled weight decay. Default: 0.01.
        eps_m (float): Small constant to prevent division by zero in
            innovation computation when m ≈ 0. Default: 1e-8.
        eps_r (float): Small constant added to denominator in r update
            to prevent division by zero. Default: 1e-8.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.92, 0.92),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        eps_m: float = 1e-8,
        eps_r: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if betas[0] != betas[1]:
            raise ValueError(
                f"AdamW_NSR requires β₁ = β₂ (equal betas), got betas={betas}. "
                f"The NSR recursion is only valid when both momentum parameters are equal."
            )
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, eps_m=eps_m, eps_r=eps_r)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            Optional loss value from the closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["betas"][0]  # β₁ = β₂
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            eps_m = group["eps_m"]
            eps_r = group["eps_r"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW_NSR does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["nsr_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m = state["exp_avg"]  # first moment
                r = state["nsr_sq"]  # NSR² = (v - m²) / m²
                state["step"] += 1
                step = state["step"]

                # ----------------------------------------------------------
                # Step 1: special-case initialization
                #   m starts at 0, and the multiplicative recursion
                #   m_new = m * (1 + (1-β)*u) cannot escape zero.
                #   So for the first step, use the standard formulas
                #   and derive the initial r analytically.
                # ----------------------------------------------------------
                # We maintain v = m²(1+r) as an invariant.
                # For the standard EMA update: m_new = β*m + (1-β)*g
                # We always compute m via the standard (stable) additive form,
                # then use the NSR recursion for r only where |m| is large enough.
                alpha = 1.0 - beta

                # --- Update m (always use stable additive form) ---
                m_old = m.clone()
                m.mul_(beta).add_(grad, alpha=alpha)  # m = β*m + (1-β)*g

                r_fallback = beta / alpha

                if step == 1:
                    # First step: r = β/(1-β) analytically
                    r.fill_(r_fallback)
                else:
                    # --- Update r via NSR recursion where numerically safe ---
                    # Safe mask: both |m_old| and |m_new| must be large enough.
                    # - |m_old| small → u = (g-m)/m explodes
                    # - |m_new| small → scale = m_new/m_old ≈ 0 → r denominator explodes
                    # Both cause r to blow up to huge values.
                    safe = (m_old.abs() > eps_m) & (m.abs() > eps_m)

                    # Compute innovation and scale only for safe elements
                    m_old_safe = m_old.clone()
                    m_old_safe[~safe] = 1.0  # dummy value to avoid div-by-zero
                    u = (grad - m_old) / m_old_safe
                    scale = 1.0 + alpha * u

                    # r_new = (β*r + β*(1-β)*u²) / (scale² + eps_r)
                    r_new = (beta * r + beta * alpha * u.square()) / (scale.square() + eps_r)

                    # For safe elements: use NSR recursion
                    # For unsafe elements: reset r = β/(1-β)
                    r.copy_(torch.where(safe, r_new, torch.full_like(r, r_fallback)))

                # ----------------------------------------------------------
                # Compute parameter update (equivalent to standard Adam)
                #
                # Standard Adam:  update = m̂ / (sqrt(v̂) + ε)
                # With v = m²(1+r) and bias correction bc = 1 - β^t:
                #   m̂ = m / bc
                #   sqrt(v̂) = |m| * sqrt((1+r) / bc)
                #   update = m̂ / (sqrt(v̂) + ε)
                #          = (m / bc) / (|m| * sqrt((1+r)/bc) + ε)
                # ----------------------------------------------------------
                bc = 1.0 - beta**step
                m_hat = m / bc
                denom = m.abs() * torch.sqrt((1.0 + r.clamp(min=0.0)) / bc) + eps
                update = m_hat / denom

                # AdamW: decoupled weight decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Apply update
                p.data.add_(update, alpha=-lr)

        return loss
