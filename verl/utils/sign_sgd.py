"""
SignSGD optimizers.

Notes:
- Keep one canonical momentum+weight-decay implementation:
  SignSGD_Momentum_WD (decoupled WD).
- SignSGD_Momentum is retained as a thin compatibility wrapper to avoid
  breaking existing scripts, but it reuses the same decoupled implementation.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["SignSGD", "SignSGD_Momentum_", "SignSGD_Momentum_WD", "SignSGD_Momentum_Safe", "Signum"]


class SignSGD(Optimizer):
    """SignSGD with optional decoupled weight decay.

    Update:
      p <- (1 - lr * wd) * p
      p <- p - lr * sign(g)
    """

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("SignSGD does not support sparse gradients")

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                p.data.add_(g.sign(), alpha=-lr)

        return loss


class SignSGD_Momentum_WD(Optimizer):
    """SignSGD with momentum and decoupled weight decay (canonical WD variant).

    Momentum buffer:
      buf <- momentum * buf + grad
    Update:
      p <- (1 - lr * wd) * p
      p <- p - lr * sign(buf)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("SignSGD_Momentum_WD does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                p.data.add_(buf.sign(), alpha=-lr)

        return loss



class SignSGD_Momentum_(SignSGD_Momentum_WD):
    """SignSGD momentum variant without weight decay.

    Convenience wrapper: identical to SignSGD_Momentum_WD but forces
    weight_decay=0.0 regardless of caller input.
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(params=params, lr=lr, momentum=momentum, weight_decay=0.0)


class SignSGD_Momentum_Safe(Optimizer):
    """SignSGD with SGD momentum + safe-region downscale for tiny momentum buffers."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        momentum_safe_eps: float = 1e-8,
        unsafe_update_scale: float = 0.1,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if momentum_safe_eps < 0.0:
            raise ValueError(f"Invalid momentum_safe_eps: {momentum_safe_eps}")
        if unsafe_update_scale < 0.0:
            raise ValueError(f"Invalid unsafe_update_scale: {unsafe_update_scale}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            momentum_safe_eps=momentum_safe_eps,
            unsafe_update_scale=unsafe_update_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            eps_m = group["momentum_safe_eps"]
            unsafe_scale = group["unsafe_update_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("SignSGD_Momentum_Safe does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                update = buf.sign()
                if unsafe_scale != 1.0:
                    safe = buf.abs() > eps_m
                    update = torch.where(safe, update, update * unsafe_scale)

                p.data.add_(update, alpha=-lr)

        return loss


class Signum(Optimizer):
    """Signum optimizer with momentum and optional decoupled weight decay.

    Safe-region boost:
    - If |momentum_buffer| <= momentum_safe_eps, temporarily scale lr by
      safe_region_lr_scale (default 10x) for that element update.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        momentum_safe_eps: float = 1e-5,
        safe_region_lr_scale: float = 10.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if momentum_safe_eps < 0.0:
            raise ValueError(f"Invalid momentum_safe_eps: {momentum_safe_eps}")
        if safe_region_lr_scale < 0.0:
            raise ValueError(f"Invalid safe_region_lr_scale: {safe_region_lr_scale}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            momentum_safe_eps=momentum_safe_eps,
            safe_region_lr_scale=safe_region_lr_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            eps_m = group.get("momentum_safe_eps", 1e-5)
            lr_scale = group.get("safe_region_lr_scale", 10.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Signum does not support sparse gradients")

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                if mu == 0.0:
                    sign_dir = g.sign()
                    safe_mask = g.abs() <= eps_m
                    eff_scale = torch.where(safe_mask, torch.full_like(sign_dir, lr_scale), torch.ones_like(sign_dir))
                    p.data.add_(sign_dir * eff_scale, alpha=-lr)
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g, alpha=(1.0 - mu))

                sign_dir = buf.sign()
                safe_mask = buf.abs() <= eps_m
                eff_scale = torch.where(safe_mask, torch.full_like(sign_dir, lr_scale), torch.ones_like(sign_dir))
                p.data.add_(sign_dir * eff_scale, alpha=-lr)

        return loss

