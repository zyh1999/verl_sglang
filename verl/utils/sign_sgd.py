"""
SignSGD optimizers.

Two variants:
1) SignSGD:         update uses sign(grad)
2) SignSGD_Momentum update uses sign(momentum_buffer), momentum buffer follows SGD momentum

We keep the signature compatible with verl's dynamic optimizer builder:
- accepts lr, weight_decay
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["SignSGD", "SignSGD_Momentum", "Signum"]


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


class SignSGD_Momentum(Optimizer):
    """SignSGD with momentum (SGD-style) and optional decoupled weight decay.

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
                    raise RuntimeError("SignSGD_Momentum does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                p.data.add_(buf.sign(), alpha=-lr)

        return loss



class Signum(Optimizer):
    """Signum optimizer with momentum and optional decoupled weight decay.

    Reference-style update:
      buf <- momentum * buf + (1 - momentum) * grad
      p   <- (1 - lr * wd) * p
      p   <- p - lr * sign(buf)

    Notes:
      - momentum=0 reduces to SignSGD.
      - sparse gradients are not supported.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
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
                    raise RuntimeError("Signum does not support sparse gradients")

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                if mu == 0.0:
                    p.data.add_(g.sign(), alpha=-lr)
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g, alpha=(1.0 - mu))
                p.data.add_(buf.sign(), alpha=-lr)

        return loss
