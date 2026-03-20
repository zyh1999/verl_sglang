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
"""Adam NSR/moment metric utilities."""

import torch
from torch.distributed.tensor import DTensor

__all__ = ["compute_adam_snr_metrics"]


def _to_local(x):
    if isinstance(x, DTensor):
        return x._local_tensor
    return x


def _acc_basic(stats: dict, name: str, x: torch.Tensor):
    n = x.numel()
    if n == 0:
        return
    stats[f"{name}/count"] += n
    stats[f"{name}/sum"] += x.sum().item()
    x_max = x.max().item()
    x_min = x.min().item()
    if x_max > stats[f"{name}/max"]:
        stats[f"{name}/max"] = x_max
    if x_min < stats[f"{name}/min"]:
        stats[f"{name}/min"] = x_min


def _emit_basic(stats: dict, name: str, out: dict):
    n = stats[f"{name}/count"]
    if n <= 0:
        return
    out[f"{name}/mean"] = stats[f"{name}/sum"] / n
    out[f"{name}/min"] = stats[f"{name}/min"]
    out[f"{name}/max"] = stats[f"{name}/max"]


@torch.no_grad()
def compute_adam_snr_metrics(optimizer) -> dict[str, float]:
    """Compute Adam/Adam-NSR diagnostics.

    Includes:
    - adam_nsr/*
    - adam_m1_abs/*
    - adam_m1_sq_abs/* (for tiny-threshold statistics)
    - adam_m2_abs/*
    - adam_m2_minus_m1sq_abs/* = |v_hat - m_hat^2|  (aligned with NSR numerator)

    Tiny-threshold fractions are logged for thresholds 1e-18 ... 1e-8.
    """
    if optimizer is None:
        return {}

    eps = 1e-30

    is_nsr_optimizer = False
    for p in optimizer.state:
        if "nsr_sq" in optimizer.state[p]:
            is_nsr_optimizer = True
        break

    nsr_thresholds = (0.5, 1.0, *tuple(float(i) for i in range(2, 21)), 50.0, 100.0, 500.0, 1000.0)
    tiny_thresholds = tuple(10.0**e for e in range(-18, -7))  # 1e-18 ... 1e-8

    metric_names = [
        "adam_nsr",
        "adam_m1_abs",
        "adam_m1_sq_abs",
        "adam_m2_abs",
        "adam_m2_minus_m1sq_abs",
    ]
    stats = {}
    for n in metric_names:
        stats[f"{n}/count"] = 0
        stats[f"{n}/sum"] = 0.0
        stats[f"{n}/max"] = float("-inf")
        stats[f"{n}/min"] = float("inf")

    count_lt_nsr = {t: 0 for t in nsr_thresholds}
    count_lt_m1sq = {t: 0 for t in tiny_thresholds}
    count_lt_m2 = {t: 0 for t in tiny_thresholds}
    count_lt_m2m1sq = {t: 0 for t in tiny_thresholds}

    for param_group in optimizer.param_groups:
        beta1, beta2 = param_group.get("betas", (0.9, 0.999))

        for param in param_group["params"]:
            state = optimizer.state.get(param)
            if state is None or "exp_avg" not in state:
                continue

            m = _to_local(state["exp_avg"]).float()
            step = state.get("step", 1)
            step_f = float(step) if not isinstance(step, torch.Tensor) else step.float().item()
            if step_f < 1:
                continue

            bc1 = 1.0 - beta1**step_f
            m_hat = m / bc1

            if is_nsr_optimizer and "nsr_sq" in state:
                r = _to_local(state["nsr_sq"]).float().clamp(min=0.0)
                bc2 = 1.0 - beta2**step_f
                v_hat = (m.square() * (1.0 + r)) / bc2
                nsr = torch.sqrt(r)
            else:
                if "exp_avg_sq" not in state:
                    continue
                v = _to_local(state["exp_avg_sq"]).float()
                bc2 = 1.0 - beta2**step_f
                v_hat = v / bc2
                var = torch.clamp(v_hat - m_hat.square(), min=0.0)
                nsr = torch.sqrt(var) / (m_hat.abs() + eps)

            m1_abs = m_hat.abs()
            m1_sq_abs = m_hat.square().abs()
            m2_abs = v_hat.abs()
            m2m1sq_abs = (v_hat - m_hat.square()).abs()

            _acc_basic(stats, "adam_nsr", nsr)
            _acc_basic(stats, "adam_m1_abs", m1_abs)
            _acc_basic(stats, "adam_m1_sq_abs", m1_sq_abs)
            _acc_basic(stats, "adam_m2_abs", m2_abs)
            _acc_basic(stats, "adam_m2_minus_m1sq_abs", m2m1sq_abs)

            for t in nsr_thresholds:
                count_lt_nsr[t] += (nsr < t).sum().item()
            for t in tiny_thresholds:
                count_lt_m1sq[t] += (m1_sq_abs < t).sum().item()
                count_lt_m2[t] += (m2_abs < t).sum().item()
                count_lt_m2m1sq[t] += (m2m1sq_abs < t).sum().item()

    if stats["adam_nsr/count"] == 0:
        return {}

    out = {}
    for n in metric_names:
        _emit_basic(stats, n, out)

    n_nsr = stats["adam_nsr/count"]
    for t in nsr_thresholds:
        key = "adam_nsr/frac_lt_" + (str(int(t)) if t == int(t) else str(t).replace(".", "p"))
        out[key] = count_lt_nsr[t] / n_nsr

    n_m1sq = stats["adam_m1_sq_abs/count"]
    n_m2 = stats["adam_m2_abs/count"]
    n_m2m1sq = stats["adam_m2_minus_m1sq_abs/count"]
    for t in tiny_thresholds:
        exp = int(round(-torch.log10(torch.tensor(t)).item()))
        tag = f"1em{exp}"
        out[f"adam_m1_sq_abs/frac_lt_{tag}"] = count_lt_m1sq[t] / n_m1sq if n_m1sq > 0 else 0.0
        out[f"adam_m2_abs/frac_lt_{tag}"] = count_lt_m2[t] / n_m2 if n_m2 > 0 else 0.0
        out[f"adam_m2_minus_m1sq_abs/frac_lt_{tag}"] = count_lt_m2m1sq[t] / n_m2m1sq if n_m2m1sq > 0 else 0.0

    return out
