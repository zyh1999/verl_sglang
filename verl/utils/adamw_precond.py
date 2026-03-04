import torch



class AdamWPrecond(torch.optim.AdamW):
    def __init__(self, params, *args, log_precond_stats: bool = False, precond_stat_prefix: str = "actor", **kwargs):
        super().__init__(params, *args, **kwargs)
        self.log_precond_stats = bool(log_precond_stats)
        self.precond_stat_prefix = precond_stat_prefix

        self.optim_step = 0
        self.current_dense_step = False
        self._last_precond_stats = {}
        self._dense_series_buffer = []

    def set_dense_step(self, dense_step: bool):
        self.current_dense_step = bool(dense_step)

    def _is_dense_step(self):
        # Dense-window control is decided by the trainer/actor and injected via
        # set_dense_step(). Do not infer from optim_step/global_step here.
        return bool(self.current_dense_step)

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure=closure)
        self.optim_step += 1

        if not self.log_precond_stats:
            self._last_precond_stats = {}
            return loss

        # collect/report only when the current global step is marked dense by upstream
        if not self._is_dense_step():
            self._last_precond_stats = {}
            self._dense_series_buffer = []
            return loss

        vals = []
        for group in self.param_groups:
            eps = float(group.get("eps", 1e-8))
            for p in group.get("params", []):
                if p is None:
                    continue
                st = self.state.get(p, None)
                if not st:
                    continue
                m = st.get("exp_avg", None)
                v = st.get("exp_avg_sq", None)
                if m is None or v is None:
                    continue
                u = (m.abs() / (v.sqrt() + eps)).mean()
                if torch.isfinite(u):
                    vals.append(float(u.item()))

        if vals:
            mean = float(sum(vals) / len(vals))
            var = float(sum((x - mean) ** 2 for x in vals) / len(vals))
            self._last_precond_stats = {
                f"{self.precond_stat_prefix}/precond_proxy_mean": mean,
                f"{self.precond_stat_prefix}/precond_proxy_std": var ** 0.5,
                f"{self.precond_stat_prefix}/precond_proxy_max": max(vals),
                f"{self.precond_stat_prefix}/precond_proxy_n": float(len(vals)),
            }
            self._dense_series_buffer.append(
                {
                    "optim_step": float(self.optim_step),
                    "mean": mean,
                    "std": var ** 0.5,
                    "max": max(vals),
                    "n": float(len(vals)),
                }
            )

        else:
            self._last_precond_stats = {}

        return loss

    def get_last_precond_stats(self):
        return dict(self._last_precond_stats)

    def pop_dense_series(self):
        out = self._dense_series_buffer
        self._dense_series_buffer = []
        return out
