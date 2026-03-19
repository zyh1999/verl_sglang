import os
import torch
import verl.utils.precond_sharpness as ps


class AdamWPrecond(torch.optim.AdamW):
    def __init__(
        self,
        params,
        *args,
        log_precond_stats: bool = False,
        precond_stat_prefix: str = "actor",
        precond_n_power_iter: int = 5,
        precond_tol: float = 1e-3,
        **kwargs,
    ):
        super().__init__(params, *args, **kwargs)
        self.log_precond_stats = bool(log_precond_stats)
        self.precond_stat_prefix = precond_stat_prefix
        self.precond_n_power_iter = int(precond_n_power_iter)
        self.precond_tol = float(precond_tol)

        self.optim_step = 0
        self.current_dense_step = False
        self._last_precond_stats = {}
        self._dense_series_buffer = []
        self._named_params_cache = []
        self._v_cache = {}

    def set_dense_step(self, dense_step: bool):
        self.current_dense_step = bool(dense_step)

    def set_named_params(self, named_params):
        self._named_params_cache = list(named_params) if named_params is not None else []

    def _is_dense_step(self):
        return bool(self.current_dense_step)

    def _pick_three_blocks(self):
        named = [(n, p) for n, p in self._named_params_cache if p is not None and p.requires_grad]
        if not named:
            return {}

        blocks = ps._split_blocks_by_transformer_layer(named)
        layer_blocks = [(k, v) for k, v in blocks if k.startswith("layer_") and len(v) > 0]

        chosen = {}
        if len(layer_blocks) >= 3:
            chosen = {
                "front": (layer_blocks[0][0], list(layer_blocks[0][1])),
                "mid": (layer_blocks[len(layer_blocks) // 2][0], list(layer_blocks[len(layer_blocks) // 2][1])),
                "back": (layer_blocks[-1][0], list(layer_blocks[-1][1])),
            }
        elif len(layer_blocks) > 0:
            tags = ["front", "mid", "back"]
            for i, (name, plist) in enumerate(layer_blocks[:3]):
                chosen[tags[i]] = (name, list(plist))
        else:
            vals = [p for _, p in named]
            u = ps._split_blocks(vals, 3)
            tags = ["front", "mid", "back"]
            for i, (uname, ps_blk) in enumerate(u[:3]):
                chosen[tags[i]] = (uname, list(ps_blk))

        return chosen

    def step(self, closure=None):
        loss = super().step(closure=closure)
        self.optim_step += 1

        if not self.log_precond_stats or (not self._is_dense_step()) or (closure is None):
            if self.log_precond_stats and self.optim_step <= 8:
                print("[precond_gate] step=%s dense=%s closure=%s skip=gate" % (self.optim_step, self._is_dense_step(), closure is not None), flush=True)
            self._last_precond_stats = {}
            return loss

        chosen = self._pick_three_blocks()

        hvp_mode = os.getenv("HVP_LOCAL_GRAPH_MODE", "").lower()
        if hvp_mode == "lm_head_only":
            forced_params = getattr(self, "_forced_hvp_params", None)
            lm_params = [p for (n, p) in self._named_params_cache if (p is not None and "lm_head" in n)]
            if forced_params:
                chosen = {"back": ("lm_head", list(forced_params))}
            elif lm_params:
                chosen = {"back": ("lm_head", lm_params)}
            elif "back" in chosen:
                chosen = {"back": chosen["back"]}
            elif chosen:
                _k = list(chosen.keys())[-1]
                chosen = {_k: chosen[_k]}

        if self.optim_step <= 8:
            try:
                print("[precond_gate] step=%s dense=%s mode=%s chosen_keys=%s" % (self.optim_step, self._is_dense_step(), hvp_mode, list(chosen.keys())), flush=True)
            except Exception:
                pass
        if not chosen:
            self._last_precond_stats = {}
            return loss

        per_block = {}
        per_block_raw = {}
        dense_payloads = []

        block_mode = os.getenv("HVP_BLOCK_MODE", "back_only")
        if block_mode == "back_only":
            if "back" in chosen:
                chosen = {"back": chosen["back"]}
            elif chosen:
                _k = next(iter(chosen.keys()))
                chosen = {_k: chosen[_k]}

        for tag, (_blk_name, blk_params_list) in chosen.items():
            blk_params = [p for p in blk_params_list if p is not None and p.requires_grad]
            if not blk_params:
                continue

            init_v = self._v_cache.get(tag)
            self._hvp_target_params = set(blk_params)
            lam_raw, vj = ps._power_iter_precond_block(
                evaluate_loss_fn=closure,
                block_params=blk_params,
                optimizer=self,
                n_power_iter=self.precond_n_power_iter,
                tol=self.precond_tol,
                init_v=init_v,
                sign_align=True,
                jitter=0.0,
            )
            self._hvp_target_params = None
            self._v_cache[tag] = vj
            per_block_raw[tag] = float(lam_raw)
            lam = float(max(lam_raw, 0.0))
            per_block[tag] = lam
            if os.getenv("HVP_DEBUG_LAM", "0") == "1":
                print(f"[precond_debug] step={self.optim_step} tag={tag} lam_raw={lam_raw} lam_clipped={lam}", flush=True)

            family = f"{self.precond_stat_prefix}/precond_proxy_update_{tag}"
            payload = {
                f"{self.precond_stat_prefix}/precond_proxy_update/optim_step": float(self.optim_step),
                f"{self.precond_stat_prefix}/precond_proxy_update/mean": float(lam),
                f"{self.precond_stat_prefix}/precond_proxy_update/raw": float(lam_raw),
                f"{self.precond_stat_prefix}/precond_proxy_update/std": 0.0,
                f"{self.precond_stat_prefix}/precond_proxy_update/max": float(lam),
                f"{self.precond_stat_prefix}/precond_proxy_update/n": float(len(blk_params)),
                f"{family}/optim_step": float(self.optim_step),
                f"{family}/mean": float(lam),
                f"{family}/raw": float(lam_raw),
                f"{family}/std": 0.0,
                f"{family}/max": float(lam),
                f"{family}/n": float(len(blk_params)),
            }
            dense_payloads.append(payload)

        if not per_block:
            self._last_precond_stats = {}
            return loss

        vals = list(per_block.values())
        raw_vals = list(per_block_raw.values()) if per_block_raw else vals
        stats = {
            f"{self.precond_stat_prefix}/precond_proxy_mean": float(sum(vals) / len(vals)),
            f"{self.precond_stat_prefix}/precond_proxy_max": float(max(vals)),
            f"{self.precond_stat_prefix}/precond_proxy_n": float(len(vals)),
            f"{self.precond_stat_prefix}/precond_proxy_raw_mean": float(sum(raw_vals) / len(raw_vals)),
            f"{self.precond_stat_prefix}/precond_proxy_raw_max": float(max(raw_vals)),
        }
        for tag, v in per_block.items():
            stats[f"{self.precond_stat_prefix}/precond_sharpness/{tag}"] = float(v)

        self._last_precond_stats = stats
        self._dense_series_buffer.extend(dense_payloads)
        return loss

    def get_last_precond_stats(self):
        return dict(self._last_precond_stats)

    def pop_dense_series(self):
        out = self._dense_series_buffer
        self._dense_series_buffer = []
        return out
