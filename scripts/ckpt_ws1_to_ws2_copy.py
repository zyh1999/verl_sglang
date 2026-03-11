#!/usr/bin/env python3
"""
Quick-and-dirty checkpoint "reshard" for verl FSDP checkpoints:

verl's FSDPCheckpointManager loads files by name:
  model_world_size_{world_size}_rank_{rank}.pt
  optim_world_size_{world_size}_rank_{rank}.pt
  extra_state_world_size_{world_size}_rank_{rank}.pt

For world_size=1, these files often contain the *full* state dict because there is no sharding.
If we want to resume with world_size=2 WITHOUT changing trainer code, one pragmatic way is:
  - copy the rank0 files into rank0+rank1 filenames for world_size=2.

This is not a true reshard; it relies on FSDP/state_dict loading behavior to properly shard at runtime.
It may fail for some optimizer state formats. Use at your own risk.

Usage:
  python ckpt_ws1_to_ws2_copy.py /abs/path/to/global_step_XXX/actor
"""

import json
import os
import shutil
import sys


FILES = (
    "model",
    "optim",
    "extra_state",
)


def pjoin(*parts: str) -> str:
    return os.path.abspath(os.path.join(*parts))


def ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def copy_if_missing(src: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"[skip] exists: {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[copy] {src} -> {dst}")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python ckpt_ws1_to_ws2_copy.py /abs/path/to/global_step_XXX/actor", file=sys.stderr)
        return 2

    actor_dir = os.path.abspath(sys.argv[1])
    ensure_exists(actor_dir)

    # minimal "spec" (keep python3.6 compatible: no dataclasses, no typing features)
    prefix = actor_dir
    src_world_size = 1
    src_rank = 0
    dst_world_size = 2

    # sanity: require src files
    src_paths = {}
    for kind in FILES:
        src = pjoin(prefix, "{}_world_size_{}_rank_{}.pt".format(kind, src_world_size, src_rank))
        ensure_exists(src)
        src_paths[kind] = src

    # create dst files for rank0+rank1
    for dst_rank in range(dst_world_size):
        for kind in FILES:
            dst = pjoin(prefix, "{}_world_size_{}_rank_{}.pt".format(kind, dst_world_size, dst_rank))
            copy_if_missing(src_paths[kind], dst)

    # also write a helper fsdp config (NOT used by loader, but useful for humans)
    fsdp_cfg_path = pjoin(prefix, "fsdp_config.json")
    if os.path.exists(fsdp_cfg_path):
        try:
            with open(fsdp_cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = None
        if isinstance(cfg, dict):
            cfg2 = dict(cfg)
            cfg2["world_size"] = dst_world_size
            out_path = pjoin(prefix, "fsdp_config_world_size_{}.json".format(dst_world_size))
            if not os.path.exists(out_path):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(cfg2, f, ensure_ascii=False, indent=2)
                print(f"[write] {out_path}")
            else:
                print(f"[skip] exists: {out_path}")

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

