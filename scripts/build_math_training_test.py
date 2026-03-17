#!/usr/bin/env python3
"""Build deterministic subset validation set from math_training_full.

No randomness: keep first N rows after preserving input order.
"""
from pathlib import Path
import pandas as pd

ROOT = Path('/mnt/iusers01/fatpou01/compsci01/h99859yz/verl_new')
SRC = ROOT / 'data' / 'math_training_full'
DST = ROOT / 'data' / 'math_training_test'

PLAN = {
    'math_task/test.parquet': 100,
    'math_task_hard/test.parquet': 200,
    'math_task_aime2024/test.parquet': None,
    'math_task_aime2025/test.parquet': None,
    'math_task_gpqa/test.parquet': 100,
    'hendrycks_math_in_domian/test.parquet': 200,
}


def subset_df(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.iloc[:n].reset_index(drop=True)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    total = 0
    for rel, n in PLAN.items():
        src = SRC / rel
        dst = DST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src)
        out = subset_df(df, n)
        out.to_parquet(dst, index=False)

        ds = out['data_source'].value_counts().to_dict() if 'data_source' in out.columns else {}
        print(f"{rel}: {len(df)} -> {len(out)} (target={n}) data_source={ds}")
        total += len(out)

    print(f"DONE subset rows={total}")
    print(f"Output root: {DST}")


if __name__ == '__main__':
    main()
