#!/usr/bin/env python3
"""Build deterministic full validation set for RL_math.

- Copies full test sets from data_normalized
- Normalizes data_source to math_* namespace (no serl_* labels)
"""
from pathlib import Path
import pandas as pd

ROOT = Path('/mnt/iusers01/fatpou01/compsci01/h99859yz/verl_new')
SRC = ROOT / 'data_normalized'
DST = ROOT / 'data' / 'math_training_full'

FILES = [
    'math_task/test.parquet',
    'math_task_hard/test.parquet',
    'math_task_aime2024/test.parquet',
    'math_task_aime2025/test.parquet',
    'math_task_gpqa/test.parquet',
    'hendrycks_math_in_domian/test.parquet',
]

DATA_SOURCE_MAP = {
    'aime2024': 'math_aime2024',
    'aime2025': 'math_aime2025',
    'serl_math_500': 'math_500',
    'serl_math_hard': 'math_hard',
    'serl_math_in_domain': 'math_in_domain',
    'serl_math_gpqa': 'math_gpqa',
    'serl_math_aime2024': 'math_aime2024',
    'serl_math_aime2025': 'math_aime2025',
}


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    total = 0
    for rel in FILES:
        src = SRC / rel
        dst = DST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src)
        if 'data_source' in df.columns:
            df['data_source'] = df['data_source'].map(lambda x: DATA_SOURCE_MAP.get(x, x))

        df.to_parquet(dst, index=False)
        total += len(df)
        ds = df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {}
        print(f"{rel}: rows={len(df)} data_source={ds}")

    print(f"DONE full set rows={total}")
    print(f"Output root: {DST}")


if __name__ == '__main__':
    main()
