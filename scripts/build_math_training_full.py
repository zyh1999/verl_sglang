#!/usr/bin/env python3
"""Build deterministic full validation set for RL_math.

- Copies available test sets from discovered data roots
- Normalizes AIME data_source to serl_math_* namespace so reward path is consistent
- Keeps GPQA in boxed-answer style
"""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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
    'aime2024': 'serl_math_aime2024',
    'aime2025': 'serl_math_aime2025',
}


def get_source_roots() -> list[Path]:
    env_root = os.environ.get('DATA_NORMALIZED_ROOT')
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend([
        ROOT / 'data_normalized',
        ROOT / 'data',
    ])
    seen = set()
    roots = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.add(candidate)
            roots.append(candidate)
    return roots


def resolve_sources() -> tuple[dict[str, Path], list[str]]:
    roots = get_source_roots()
    resolved = {}
    missing = []
    for rel in FILES:
        found = None
        for root in roots:
            candidate = root / rel
            if candidate.exists():
                found = candidate
                break
        if found is None:
            missing.append(rel)
        else:
            resolved[rel] = found

    if resolved:
        return resolved, missing

    details = "\n".join(f"- {root}" for root in roots)
    raise FileNotFoundError(
        "Could not find any input parquet files.\n"
        "Looked under these roots:\n"
        f"{details}\n"
        "Set DATA_NORMALIZED_ROOT to a directory containing one or more expected dataset files."
    )


def main() -> None:
    sources, missing = resolve_sources()
    DST.mkdir(parents=True, exist_ok=True)
    total = 0
    for rel in FILES:
        if rel not in sources:
            continue
        src = sources[rel]
        dst = DST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src)
        if 'data_source' in df.columns:
            df['data_source'] = df['data_source'].map(lambda x: DATA_SOURCE_MAP.get(x, x))

        df.to_parquet(dst, index=False)
        total += len(df)
        ds = df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {}
        print(f"{rel}: rows={len(df)} data_source={ds}")

    if missing:
        print("WARNING missing input files:")
        for rel in missing:
            print(f"  - {rel}")

    print(f"DONE full set rows={total}")
    print("Input files:")
    for rel in FILES:
        if rel in sources:
            print(f"  - {rel} <- {sources[rel]}")
    print(f"Output root: {DST}")


if __name__ == '__main__':
    main()
