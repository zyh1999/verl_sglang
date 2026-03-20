#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
#ROOT="/scratch/h99859yz/verl/data"
cd "$ROOT"

echo "[1/6] Build base math train + math500 + math_hard from HF"
python scripts/hf_math_to_verl_parquet.py \
  --out-math-dir data/math_task \
  --out-hard-dir data/math_task_hard \
  --train-data-source math_train_full \
  --test500-data-source math_500 \
  --testhard-data-source math_hard

echo "[2/6] Build AIME2024/AIME2025/GPQA from HF"
python scripts/hf_aime_to_verl_parquet.py --out-root data

echo "[3/6] Build Hendrycks in-domain TEST (full 5000)"
python - <<'PY2'
from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path

ROOT = Path('.')
out = ROOT / 'data' / 'hendrycks_math_in_domian' / 'test.parquet'
out.parent.mkdir(parents=True, exist_ok=True)

configs = [
    'algebra',
    'counting_and_probability',
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus',
]

parts = [load_dataset('EleutherAI/hendrycks_math', cfg, split='test') for cfg in configs]
ds = concatenate_datasets(parts)

instruction = 'Let us think step by step and output the final answer within \\boxed{}.'
rows = []
for i, ex in enumerate(ds):
    q = (ex.get('problem') or ex.get('question') or '').strip()
    ans = ex.get('answer') or ex.get('solution') or ''
    rows.append({
        'data_source': 'math_in_domain',
        'prompt': [{'role': 'user', 'content': (q + ' ' + instruction).strip()}],
        'ability': 'math',
        'reward_model': {'style': 'rule', 'ground_truth': str(ans).strip()},
        'extra_info': {
            'split': 'test',
            'idx': i,
            'unique_id': ex.get('unique_id') or i,
            'subject': ex.get('subject'),
            'level': ex.get('level') if isinstance(ex.get('level'), int) else None,
            'answer_raw': str(ans),
            'solution': ex.get('solution'),
            'source_file': 'EleutherAI/hendrycks_math:test',
            'hendrycks_config': ex.get('type'),
        },
    })

Dataset.from_list(rows).to_parquet(str(out))
print(f'[OK] wrote {out} rows={len(rows)} (expect ~5000)')
PY2

echo "[4/6] Normalize eval data_source names"
python - <<'PY2'
from pathlib import Path
import pandas as pd

ROOT = Path('.')
files = [
    ROOT / 'data' / 'math_task' / 'test.parquet',
    ROOT / 'data' / 'math_task_hard' / 'test.parquet',
    ROOT / 'data' / 'math_task_aime2024' / 'test.parquet',
    ROOT / 'data' / 'math_task_aime2025' / 'test.parquet',
    ROOT / 'data' / 'math_task_gpqa' / 'test.parquet',
    ROOT / 'data' / 'hendrycks_math_in_domian' / 'test.parquet',
]
map_ds = {
    'aime2024': 'math_aime2024',
    'aime2025': 'math_aime2025',
    'math_aime2024': 'math_aime2024',
    'math_aime2025': 'math_aime2025',
    'math_gpqa': 'math_gpqa',
    'math_500': 'math_500',
    'math_hard': 'math_hard',
    'math_in_domain': 'math_in_domain',
}
for p in files:
    df = pd.read_parquet(p)
    if 'data_source' in df.columns:
        df['data_source'] = df['data_source'].map(lambda x: map_ds.get(x, x))
    df.to_parquet(p, index=False)
    print(f'[OK] normalized {p}')
PY2

echo "[5/6] Assemble full eval bundle -> data/math_training_full"
python - <<'PY2'
from pathlib import Path
import pandas as pd

ROOT = Path('.')
DST = ROOT / 'data' / 'math_training_full'
DST.mkdir(parents=True, exist_ok=True)
rels = [
    'math_task/test.parquet',
    'math_task_hard/test.parquet',
    'math_task_aime2024/test.parquet',
    'math_task_aime2025/test.parquet',
    'math_task_gpqa/test.parquet',
    'hendrycks_math_in_domian/test.parquet',
]
for rel in rels:
    src = ROOT / 'data' / rel
    dst = DST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)
    df.to_parquet(dst, index=False)
    print(f'[OK] copied {rel}: rows={len(df)}')
PY2

echo "[6/6] Sanity check row counts"
python - <<'PY2'
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path('.')
checks = {
    'data/math_task/train.parquet': None,
    'data/math_task/test.parquet': 500,
    'data/math_task_hard/test.parquet': 1324,
    'data/math_task_aime2024/test.parquet': 30,
    'data/math_task_aime2025/test.parquet': 30,
    'data/math_task_gpqa/test.parquet': 448,
    'data/hendrycks_math_in_domian/test.parquet': 5000,
    'data/math_training_full/math_task/test.parquet': 500,
    'data/math_training_full/math_task_hard/test.parquet': 1324,
}
for rel, expect in checks.items():
    p = ROOT / rel
    n = pq.ParquetFile(p).metadata.num_rows
    if expect is not None and n != expect:
        raise SystemExit(f'[FAIL] {rel}: rows={n}, expect={expect}')
    print(f'[OK] {rel}: rows={n}' + ('' if expect is None else f' (expect={expect})'))

print('[DONE] Full dataset generation completed (no sampling).')
PY2
