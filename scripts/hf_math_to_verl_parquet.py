#!/usr/bin/env python3
import argparse, os, re
from typing import Any, Dict, List, Optional
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset

_BOXED_RE = re.compile(r'^\\boxed\{(.*)\}$', re.DOTALL)

def _clean_ground_truth(x: Any) -> str:
    s = '' if x is None else (x if isinstance(x, str) else str(x))
    s = s.strip()
    m = _BOXED_RE.match(s)
    return m.group(1).strip() if m else s

def _last_boxed_substring(string: str) -> Optional[str]:
    idx = string.rfind('\\boxed')
    if '\\boxed ' in string:
        return '\\boxed ' + string.split('\\boxed ')[-1].split('$')[0]
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None
    i, right, n = idx, None, 0
    while i < len(string):
        if string[i] == '{': n += 1
        if string[i] == '}':
            n -= 1
            if n == 0:
                right = i; break
        i += 1
    return None if right is None else string[idx:right+1]

def _remove_boxed_wrapper(s: str) -> str:
    s = s.strip()
    if s.startswith('\\boxed '): return s[len('\\boxed '):].strip()
    if s.startswith('\\boxed{') and s.endswith('}'): return s[len('\\boxed{'):-1].strip()
    return s



def _as_int_or_none(x: Any):
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x) if x.is_integer() else None
    if isinstance(x, str):
        m = re.search(r"(-?\d+)", x)
        return int(m.group(1)) if m else None
    return None


def _pick_problem(ex: Dict[str, Any]) -> str:
    return (ex.get('problem') or ex.get('question') or ex.get('prompt') or '').strip()

def _pick_answer_like(ex: Dict[str, Any]) -> str:
    for k in ('answer','ground_truth','final_answer','solution'):
        if ex.get(k): return str(ex[k])
    return ''

def _extract_ground_truth(answer_like: str) -> str:
    gt = _clean_ground_truth(answer_like)
    boxed = _last_boxed_substring(answer_like)
    return _remove_boxed_wrapper(boxed) if boxed else gt

def _to_rows(ds: Dataset, split: str, data_source: str, instruction: str, ability: str, source_name: str) -> List[Dict[str, Any]]:
    rows = []
    for idx, ex in enumerate(ds):
        problem = _pick_problem(ex)
        ans_like = _pick_answer_like(ex)
        gt = _extract_ground_truth(ans_like)
        rows.append({
            'data_source': data_source,
            'prompt': [{'role':'user','content': (problem + ' ' + instruction).strip()}],
            'ability': ability,
            'reward_model': {'style':'rule','ground_truth': gt},
            'extra_info': {
                'split': split, 'idx': idx,
                'unique_id': ex.get('unique_id') or ex.get('id'),
                'subject': ex.get('subject') or ex.get('type'),
                'level': _as_int_or_none(ex.get('level')),
                'answer_raw': ans_like,
                'solution': ex.get('solution'),
                'source_file': source_name,
            },
        })
    return rows

def _load_hendrycks_train(train_dataset: str) -> Dataset:
    cfgs = get_dataset_config_names(train_dataset)
    return concatenate_datasets([load_dataset(train_dataset, cfg, split='train') for cfg in cfgs])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dataset', default='EleutherAI/hendrycks_math')
    ap.add_argument('--math500-dataset', default='HuggingFaceH4/MATH-500')
    ap.add_argument('--mathhard-dataset', default='lighteval/MATH-Hard')
    ap.add_argument('--mathhard-split', default='test')
    ap.add_argument('--out-math-dir', default='/mnt/home/verl_v0.4.x/data/math_task')
    ap.add_argument('--out-hard-dir', default='/mnt/home/verl_v0.4.x/data/math_task_hard')
    ap.add_argument('--instruction', default='Let us think step by step and output the final answer within \\boxed{}.')
    ap.add_argument('--ability', default='math')
    ap.add_argument('--train-data-source', default='serl_math_train_full')
    ap.add_argument('--test500-data-source', default='serl_math_500')
    ap.add_argument('--testhard-data-source', default='serl_math_hard')
    args = ap.parse_args()

    train_ds = _load_hendrycks_train(args.train_dataset)
    test500_ds = load_dataset(args.math500_dataset, split='test')
    testhard_ds = load_dataset(args.mathhard_dataset, split=args.mathhard_split)

    train_rows = _to_rows(train_ds, 'train', args.train_data_source, args.instruction, args.ability, args.train_dataset)
    test500_rows = _to_rows(test500_ds, 'test', args.test500_data_source, args.instruction, args.ability, args.math500_dataset)
    testhard_rows = _to_rows(testhard_ds, 'test', args.testhard_data_source, args.instruction, args.ability, args.mathhard_dataset)

    os.makedirs(args.out_math_dir, exist_ok=True)
    os.makedirs(args.out_hard_dir, exist_ok=True)
    out_train = os.path.join(args.out_math_dir, 'train.parquet')
    out_test500 = os.path.join(args.out_math_dir, 'test.parquet')
    out_testhard = os.path.join(args.out_hard_dir, 'test.parquet')
    Dataset.from_list(train_rows).to_parquet(out_train)
    Dataset.from_list(test500_rows).to_parquet(out_test500)
    Dataset.from_list(testhard_rows).to_parquet(out_testhard)
    print(f'[OK] train  -> {out_train} rows={len(train_rows)}')
    print(f'[OK] math500-> {out_test500} rows={len(test500_rows)}')
    print(f'[OK] hard   -> {out_testhard} rows={len(testhard_rows)}')

if __name__ == '__main__':
    main()
