#!/usr/bin/env python3
import argparse, os
from datasets import Dataset, load_dataset


def to_rows(ds, data_source: str, instruction: str):
    rows = []
    for i, ex in enumerate(ds):
        problem = (ex.get('problem') or ex.get('question') or '').strip()
        ans = ex.get('answer')
        if ans is None:
            ans = ex.get('solution', ex.get('final_answer', ''))
        gt = str(ans).strip()
        rows.append({
            'data_source': data_source,
            'prompt': [{'role': 'user', 'content': (problem + ' ' + instruction).strip()}],
            'ability': 'math',
            'reward_model': {'style': 'rule', 'ground_truth': gt},
            'extra_info': {
                'split': 'test',
                'idx': i,
                'unique_id': ex.get('id') or ex.get('problem_idx') or i,
                'subject': ex.get('domain') or ex.get('problem_type') or 'gpqa',
                'level': None,
                'answer_raw': gt,
                'solution': ex.get('solution'),
                'source_file': data_source,
            },
        })
    return rows


def write_parquet(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Dataset.from_list(rows).to_parquet(out_path)
    print(f'[OK] {out_path} rows={len(rows)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default='data')
    ap.add_argument('--instruction', default='Let us think step by step and output the final answer within \\boxed{}.')
    args = ap.parse_args()

    ds24 = load_dataset('HuggingFaceH4/aime_2024', split='train')
    write_parquet(to_rows(ds24, 'aime2024', args.instruction), os.path.join(args.out_root, 'math_task_aime2024', 'test.parquet'))

    ds25 = load_dataset('MathArena/aime_2025', split='train')
    write_parquet(to_rows(ds25, 'aime2025', args.instruction), os.path.join(args.out_root, 'math_task_aime2025', 'test.parquet'))

    # GPQA main full set (448), multiple-choice boxed answers
    dsg = load_dataset('hendrydong/gpqa_main_mc', split='test')
    write_parquet(to_rows(dsg, 'serl_math_gpqa', args.instruction), os.path.join(args.out_root, 'math_task_gpqa', 'test.parquet'))


if __name__ == '__main__':
    main()
