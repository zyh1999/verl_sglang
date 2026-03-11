import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except Exception:
        b = boxed.strip()
        bs = '\\'
        pref = bs + 'boxed'
        if b.startswith(pref + '{') and b.endswith('}'):
            return b[len(pref) + 1 : -1].strip()
        if b.startswith(pref):
            return b[len(pref):].strip(' {}$')
        return b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--local_dataset_path', default=None)
    parser.add_argument('--dataset_name', default='AI-MO/NuminaMath-CoT')
    parser.add_argument('--train_sample_size', type=int, default=35000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_save_dir', default='~/data/numinamath_cot_35k')
    args = parser.parse_args()

    data_source = args.dataset_name
    print(f'Loading dataset: {data_source}', flush=True)
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    test_dataset = dataset['test'] if 'test' in dataset else None

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    sample_size = min(args.train_sample_size, len(train_dataset))
    train_dataset = train_dataset.shuffle(seed=args.seed).select(range(sample_size))
    print(f'Sampled train examples: {len(train_dataset)}', flush=True)

    def make_map_fn(split):
        def process_fn(example, idx):
            q = example.get('problem', '') + ' ' + instruction
            ans = example.get('solution', '')
            gt = extract_solution(ans)
            return {
                'data_source': data_source,
                'prompt': [{'role': 'user', 'content': q}],
                'ability': 'math',
                'reward_model': {'style': 'rule', 'ground_truth': gt},
                'extra_info': {'split': split, 'index': idx, 'source': example.get('source', None)},
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    if test_dataset is not None:
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    save_dir = args.local_dir if args.local_dir is not None else args.local_save_dir
    local_dir = os.path.expanduser(save_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    with open(os.path.join(local_dir, 'train_example.json'), 'w') as f:
        json.dump(train_dataset[0], f, indent=2)

    if test_dataset is not None:
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
        with open(os.path.join(local_dir, 'test_example.json'), 'w') as f:
            json.dump(test_dataset[0], f, indent=2)

    with open(os.path.join(local_dir, 'meta.json'), 'w') as f:
        json.dump(
            {
                'dataset': data_source,
                'seed': args.seed,
                'train_sample_size': len(train_dataset),
                'test_size': len(test_dataset) if test_dataset is not None else 0,
            },
            f,
            indent=2,
        )

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print(f'Done. Saved to: {local_dir}', flush=True)
