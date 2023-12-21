"""Split CXR data."""

import argparse
import os
import numpy as np
import json

INPUT_DIR = '/Mounts/rbg-storage1/users/quach/outputs/uncertainty/cxr2'
OUTPUT_DIR = 'data/cxr'
PREFIXES = [
    'calibration_2000_5000',
    'valid_5000_8000',
    'test_8000_13000',
    'valid_scorer_13000_14000',
]


def load_all(dirname, suffix):
    combined = []
    for prefix in PREFIXES:
        filename = f'{prefix}_{suffix}.npy'
        combined.append(np.load(os.path.join(dirname, filename)))
    return np.concatenate(combined, axis=0)

def load_diversity(dirname, name):
    combined = []
    for prefix in PREFIXES:
        filename = f'diversity/{prefix}/{name}.npy'
        combined.append(np.load(os.path.join(dirname, filename)))
    return np.concatenate(combined, axis=0)

def load_rouge(dirname, name):
    combined = []
    for prefix in PREFIXES:
        filename = f'components/{prefix}/rouge_scores_42_62/{name}.jsonl'
        with open(os.path.join(dirname, filename)) as f:
            combined.extend([json.loads(line) for line in f])
    return combined

def load_component_scores(dirname, name):
    combined = []
    for prefix in PREFIXES:
        filename = f'components/{prefix}/{name}.jsonl'
        with open(os.path.join(dirname, filename)) as f:
            combined.extend([json.loads(line) for line in f])
    return combined

def save_jsonl(dirname, name, data, indices):
    with open(os.path.join(dirname, name), 'w') as f:
        for i in indices:
            f.write(json.dumps(data[i]) + '\n')


def main(args):
    np.random.seed(0)
    print("Loading data...")
    all_labels = load_all(args.input_dir, 'soft') <= args.loss_threshold
    all_probs = load_all(args.input_dir, 'normprob_scores')
    all_image = load_all(args.input_dir, 'image-report_scores')
    all_text = load_all(args.input_dir, 'text_scores')
    all_gnn = load_all(args.input_dir, 'gnn_scores')

    all_diversity_chexbert = load_diversity(args.input_dir, "chexbert_eq")
    all_diversity_rouge = load_diversity(args.input_dir, "rouge_scores")

    all_image_component_scores = load_component_scores(args.input_dir, "image_sentence_scores")
    all_normprob_component_scores = load_component_scores(args.input_dir, "normprob_sentence_scores")

    print("Reading rouge scores...")
    all_row_rouge = load_rouge(args.input_dir, "row_rouge_scores")
    all_row_generation = load_rouge(args.input_dir, "row_generation_idx_to_row_idx")
    all_row_reference = load_rouge(args.input_dir, "row_reference_idx_to_row_idx")

    if args.shuffle_file is not None:
        splits = np.load(args.shuffle_file)

    else:
        shuffle = np.random.permutation(len(all_labels))
        splits = {
            'train': shuffle[:args.num_train],
            'val': shuffle[args.num_train:args.num_train + args.num_val],
            'test': shuffle[args.num_train + args.num_val:],
        }

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, 'splits.npz'),
             train=splits['train'], val=splits['val'], test=splits['test'])
    for split, idx in splits.items():
        print(f"Saving {split}")
        dirname = os.path.join(args.output_dir, split)
        os.makedirs(dirname, exist_ok=True)
        np.save(os.path.join(dirname, 'idx.npy'), idx)
        np.save(os.path.join(dirname, 'labels.npy'), all_labels[idx])
        np.save(os.path.join(dirname, 'probs.npy'), all_probs[idx])
        np.save(os.path.join(dirname, 'image_report.npy'), all_image[idx])
        np.save(os.path.join(dirname, 'report.npy'), all_text[idx])
        np.save(os.path.join(dirname, 'gnn.npy'), all_gnn[idx])
        np.save(os.path.join(dirname, 'diversity_chexbert.npy'), all_diversity_chexbert[idx])
        np.save(os.path.join(dirname, 'diversity_rouge.npy'), all_diversity_rouge[idx])
        save_jsonl(dirname, 'row_rouge.jsonl', all_row_rouge, idx)
        save_jsonl(dirname, 'row_generation.jsonl', all_row_generation, idx)
        save_jsonl(dirname, 'row_reference.jsonl', all_row_reference, idx)
        save_jsonl(dirname, 'image_component_scores.jsonl', all_image_component_scores, idx)
        save_jsonl(dirname, 'normprob_component_scores.jsonl', all_normprob_component_scores, idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--shuffle_file', type=str, default=None)
    parser.add_argument('--loss_threshold', type=float, default=0.0)
    parser.add_argument('--num_train', type=int, default=2000)
    parser.add_argument('--num_val', type=int, default=2000)
    args = parser.parse_args()
    main(args)
