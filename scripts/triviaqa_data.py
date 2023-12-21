"""Split TriviaQA data."""

import argparse
import os
import numpy as np

INPUT_DIR = '/Mounts/rbg-storage1/users/quach/outputs/uncertainty/triviaqa'
OUTPUT_DIR = 'data/triviaqa'


def main(args):
    np.random.seed(0)
    all_labels = 1 - np.load(os.path.join(args.input_dir, 'all_losses.npy'))
    all_probs = np.load(os.path.join(args.input_dir, 'all_prob_scores.npy'))
    all_self_eval = np.load(os.path.join(args.input_dir, 'all_self_eval.npy'))
    diversity = np.load(os.path.join(args.input_dir, 'diversity.npy'))

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
        dirname = os.path.join(args.output_dir, split)
        os.makedirs(dirname, exist_ok=True)
        np.save(os.path.join(dirname, 'idx.npy'), idx)
        np.save(os.path.join(dirname, 'labels.npy'), all_labels[idx])
        np.save(os.path.join(dirname, 'probs.npy'), all_probs[idx])
        np.save(os.path.join(dirname, 'self_eval.npy'), all_self_eval[idx])
        np.save(os.path.join(dirname, 'diversity.npy'), diversity[idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--num_train', type=int, default=2000)
    parser.add_argument('--num_val', type=int, default=2000)
    args = parser.parse_args()
    main(args)
