"""Script to generate outputs."""

import argparse
import os
import numpy as np

from clm import components
from clm import utils


def main(args):
    utils.set_seed(0)

    # Load dataset.
    train_scores = np.load(args.train_score_file)
    train_rouge = np.load(args.train_rouge_file)
    train_report_labels = np.load(args.train_report_labels_file)
    train_data = utils.ComponentDataset(scores=train_scores, rouge_with_refs=train_rouge, report_labels=train_report_labels)

    test_scores = np.load(args.test_score_file)
    test_rouge = np.load(args.test_rouge_file)
    test_report_labels = np.load(args.test_report_labels_file)
    test_data = utils.ComponentDataset(scores=test_scores, rouge_with_refs=test_rouge, report_labels=test_report_labels)

    epsilons, results = components.run_trials(
        train_data=train_data,
        test_data=test_data,
        p_cal=args.p_cal,
        num_trials=args.num_trials,
        filter_for_answerable=args.filter_for_answerable,
        rouge_threshold=args.rouge_threshold,
        scale_type=args.scale_type,
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.savez(args.output_file, epsilons=epsilons, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_score_file', type=str)
    parser.add_argument('--train_rouge_file', type=str)
    parser.add_argument('--train_report_labels_file', type=str)

    parser.add_argument('--test_score_file', type=str)
    parser.add_argument('--test_rouge_file', type=str)
    parser.add_argument('--test_report_labels_file', type=str)

    parser.add_argument('--output_file', type=str)
    parser.add_argument('--p_cal', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--num_trials', type=int, default=100)

    parser.add_argument('--filter_for_answerable', type=bool, default=False)
    parser.add_argument('--rouge_threshold', type=float, default=0.4)
    parser.add_argument('--scale_type', type=str, default='none')

    args = parser.parse_args()
    main(args)
