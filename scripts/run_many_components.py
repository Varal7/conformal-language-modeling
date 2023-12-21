from collections import defaultdict
import os
import multiprocessing
import subprocess
import numpy as np
import datetime
import argparse

NUM_TRIALS = 100
TEMP_DIR = '/tmp/clm/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_filename(task, score, split, filter_for_answerable, rouge_threshold, idx_trial):
    return os.path.join(TEMP_DIR, f'{task}_{score}_{split}_{filter_for_answerable}_{rouge_threshold}_{idx_trial}.npz')

def run_component_trials(task, score, split, filter_for_answerable, rouge_threshold, idx_trial, batch_size):
    filename = get_filename(task, score, split, filter_for_answerable, rouge_threshold, idx_trial)

    command = [
        'python', 'scripts/run_component_trials.py',
        '--train_score_file', f'data/{task}/train/components/{score}_preds.npy',
        '--train_rouge_file', f'data/{task}/train/components/rouge_with_refs.npy',
        '--train_report_labels_file', f'data/{task}/train/labels.npy',
        '--test_score_file', f'data/{task}/{split}/components/{score}_preds.npy',
        '--test_rouge_file', f'data/{task}/{split}/components/rouge_with_refs.npy',
        '--test_report_labels_file', f'data/{task}/{split}/labels.npy',
        '--rouge_threshold', str(rouge_threshold),
        '--num_trials', str(batch_size),
        '--filter_for_answerable', str(filter_for_answerable),
        '--output_file', filename,
    ]
    subprocess.run(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cxr')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--scores', nargs='+', default=['image', 'probs', 'random', 'firstk'])
    parser.add_argument('--filter_for_answerable', nargs='+', type=int, default=[False, True])
    parser.add_argument('--rouge_threshold', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_trials', type=int, default=NUM_TRIALS)
    parser.add_argument('--results_base', type=str, default="results")
    args = parser.parse_args()

    num_runs = args.num_trials // args.batch_size

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for score in args.scores:
        for filter_flag in args.filter_for_answerable:
            for idx_trial in range(num_runs):
                pool.apply_async(run_component_trials, args=(args.task, score, args.split, filter_flag, args.rouge_threshold, idx_trial, args.batch_size))

    pool.close()
    pool.join()

    for score in args.scores:
        for filter_flag in args.filter_for_answerable:
            result_dir = f'{args.results_base}/{args.task}/{args.split}/filter_{filter_flag}/rouge_{args.rouge_threshold}/'
            os.makedirs(result_dir, exist_ok=True)

            combined = defaultdict(list)
            stacked_results = {}
            epsilons = None

            for idx_trial in range(num_runs):
                filename = get_filename(args.task, score, args.split, filter_flag, args.rouge_threshold, idx_trial)
                output = np.load(filename, allow_pickle=True)
                results = output['results'].item()
                epsilons = output['epsilons']

                for k, v in results.items():
                    combined[k].append(v)

            for k, v in combined.items():
                stacked_results[k] = np.concatenate(v, axis=0)

            np.savez(f'{result_dir}/{score}_components.npz', epsilons=epsilons, results=stacked_results)
