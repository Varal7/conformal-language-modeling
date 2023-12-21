from p_tqdm import p_map

import os
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
from rouge_score import rouge_scorer

INPUT_DIR = '/Mounts/rbg-storage1/users/quach/outputs/uncertainty/xl_topp095_temp07'
FILENAME = 'cnn_dailymail_v002-predict_with_aux_with_sent_splits_and_scores_and_nli.jsonl'

parser = argparse.ArgumentParser(description='Compute ROUGE scores for sentences')
parser.add_argument('--input_dir', type=str, default=INPUT_DIR)
parser.add_argument('--filename', type=str, default=FILENAME)
args = parser.parse_args()


scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

with open(os.path.join(args.input_dir, args.filename)) as f:
    # Must be global for p_map
    samples = [json.loads(line) for line in tqdm(f)]

num_samples = len(samples)
num_predictions = len(samples[0]['prediction'])

print(f'num_samples: {num_samples}')
print(f'num_predictions: {num_predictions}')

def compute_rouge_scores(i):
    arr = np.zeros((num_predictions, num_predictions))
    for k1 in range(num_predictions):
        for k2 in range(k1, num_predictions):
            score = scorer.score(samples[i]['prediction'][k1], samples[i]['prediction'][k2])
            arr[k1, k2] = score['rougeL'].fmeasure
            arr[k2, k1] = arr[k1, k2]

    return arr

all_scores = p_map(compute_rouge_scores, range(num_samples))

all_scores = np.array(all_scores)

np.save(os.path.join(args.input_dir, "diversity_rouge_scores.npy"), all_scores)

