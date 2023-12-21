# %cd ..
# %load_ext autoreload
# %autoreload 2

import functools
import os
import numpy as np
from clm import utils, uncertainty
import tqdm
from scipy.stats import binom
import collections


methods = [
    dict(scaling=('none', {}), scoring='first_k', rejection=False),
    dict(scaling=('none', {}), scoring='max', rejection=False),
    dict(scaling=('none', {}), scoring='max', rejection=True),
    dict(scaling=('none', {}), scoring='sum', rejection=True),
    dict(scaling=('platt', {}), scoring='geo', rejection=False),
    dict(scaling=('platt', {}), scoring='geo', rejection=True),
]


risk = 0.7

base = "data/cxr"
splits = np.load(os.path.join(base, "splits.npz"))

train_scores = np.load(os.path.join(base, "train", "probs.npy"))
train_labels = np.load(os.path.join(base, "train", "labels.npy"))
train_similarity = np.load(os.path.join(base, "train", "diversity_rouge.npy"))
train_data = utils.Dataset(train_scores, train_similarity, train_labels)

test_scores = np.load(os.path.join(base, "test", "probs.npy"))
test_labels = np.load(os.path.join(base, "test", "labels.npy"))
test_similarity = np.load(os.path.join(base, "test", "diversity_rouge.npy"))
test_data = utils.Dataset(test_scores, test_similarity, test_labels)

all_item_scores = []
for method in tqdm.tqdm(methods, desc='scaling'):
    scale_type, scale_kwargs = method.get('scaling', ('none', {}))
    scaler = uncertainty.NAME_TO_SCALER[scale_type](**scale_kwargs)
    scaler.fit(train_data.scores, train_data.labels)
    all_item_scores.append(scaler.predict(test_data.scores))

delta=0.05,
epsilons = uncertainty.DEFAULT_EPSILONS
p_cal = 0.3

N = len(test_data.scores)
N_cal = int(p_cal * N)
N_pcal = int(0.7 * N_cal)
splits = []
randperm = np.random.permutation(N)
pareto_idx, test_idx = randperm[:N_cal], randperm[N_cal:]
cal_idx, opt_idx = pareto_idx[:N_pcal], pareto_idx[N_pcal:]


# Compute results for different methods.
all_trial_results = []
for i, item_scores in enumerate(all_item_scores):
    opt_item_scores = item_scores[opt_idx]
    opt_similarity_scores = test_data.similarity[opt_idx]
    opt_item_labels = test_data.labels[opt_idx]

    cal_item_scores = item_scores[cal_idx]
    cal_similarity_scores = test_data.similarity[cal_idx]
    cal_item_labels = test_data.labels[cal_idx]

    test_item_scores = item_scores[test_idx]
    test_similarity_scores = test_data.similarity[test_idx]
    test_item_labels = test_data.labels[test_idx]

    # Get scoring mechanism for this method.
    set_score_fn = uncertainty.NAME_TO_SCORE[methods[i].get('scoring', 'none')]
    do_rejection = methods[i].get('rejection', False)

    # Get pareto frontier for Pareto Testing.
    configs = uncertainty.get_pareto_frontier(
        item_scores=opt_item_scores,
        similarity_scores=opt_similarity_scores,
        item_labels=opt_item_labels,
        set_score_fn=set_score_fn,
        do_rejection=do_rejection)

    # Choose best valid configs (configs are already ordered).
    best_valid_configs = [[np.nan] * 3] * len(epsilons)
    is_stopped = [False] * len(epsilons)
    for config in configs:
        values = uncertainty.compute_values(
            config=config,
            item_scores=cal_item_scores,
            similarity_scores=cal_similarity_scores,
            item_labels=cal_item_labels,
            set_score_fn=set_score_fn,
            do_rejection=do_rejection)

        for j, epsilon in enumerate(epsilons):
            loss = values['L_avg']
            n = len(cal_idx)
            p_value = binom.cdf(n * loss, n, epsilon)
            if p_value <= delta and not is_stopped[j]:
                best_valid_configs[j] = config
            else:
                is_stopped[j] = True

    # Compute test metrics.
    trial_results = collections.defaultdict(list)
    trial_results['configs'] = np.array(best_valid_configs)
    for j, config in enumerate(best_valid_configs):
        values = uncertainty.compute_values(
            config=config,
            item_scores=test_item_scores,
            similarity_scores=test_similarity_scores,
            item_labels=test_item_labels,
            set_score_fn=set_score_fn,
            do_rejection=do_rejection)
        for k, v in values.items():
            trial_results[k].append(v)
    all_trial_results.append(trial_results)


len(all_trial_results)


