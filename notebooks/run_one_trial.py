# %cd ..
# %load_ext autoreload
# %autoreload 2

import itertools
import functools
import os
import numpy as np
from clm import utils, uncertainty
import tqdm
from scipy.stats import binom
import collections
import p_tqdm

method = dict(scaling=('none', {}), scoring='sum', rejection=True)

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

scale_type, scale_kwargs = method.get('scaling', ('none', {}))
scaler = uncertainty.NAME_TO_SCALER[scale_type](**scale_kwargs)
scaler.fit(train_data.scores, train_data.labels)
item_scores = (scaler.predict(test_data.scores))

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
set_score_fn = uncertainty.NAME_TO_SCORE[method.get('scoring', 'none')]
do_rejection = method.get('rejection', False)

# Get pareto frontier for Pareto Testing.

lambda_1 = utils.select_lambdas(opt_similarity_scores, max_lambdas=25)
lambda_2 = utils.select_lambdas(opt_item_scores, max_lambdas=25)
lambda_3 = utils.select_lambdas(set_score_fn(opt_item_scores), max_lambdas=25)

def get_costs_for_lambdas(config):
    values = uncertainty.compute_values(
        config=config,
        item_scores=opt_item_scores,
        similarity_scores=opt_similarity_scores,
        item_labels=opt_item_labels,
        set_score_fn=set_score_fn,
        do_rejection=do_rejection)
    return (values['L_avg'], values['C_obj_avg'])

configs = []
for config in (itertools.product(lambda_1, lambda_2, lambda_3)):
    configs.append(config)

costs = p_tqdm.p_map(get_costs_for_lambdas, configs)

configs = np.array(configs)
costs = np.array(costs)

is_efficient = np.ones(costs.shape[0], dtype=bool)
for i, c in enumerate(costs):
    if is_efficient[i]:
        is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
        is_efficient[i] = True

pareto_configs = configs[is_efficient]
pareto_costs = costs[is_efficient]
sort_idx = np.argsort(pareto_costs[:, 0])
pareto_costs = pareto_costs[sort_idx]
ordered_configs = pareto_configs[sort_idx]

configs = ordered_configs

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


output = dict(
    method=method,
    epsilons=epsilons,
    configs=np.array(best_valid_configs),
    cal_idx=cal_idx,
    opt_idx=opt_idx,
    test_idx=test_idx,
)

np.savez("/tmp/results_fast.npz", **output)
