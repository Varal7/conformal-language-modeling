# %cd ..
# %load_ext autoreload
# %autoreload 2

import torch
import os
import numpy as np
import json
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from clm.components import (
    get_rouge_with_refs,
    get_preds,
    get_random_preds,
    get_first_k_preds,
    compute_values_for_C_inner,
)

base = "data/cnndm/val"

with open(os.path.join(base, "row_rouge.jsonl")) as f:
    row_rouge_scores = [json.loads(line) for line in tqdm(f)]
with open(os.path.join(base, "row_generation.jsonl")) as f:
    row_generation_idx_to_row_idx = [json.loads(line) for line in tqdm(f)]
with open(os.path.join(base, "row_reference.jsonl")) as f:
    row_reference_idx_to_row_idx = [json.loads(line) for line in tqdm(f)]

# +
with open(os.path.join(base, "probs_scores.jsonl")) as f:
    prob_scores = [json.loads(line) for line in tqdm(f)]

with open(os.path.join(base, "nli_scores.jsonl")) as f:
    nli_scores = [json.loads(line) for line in tqdm(f)]

# +
rouge_with_refs = get_rouge_with_refs(row_rouge_scores, row_reference_idx_to_row_idx)
_, _, K = rouge_with_refs.shape

nli_preds = get_preds(nli_scores, row_generation_idx_to_row_idx, K)
probs_preds = get_preds(prob_scores, row_generation_idx_to_row_idx, K)
random_preds = get_random_preds(row_generation_idx_to_row_idx, K)
firstk_preds = get_first_k_preds(row_generation_idx_to_row_idx, K)

# +
kwargs = {
    "rouge_threshold": 0.2,
    "num_quantiles": 1000,
}

results = {
    'NLI Scores':  compute_values_for_C_inner(nli_preds, rouge_with_refs, **kwargs),
    'Prob Scores':  compute_values_for_C_inner(probs_preds, rouge_with_refs, **kwargs),
    'Random':  compute_values_for_C_inner(random_preds, rouge_with_refs, **kwargs),
    'First-K':  compute_values_for_C_inner(firstk_preds, rouge_with_refs, **kwargs),
}


# -

def plot_values(results, x_axis, y_axis, ax=None, **kwargs):
    """Plot y_axis vs. x_axis given a dict of {method: [(size, loss), ...]}."""
    if ax is None:
        _, ax = plt.subplots(1, 1)

    for method, values in results.items():
        x_values = values[x_axis]
        y_values = values[y_axis]
        ax.plot(x_values, y_values, label=method, **kwargs)

    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)
    ax.legend()
    return ax


plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(1, 1)
plot_values(results, 'L_avg', 'C_size_avg', ax=ax, marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}|]$")
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 5)
ax.set_title(f"C_inner, with rougeL_threshold = {kwargs['rouge_threshold']}")
plt.show()

plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(1, 1)
plot_values(results, 'L_avg', 'C_relative_size', ax=ax, marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}| / |C^*|]$")
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 4)
ax.set_title(f"C_inner (relative size), with rougeL_threshold = {kwargs['rouge_threshold']}")
plt.show()

plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(1, 1)
plot_values(results, 'L_avg', 'recall', ax=ax, marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}\cap C^*| / |C^*|]$")
ax.set_xlim(0, 0.9)
ax.set_ylim(0, 1)
ax.set_title(f"C_inner (recall), with rougeL_threshold = {kwargs['rouge_threshold']}")
plt.show()

plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(1, 1)
plot_values(results, 'L_avg', 'micro_recall', ax=ax, marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}\cap C^*|] / \mathbb{E}[|C^*|]$")
ax.set_xlim(0, 0.9)
ax.set_ylim(0, 1)
ax.set_title(f"C_inner (micro recall), with rougeL_threshold = {kwargs['rouge_threshold']}")
plt.show()

plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(1, 1)
plot_values(results, 'L_avg', 'fake_recall', ax=ax, marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}\cap C^*| / |C^*|]$")
ax.set_xlim(0, 0.9)
ax.set_ylim(0, 2)
ax.set_title(f"C_inner (fake recall), with rougeL_threshold = {kwargs['rouge_threshold']}")
plt.show()

plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(1, 1)
plot_values(results, 'L_avg', 'C_size_conditioned', ax=ax, marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}| | ok]$")
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 8)
ax.set_title(f"C_inner size conditioned, with rougeL_threshold = {kwargs['rouge_threshold']}")
plt.show()

plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(1, 1)
plot_values(results, 'L_avg', 'micro_recall_conditioned', ax=ax, marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}\cap C^*| | ok] / \mathbb{E}[|C^*| | ok ]$")
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 1)
ax.set_title(f"C_inner, with rougeL_threshold = {kwargs['rouge_threshold']}")
plt.show()




