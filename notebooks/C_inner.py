# %cd ..
# %load_ext autoreload
# %autoreload 2

import os
import numpy as np
import matplotlib.pyplot as plt

from clm.components import (
    compute_values_for_C_inner,
)


base = "data/cxr/val"

# +
rouge_label = np.load(os.path.join(base, "components", "rouge_label.npy"))
rouge_with_refs = np.load(os.path.join(base, "components", "rouge_with_refs.npy"))
oracle_size = np.load(os.path.join(base, "components", "oracle_size.npy"))

image_preds = np.load(os.path.join(base, "components", "image_preds.npy"))
probs_preds = np.load(os.path.join(base, "components", "probs_preds.npy"))
random_preds = np.load(os.path.join(base, "components", "random_preds.npy"))
firstk_preds = np.load(os.path.join(base, "components", "firstk_preds.npy"))

mask = np.load(os.path.join(base, "components", "mask.npy"))
oracle_size = np.load(os.path.join(base, "components", "oracle_size.npy"))

labels = np.load(os.path.join(base, "labels.npy"))
oracle_subset = labels.any(axis=1)

# +
kwargs = {
    "rouge_threshold": 0.4,
    "num_quantiles": 1000,
    "subset": oracle_subset,
}

results = {
    'Image-sentence Scores':  compute_values_for_C_inner(image_preds, rouge_with_refs, **kwargs),
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
ax.set_ylim(0, 20)
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
plot_values(results, 'L_avg', 'micro_recall_conditioned', ax=ax, marker=".")
ax.set_xlabel("$1 - \mathbb{P}(C_{inner} \subset C^*)$")
ax.set_ylabel("$\mathbb{E}[|C_{inner}\cap C^*| | ok] / \mathbb{E}[|C^*| | ok ]$")
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 1)
ax.set_title(f"C_inner, with rougeL_threshold = {kwargs['rouge_threshold']}")
plt.show()



