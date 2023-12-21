"""Utilities for experiments."""

import collections
import random
import torch
import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.15, rc={'figure.figsize': (5, 4)})

Dataset = collections.namedtuple('Dataset', ['scores', 'similarity', 'labels'])
ComponentDataset = collections.namedtuple('ComponentDataset', ['scores', 'rouge_with_refs', 'report_labels'])

# Throw out alpha if more than this percent of trials are
# either uncalibratable or trivial (first example satisfies).
MAX_INVIABLE_THRESHOLD = 0.75

COLORS = [
    u"#1f77b4",
    u"#2ca02c",
    u"#ff7f0e",
    u"#9467bd",
    u"#8c564b",
    u"#e377c2",
    u"#7f7f7f",
    u"#bcbd22",
    u"#17becf",
]


def set_seed(seed=0):
    """Set random seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def assert_monotone(array, axis=-1):
    """Check if array is monotone increasing along axis."""
    assert(np.all(np.diff(array, axis=-1) <= 0))


def select_lambdas(values, max_lambdas=1000):
    """Select unique quantiles of the empirical distribution."""
    quantiles = np.linspace(0, 1, max_lambdas)
    lambdas = np.unique(np.quantile(values, quantiles, method='nearest'))
    lambdas = np.concatenate([[-np.inf], lambdas, [np.inf]])
    return lambdas


def set_losses_from_labels(set_labels):
    """Given individual labels, compute set loss."""
    return np.cumprod(1 - set_labels, axis=-1)


def compute_auc(xs, ys):
    """Compute area under the curve, restricted to support."""
    area = np.trapz(ys, xs) / (max(xs) - min(xs) + 0.02)
    return area


def plot_results(
    results,
    epsilons,
    y_axis,
    y_axis_name=None,
    title=None,
    show_calibrated_only=True,
    add_diagonal=False,
    ylim_is_xlim=False,
    colors=None,
    ax=None,
    **kwargs,
):
    """Plot y_axis vs. alphas given a dict of {method: {y_axis: values}}."""
    if ax is None:
        _, ax = plt.subplots(1, 1)

    mask = np.zeros_like(epsilons)
    for values in results.values():
        mask += np.mean(np.all(np.isnan(values['configs']), -1), 0) > 0
        mask += np.maximum.accumulate(np.mean(np.all(np.isinf(values['configs']), -1), 0)) > 0.05

    mask = mask == 0
    if show_calibrated_only:
        epsilons = epsilons[mask]

    for i, (name, values) in enumerate(results.items()):
        values = values[y_axis]
        if show_calibrated_only:
            values = values[:, mask]

        middle = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        lower = middle - std
        upper = middle + std

        auc = compute_auc(epsilons, middle)
        label = f'{name} ({auc:.2f})'
        color = colors[name] if colors else COLORS[i]
        ax.plot(epsilons, middle, label=label, color=color, **kwargs)
        ax.fill_between(epsilons, lower, upper, color=color, alpha=0.2)

    if add_diagonal:
        diag_auc = compute_auc(epsilons, epsilons)
        ax.plot(epsilons, epsilons, '--', label=f'diagonal ({diag_auc:.2f})', color='k')

    ax.legend()
    ax.set_xlim(min(epsilons) - 0.01, max(epsilons) + 0.01)
    if ylim_is_xlim:
        ax.set_ylim(min(epsilons) - 0.01, max(epsilons) + 0.01)

    ax.set_xlabel(r'Risk Level ($\epsilon$)')
    ax.set_ylabel(y_axis_name or y_axis)
    if title:
        ax.set_title(title)

    return ax


def print_methods(methods):
    table = pt.PrettyTable(field_names=['id', 'scaling', 'scoring', 'rejection'])
    for i, method in enumerate(methods):
        table.add_row([i, method['scaling'], method['scoring'], method['rejection']])
    print(table)
