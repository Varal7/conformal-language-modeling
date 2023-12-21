"""Evaluate set uncertainty metrics."""

import collections
import itertools
import tqdm
import multiprocessing
import functools
import numpy as np
from scipy.stats import binom

from clm import scaling
from clm import scoring
from clm import utils

NAME_TO_SCALER = {
    'platt': scaling.PlattScaler,
    'bin': scaling.BinningScaler,
    'platt_bin': scaling.PlattBinningScaler,
    'rnn': scaling.RecurrentScaler,
    'none': scaling.BaseScaler,
}

NAME_TO_SCORE = {
    'geo': scoring.geometric,
    'marginal': scoring.marginal,
    'first_k': scoring.first_k,
    'first_k_no_mask': scoring.first_k_no_mask,
    'max': scoring.max,
    'sum': scoring.sum,
    'none': lambda x, m=None: x,
}

DEFAULT_METHODS = [
    dict(scaling=('none', {}), scoring='first_k', rejection=False),
    dict(scaling=('none', {}), scoring='first_k', rejection=True),
    # dict(scaling=('none', {}), scoring='max', rejection=False),
    dict(scaling=('none', {}), scoring='max', rejection=True),
    dict(scaling=('none', {}), scoring='sum', rejection=True),
    dict(scaling=('none', {}), scoring='first_k_no_mask', rejection=True),
    # dict(scaling=('platt', {}), scoring='geo', rejection=False),
    # dict(scaling=('platt', {}), scoring='geo', rejection=True),
]

# Default risk levels epsilon to calibrate.
DEFAULT_EPSILONS = np.linspace(0, 1, 101)


def compute_worst_case(values, losses, num_bins=20):
    """Compute worst case conditional loss.

    Values are binned by equal mass, and then we compute the loss over each
    bin. We then return the worst loss over all bins.

    Args:
        values: [num_examples]
            Array of example values (e.g., set size).
        losses: [num_examples]
            Array of set losses.

    Returns:
        [num_thresholds] array of worst-case average losses.
    """
    bins = np.quantile(values, np.linspace(0, 1, num_bins))
    binids = np.digitize(values, [0] + np.unique(bins))

    L_worst_avg = -1
    for binid in np.unique(binids):
        kept = binids == binid
        num_kept_examples = np.maximum(np.sum(kept), 1)
        Ls_mask_avg = np.sum(losses * kept) / num_kept_examples
        L_worst_avg = max(L_worst_avg, Ls_mask_avg)

    return L_worst_avg


def get_C_cutoff(set_scores, set_lambda):
    """Compute prediction sets C for given thresholds tau.

    Args:
        set_scores: [num_examples, max_generations]
            set_scores[i, j] = score of set j for example i.
        set_lambda: Threshold to use.

    Returns:
        C_indices: [num_examples]
            Indices of the selected sets C for each example.
    """
    cummax_scores = np.maximum.accumulate(set_scores, axis=-1)
    mask = cummax_scores < set_lambda
    C_indices = np.sum(mask, axis=-1)
    C_indices = C_indices.clip(0, set_scores.shape[1] - 1)
    return C_indices


def compute_values(
    config,
    item_scores,
    similarity_scores,
    item_labels,
    set_score_fn,
    do_rejection=True,
    return_indices=False,
):
    """Compute list of metrics for a given config.

    Args:
        set_scores: [num_examples, max_generations]
            set_scores[i, j] = score of set after sample j for example i.
        set_sizes: [num_examples, max_generations]
            set_sizes[i, j] = effective set size after sample j for example i.
        set_losses: [num_examples, max_generations]
            set_loss[i, j] = loss of set after sample j for example i.
        lambdas: [num_thresholds]
            Array of thresholds to test.

    Returns:
        Dictionary of metrics (per lambda).
    """
    if np.any([np.isnan(l) for l in config]):
        return dict(
            L_avg=np.nan,
            L_worst_pred_avg=np.nan,
            C_size_avg=np.nan,
            C_samples_avg=np.nan,
            C_excess_avg=np.nan,
            C_relative_excess_avg=np.nan,
            C_obj_avg=np.nan,
        )

    lambda_1, lambda_2, lambda_3 = config

    # If doing rejections, remove low quality and low diversity items.
    if do_rejection:
        # Reject low-quality examples.
        quality_mask = (item_scores >= lambda_2)

        # Reject examples that are too similar to previous items.
        similarity_mask = np.ones_like(quality_mask)

        # Make similarity score symmetric.
        similarity_scores = np.maximum(similarity_scores, similarity_scores.transpose((0, 2, 1)))

        # Low quality scores are rejected, so they don't count --- set similarity to 0.
        similarity_scores = similarity_scores * np.expand_dims(quality_mask, 1)

        for k in range(1, similarity_scores.shape[1]):
            # Anything that has been rejected up until this point also doesn't count.
            # Set those scores to 0 so we don't pick them up as well.
            similarity_scores = similarity_scores * np.expand_dims(similarity_mask, 1)

            # We only look at scores up until this point.
            max_similarity = np.max(similarity_scores[:, k, :k], axis=-1)
            similarity_mask[:, k] = max_similarity <= lambda_1

        # Combine rejection rules.
        kept_mask = quality_mask * similarity_mask
    else:
        kept_mask = np.ones_like(item_scores)

    # Get set losses, scores, and sizes.
    set_sizes = np.cumsum(kept_mask, axis=-1)
    set_losses = utils.set_losses_from_labels(item_labels * kept_mask)
    set_scores = set_score_fn(item_scores, kept_mask)

    # Compute set selections for all values of lambda.
    C_indices = get_C_cutoff(set_scores, lambda_3)

    # Compute selected set losses.
    num_examples = set_losses.shape[0]
    Ls = set_losses[np.arange(num_examples), C_indices]
    L_avg = np.mean(Ls)

    # Compute effective set sizes.
    C_sizes = set_sizes[np.arange(num_examples), C_indices]
    C_size_avg = np.mean(C_sizes)

    # Compute the oracle number of samples to draw.
    max_generations = set_losses.shape[1]
    set_losses_without_rejection = utils.set_losses_from_labels(item_labels)
    C_oracle_samples = np.sum(set_losses_without_rejection > 0, axis=-1) + 1
    C_oracle_samples = C_oracle_samples.clip(0, max_generations)

    # Compute number of sample metrics.
    C_samples = C_indices + 1
    C_samples_avg = np.mean(C_samples)
    C_excess_avg = np.mean(np.maximum(C_samples - C_oracle_samples, 0))
    C_relative_excess_avg = np.mean(
        np.maximum(C_samples - C_oracle_samples, 0) / C_samples)

    # Compute worst-case set losses, conditioned on predicted size.
    L_worst_pred_avg = compute_worst_case(C_sizes, Ls)

    output = dict(
        # Average loss.
        L_avg=L_avg,
        # Worst case loss binned by set size.
        L_worst_pred_avg=L_worst_pred_avg,
        # Average set size.
        C_size_avg=C_size_avg / max_generations,
        # Average number of samples.
        C_samples_avg=C_samples_avg / max_generations,
        # Average excess samples.
        C_excess_avg=C_excess_avg,
        # Relative excess,
        C_relative_excess_avg=C_relative_excess_avg,
        # Combined metric.
        C_obj_avg=C_samples_avg + C_size_avg,
    )

    if return_indices:
        output.update(
            dict(
                C_indices=C_indices,
                kept_mask=kept_mask,
                Ls=Ls,
            )
        )

    return output


def get_pareto_frontier(
    item_scores,
    similarity_scores,
    item_labels,
    set_score_fn,
    do_rejection,
):
    """Compute a pareto frontier."""
    lambda_1 = utils.select_lambdas(similarity_scores, max_lambdas=25)
    lambda_2 = utils.select_lambdas(item_scores, max_lambdas=25)
    lambda_3 = utils.select_lambdas(set_score_fn(item_scores), max_lambdas=25)

    configs = []
    costs = []
    for config in tqdm.tqdm(itertools.product(lambda_1, lambda_2, lambda_3)):
        configs.append(config)
        values = compute_values(
            config=config,
            item_scores=item_scores,
            similarity_scores=similarity_scores,
            item_labels=item_labels,
            set_score_fn=set_score_fn,
            do_rejection=do_rejection)
        costs.append((values['L_avg'], values['C_obj_avg']))

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

    return ordered_configs


def init(shared_data):
    global SHARED_DATA
    SHARED_DATA = shared_data


def run(args, methods, epsilons, delta):
    global SHARED_DATA
    all_item_scores, test_data = SHARED_DATA
    opt_idx, cal_idx, test_idx = args

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
        set_score_fn = NAME_TO_SCORE[methods[i].get('scoring', 'none')]
        do_rejection = methods[i].get('rejection', False)

        # Get pareto frontier for Pareto Testing.
        configs = get_pareto_frontier(
            item_scores=opt_item_scores,
            similarity_scores=opt_similarity_scores,
            item_labels=opt_item_labels,
            set_score_fn=set_score_fn,
            do_rejection=do_rejection)

        # Choose best valid configs (configs are already ordered).
        best_valid_configs = [[np.nan] * 3] * len(epsilons)
        is_stopped = [False] * len(epsilons)
        for config in configs:
            values = compute_values(
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
            values = compute_values(
                config=config,
                item_scores=test_item_scores,
                similarity_scores=test_similarity_scores,
                item_labels=test_item_labels,
                set_score_fn=set_score_fn,
                do_rejection=do_rejection)
            for k, v in values.items():
                trial_results[k].append(v)
        all_trial_results.append(trial_results)

    return all_trial_results


def run_trials(
    train_data,
    test_data,
    epsilons=DEFAULT_EPSILONS,
    methods=DEFAULT_METHODS,
    delta=0.05,
    p_cal=0.3,
    num_trials=100,
    num_processes=1,
):
    """Run multiple random trials of CRC.

    Args:
        train_data: utils.Dataset of (scores, labels).
        test_data: utils.Dataset of (scores, labels).
        alphas: List of target risk levels to calibrate for.
        methods: List of ((scale_type, scale_kwargs), score_type).
            If scale_type != 'none', then train_data is used to learn a scaler.
            Then the score_type maps to a function to compute set scores.
        p_cal: Percentage of test data to use for calibration in each trial.
        num_trials: Number of random trials to do.

    Returns:
        methods: List of evaluated methods.
        alphas: List of evaluated alphas.
        results: List of dict for each method mapping result type to a
            [num_trials, num_alphas] array of scores (see compute_values).
    """
    # Compute scaled scores.
    all_item_scores = []
    for method in tqdm.tqdm(methods, desc='scaling'):
        scale_type, scale_kwargs = method.get('scaling', ('none', {}))
        scaler = NAME_TO_SCALER[scale_type](**scale_kwargs)
        scaler.fit(train_data.scores, train_data.labels)
        all_item_scores.append(scaler.predict(test_data.scores))

    # Initialize results.
    all_results = [collections.defaultdict(list) for _ in range(len(methods))]
    run_fn = functools.partial(run, methods=methods, delta=delta, epsilons=epsilons)
    if num_processes > 1:
        pool = multiprocessing.Pool(
            num_processes,
            initializer=init,
            initargs=((all_item_scores, test_data),))
        map_fn = pool.imap_unordered
    else:
        global SHARED_DATA
        SHARED_DATA = (all_item_scores, test_data)
        map_fn = map

    # Calibrate and compute results across trials.
    N = len(test_data.scores)
    N_cal = int(p_cal * N)
    N_pcal = int(0.7 * N_cal)
    splits = []
    for _ in range(num_trials):
        randperm = np.random.permutation(N)
        pareto_idx, test_idx = randperm[:N_cal], randperm[N_cal:]
        cal_idx, opt_idx = pareto_idx[:N_pcal], pareto_idx[N_pcal:]
        splits.append((opt_idx, cal_idx, test_idx))

    with tqdm.tqdm(total=num_trials, desc='running trials') as pbar:
        for trial_results in map_fn(run_fn, splits):
            for i, method_results in enumerate(trial_results):
                for k, v in method_results.items():
                    all_results[i][k].append(np.array(v))
            pbar.update()

    # Aggregate results.
    combined_results = []
    for results in all_results:
        combined = {}
        for k, v in results.items():
            combined[k] = np.stack(v, axis=0)
        combined_results.append(combined)

    return methods, epsilons, combined_results
