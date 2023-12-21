import numpy as np
import collections
import tqdm

from clm import scaling
from clm import utils
from scipy.stats import binom

# Default risk levels epsilon to calibrate.
DEFAULT_EPSILONS = np.linspace(0, 1, 51)

NAME_TO_SCALER = {
    'platt': scaling.PlattScaler,
    'bin': scaling.BinningScaler,
    'platt_bin': scaling.PlattBinningScaler,
    'none': scaling.BaseScaler,
}


def get_preds(scores, row_generation_idx_to_row_idx, K):
    """
        row_generation_idx_to_row_idx:  list of list of list of int
            The first list is for each example
            The second list is for each generation
            The third list is for each sentence in the generation
            The int is the index of the sentence in the flattened list

        returns: np.array of shape (num_examples, max_num_sentences)
    """
    N = len(row_generation_idx_to_row_idx)
    g = len(row_generation_idx_to_row_idx[0])

    preds = -np.ones((N, K))

    for i in range(N):
        for j in range(g):
            for k in range(len(row_generation_idx_to_row_idx[i][j])):
                idx = row_generation_idx_to_row_idx[i][j][k]
                preds[i, idx] = scores[i][j][k]

    return preds

def get_random_preds(row_reference_idx_to_row_idx, K):
    """
        row_reference_idx_to_row_idx:  list of list of int
            The first list is for each example
            The second list is for each sentence in the reference
            The int is the index of the sentence in the flattened list

        returns: np.array of shape (num_examples, max_num_sentence_refs)
    """
    N = len(row_reference_idx_to_row_idx)
    g = len(row_reference_idx_to_row_idx[0])

    preds = -np.ones((N, K))

    for i in range(N):
        for j in range(g):
            for k in range(len(row_reference_idx_to_row_idx[i][j])):
                idx = row_reference_idx_to_row_idx[i][j][k]
                preds[i, idx] = np.random.rand()

    return preds

def get_first_k_preds(row_reference_idx_to_row_idx, K, max_num_sentences=50):
    N = len(row_reference_idx_to_row_idx)
    g = len(row_reference_idx_to_row_idx[0])

    preds = -np.ones((N, K))

    for i in range(N):
        for j in range(g):
            for k in range(min(K, len(row_reference_idx_to_row_idx[i][j]))):
                idx = row_reference_idx_to_row_idx[i][j][k]
                pred = max(1.0 - (k / max_num_sentences), 0.0)
                preds[i, idx] = max(pred, preds[i, idx])

    return preds

def get_rouge_with_refs(row_rouge_scores, row_reference_idx_to_row_idx):
    """
        row_rouge_scores: list of np.array of shape (num_sentences, num_sentences)
            The first list is for each example
            The np.array is the rouge score matrix of each sentence pair for that row
        row_reference_idx_to_row_idx:  list of list of int
            The first list is for each example
            The second list is for each sentence in the reference
            The int is the index of the sentence in the flattened list

        returns: np.array of shape (num_examples, max_num_sentence_refs, max_num_sentences)
    """
    N = len(row_rouge_scores)
    R = max(len(row_reference_idx_to_row_idx[i]) for i in range(N))
    K = max(len(row_rouge_scores[i]) for i in range(N))

    rouge_refs = -np.ones((N, R, K))


    for i in range(N):
        arr = np.array(row_rouge_scores[i])
        refs = row_reference_idx_to_row_idx[i]
        rouge_refs[i, :len(refs), :len(arr)] = arr[refs, :]

    return rouge_refs

def get_rouge_label_for_C_inner(rouge_with_refs):
    """
        rouge_with_refs: np.array of shape (num_examples, max_num_sentence_refs, max_num_sentences)

        returns: np.array of shape (num_examples, max_num_sentences)
    """
    return rouge_with_refs.max(axis=1)


def get_oracle_size_for_C_inner(rouge_with_refs):
    """
        rouge_with_refs: np.array of shape (num_examples, max_num_sentence_refs, max_num_sentences)

        returns: np.array of shape (num_examples)
    """
    return (rouge_with_refs > 0).any(axis=2).sum(axis=1)


def compute_values_for_C_inner_basic(preds, rouge_label, mask, oracle_size, rouge_threshold=0.7, taus=None, num_quantiles=1000, subset=None):
    """
        preds: np.array of shape (num_examples, max_num_sentences)
        rouge_label: np.array of shape (num_examples, max_num_sentences)
        mask: np.array of shape (num_examples, max_num_sentences)
              mask = 0 if the sentence is padding
        oracle_size: np.array of shape (num_examples) denoting the
              number of sentences in the reference


        returns: np.array of shape (num_examples, max_num_sentences)
    """
    if taus is None:
        quantiles = np.linspace(0, 1, num_quantiles)
        taus = np.unique(np.quantile(preds.reshape(-1), quantiles))
        taus = np.concatenate([[-np.inf], taus, [np.inf]])
        taus = np.flip(taus)

    if subset is not None:
        preds = preds[subset]
        rouge_label = rouge_label[subset]
        mask = mask[subset]
        oracle_size = oracle_size[subset]

    has_refs = oracle_size > 0

    # Discard examples without references
    preds = preds[has_refs]
    rouge_label = rouge_label[has_refs]
    mask = mask[has_refs]

    oracle_size = oracle_size[has_refs].reshape(1, -1) # 1, num_examples

    accepted = rouge_label > rouge_threshold         # num_examples x num_sentences
    accepted = np.expand_dims(accepted, axis=0)          # 1 x num_examples x num_sentences

    chosen = (preds > taus.reshape(-1, 1, 1)) # num_taus x num_examples x num_sentences

    # Ok if [chosen implies accepted]
    ok = (~chosen | accepted).all(axis=-1) # num_taus x num_examples
    C_size = chosen.sum(axis=-1) # num_taus x num_examples

    # Additional metrics
    # Fake recall is counting the number of chosen sentences that are accepted and dividing by size of the reference
    # Recall is counting the number of reference sentences that have a match and dividing by the size of the reference

    C_relative_size = C_size / oracle_size
    #  fake_recall  = (chosen & accepted).sum(axis=-1) / oracle_size
    #  intersect = np.expand_dims(accepted, axis=0) & np.expand_dims(chosen, axis=2) # taus x num_examples x num_refs x num_sentences
    #  intersect = intersect.any(axis=-1) # taus x num_examples x num_refs
    #  recall = intersect.sum(axis=-1) / oracle_size # taus x num_examples
    #  micro_recall = intersect.sum(axis=-1).sum(axis=-1) / oracle_size.sum(axis=-1) # taus

    # conditioned
    C_size_conditioned = (C_size * ok).sum(axis=1) / ok.sum(axis=1) # num_taus
    #  micro_recall_conditioned = (intersect.sum(axis=-1) * ok).sum(axis=1) / ((oracle_size * ok).sum(axis=-1) + 1e-10)


    return {
        'L_avg': (1 - ok).mean(axis=-1),
        'C_size_avg': C_size.mean(axis=-1),
        'C_relative_size': C_relative_size.mean(axis=-1),
        #  'fake_recall': fake_recall.mean(axis=-1),
        #  'recall': recall.mean(axis=-1),
        #  'micro_recall': micro_recall,
        'C_size_conditioned': C_size_conditioned,
        #  'micro_recall_conditioned': micro_recall_conditioned,
    }

def compute_values_for_C_inner(preds, rouge_with_refs, rouge_threshold=0.7, taus=None, num_quantiles=1000, subset=None):
    if taus is None:
        quantiles = np.linspace(0, 1, num_quantiles)
        taus = np.unique(np.quantile(preds.reshape(-1), quantiles))
        taus = np.concatenate([[-np.inf], taus, [np.inf]])
        taus = np.flip(taus)

    if subset is not None:
        preds = preds[subset]
        rouge_with_refs = rouge_with_refs[subset]

    oracle_size = (rouge_with_refs > 0).any(axis=-1).sum(axis=-1) # num_examples
    has_refs = oracle_size > 0

    # Discard examples without references
    preds = preds[has_refs]
    rouge_with_refs = rouge_with_refs[has_refs]

    oracle_size = oracle_size[has_refs].reshape(1, -1) # 1, num_examples

    accepted = rouge_with_refs > rouge_threshold         # num_examples x num_refs x num_sentences

    any_accepted = accepted.any(axis=1)                      # num_examples x num_sentences
    any_accepted = np.expand_dims(any_accepted, axis=0)          # 1 x num_examples x num_sentences

    chosen = (preds > taus.reshape(-1, 1, 1)) # num_taus x num_examples x num_sentences

    # Ok if [chosen implies accepted]
    ok = (~chosen | any_accepted).all(axis=-1) # num_taus x num_examples
    C_size = chosen.sum(axis=-1) # num_taus x num_examples


    # Additional metrics
    # Fake recall is counting the number of chosen sentences that are accepted and dividing by size of the reference
    # Recall is counting the number of reference sentences that have a match and dividing by the size of the reference

    C_relative_size = C_size / oracle_size
    #  fake_recall  = (chosen & any_accepted).sum(axis=-1) / oracle_size
    intersect = np.expand_dims(accepted, axis=0) & np.expand_dims(chosen, axis=2) # taus x num_examples x num_refs x num_sentences
    intersect = intersect.any(axis=-1) # taus x num_examples x num_refs
    recall = intersect.sum(axis=-1) / oracle_size # taus x num_examples
    micro_recall = intersect.sum(axis=-1).sum(axis=-1) / oracle_size.sum(axis=-1) # taus

    # conditioned
    #  C_size_conditioned = (C_size * ok).sum(axis=1) / ok.sum(axis=1) # num_taus
    micro_recall_conditioned = (intersect.sum(axis=-1) * ok).sum(axis=1) / ((oracle_size * ok).sum(axis=-1) + 1e-10)


    return {
        'taus': taus,
        'L_avg': (1 - ok).mean(axis=-1),
        'C_size_avg': C_size.mean(axis=-1),
        'C_relative_size': C_relative_size.mean(axis=-1),
        #  'fake_recall': fake_recall.mean(axis=-1),
        'recall': recall.mean(axis=-1),
        'micro_recall': micro_recall,
        #  'C_size_conditioned': C_size_conditioned,
        'micro_recall_conditioned': micro_recall_conditioned,
    }


def compute_values_for_C_outer(preds, rouge_with_multi_refs, rouge_threshold=0.7, taus=None, num_quantiles=1000, subset=None):
    """
        rouge_with_multi_refs: max_num_refs x num_examples x max_num_sentence_refs x num_sentences
        preds: num_examples x num_sentences
    """

    if taus is None:
        quantiles = np.linspace(0, 1, num_quantiles)
        taus = np.unique(np.quantile(preds.reshape(-1), quantiles))
        taus = np.concatenate([[0], taus, [np.inf]])
        taus = np.flip(taus)

    if subset is not None:
        preds = preds[subset]
        rouge_with_multi_refs = rouge_with_multi_refs[:, subset]

    # Compute oracle size based on true ref
    oracle_size = (rouge_with_multi_refs[0] > 0).any(axis=-1).sum(axis=-1) # num_examples
    has_refs = oracle_size > 0

    # Discard examples without references
    preds = preds[has_refs]
    rouge_with_multi_refs = rouge_with_multi_refs[:, has_refs]

    oracle_size = oracle_size[has_refs].reshape(1, -1) # 1, num_examples

    # For each (example, ref), how many sentences are actually there
    ref_sentence_padding = (rouge_with_multi_refs < 0).all(axis=-1)  # max_num_refs x num_examples x max_num_sentence_refs

    # For each example, how many refs are actually there
    ref_mask = 1 - (rouge_with_multi_refs < 0).all(axis=-1).all(axis=-1)  # max_num_refs x num_examples


    # accept if the rouge score is above the threshold
    accepted = rouge_with_multi_refs > rouge_threshold                    # max_num_refs x num_examples x max_num_sentence_refs x num_sentences
    r_accepted  = np.expand_dims(accepted, axis=0)                       # 1 x max_num_refs x num_examples x max_num_sentence_refs x num_sentences

    # choose if preds is above the tau
    chosen = (preds  > taus.reshape(-1, 1, 1)) # num_taus x num_examples x num_sentences
    r_chosen = np.expand_dims(chosen, axis=1)  # num_taus x 1 x num_examples x  num_sentences
    r_chosen = np.expand_dims(r_chosen, axis=3) # num_taus x 1 x num_examples x 1 x num_sentences

    # For each reference_sentence, ok if at least one of the chosen sentences is accepted
    accepted_and_chosen = r_accepted & r_chosen # num_taus x max_num_refs x num_examples x max_num_sentence_refs x num_sentences

    ok_ref_sen = (accepted_and_chosen).any(axis=4)  # num_taus x max_num_refs x num_examples x max_num_sentence_refs

    # Ok if padding
    ok_ref_sen = ok_ref_sen | np.expand_dims(ref_sentence_padding, axis=0)  # num_taus x max_num_refs x num_examples x max_num_sentence_refs

    # (tau, example, ref) ok if all reference sentences are ok
    ok_ref = ok_ref_sen.all(axis=3) # num_taus x max_num_refs x num_examples

    # must be a real reference
    ok_ref = ok_ref & np.expand_dims(ref_mask, axis=0) # num_taus x max_num_refs x num_examples

    # (tau, example) ok if any of the references are ok
    ok = ok_ref.any(axis=1) # num_taus x num_examples

    C_size = chosen.sum(axis=-1) # num_taus x num_examples

    # additional metrics
    accepted_sen = accepted_and_chosen.any(axis=3).any(axis=1) # num_taus x num_examples x num_sentences
    precision = accepted_sen.sum(axis=2) / (chosen.sum(axis=2) + 1e-10) # num_taus x num_examples
    C_relative_size = C_size / oracle_size
    micro_precision = accepted_sen.sum(axis=2).sum(axis=1) / (C_size.sum(axis=1) + 1e-10) # num_taus

    # conditioned
    micro_precision_conditioned = (accepted_sen.sum(axis=2) * ok).sum(axis=1) / ((C_size * ok).sum(axis=1) + 1e-10) # num_taus
    C_size_conditioned = (C_size * ok).sum(axis=1) / ok.sum(axis=1) # num_taus

    return {
        'L_avg': (1 - ok).mean(axis=-1),
        'C_size_avg': C_size.mean(axis=-1),
        'C_relative_size': C_relative_size.mean(axis=-1),
        'precision': precision.mean(axis=-1),
        'micro_precision': micro_precision,
        'micro_precision_conditioned': micro_precision_conditioned,
        'C_size_conditioned': C_size_conditioned,
        'oracle_size': oracle_size.repeat(len(taus), axis=0).mean(axis=-1),
        'taus': taus,
    }


def run_trials(
    train_data: utils.ComponentDataset,
    test_data: utils.ComponentDataset,
    scale_type="none",
    scale_kwargs=None,
    filter_for_answerable=False,
    rouge_threshold=0.4,
    num_quantiles=1000,
    epsilons=DEFAULT_EPSILONS,
    delta=0.05,
    p_cal=0.3,
    num_trials=100,
):
    """Run multiple random trials of CRC on Components

    Args:
        train_data: utils.ComponentDataset of (scores, rouge_with_refs).
        test_data: utils.ComponentDataset of (scores, rouge_with_refs).
        alphas: List of target risk levels to calibrate for.
        scaling: One of "platt", "bin", "platt_bin", or "none".
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

    scale_kwargs = scale_kwargs or {}
    scaler = NAME_TO_SCALER[scale_type](**scale_kwargs)
    scaler.fit(train_data.scores, train_data.rouge_with_refs)
    item_scores = (scaler.predict(test_data.scores))

    # Initialize results.
    all_results = collections.defaultdict(list)

    # Calibrate and compute results across trials.
    N = len(test_data.scores)
    N_cal = int(p_cal * N)
    N_pcal = N_cal // 2
    for _ in tqdm.tqdm(range(num_trials), desc='running trials'):
        # Split into calibration and test sets.
        randperm = np.random.permutation(N)
        pareto_idx, test_idx = randperm[:N_cal], randperm[N_cal:]
        cal_idx, opt_idx = pareto_idx[:N_pcal], pareto_idx[N_pcal:]

        del opt_idx # We don't use pareto testing for components

        cal_item_scores = item_scores[cal_idx]
        cal_item_rouge = test_data.rouge_with_refs[cal_idx]

        test_item_scores = item_scores[test_idx]
        test_item_rouge = test_data.rouge_with_refs[test_idx]

        # If filtering for answerable, then only keep examples
        # where at least one generation is acceptable
        if filter_for_answerable:
            cal_labels = test_data.report_labels[cal_idx]
            test_labels = test_data.report_labels[test_idx]

            cal_oracle_subset = cal_labels.any(axis=1)
            test_oracle_subset = test_labels.any(axis=1)

            cal_item_scores = cal_item_scores[cal_oracle_subset]
            cal_item_rouge = cal_item_rouge[cal_oracle_subset]

            test_item_scores = test_item_scores[test_oracle_subset]
            test_item_rouge = test_item_rouge[test_oracle_subset]


        params = dict(
            rouge_threshold=rouge_threshold,
            num_quantiles=num_quantiles,
        )

        values = compute_values_for_C_inner(cal_item_scores, cal_item_rouge, **params)

        # Choose best valid tau (already sorted in decreasing tau)
        best_valid_taus = [np.nan] * len(epsilons)
        is_stopped = [False] * len(epsilons)
        for (tau, loss) in zip(values['taus'], values['L_avg']):
            for j, epsilon in enumerate(epsilons):
                n = len(cal_idx)
                p_value = binom.cdf(n * loss, n, epsilon)
                if p_value <= delta and not is_stopped[j]:
                    best_valid_taus[j] = tau
                else:
                    is_stopped[j] = True
        best_valid_taus = np.array(best_valid_taus)

        # Compute test metrics.
        test_values = compute_values_for_C_inner(test_item_scores, test_item_rouge, taus=best_valid_taus, **params)

        # Record trial.
        for k, v in test_values.items():
            all_results[k].append(v)

    # Aggregate results.
    combined = {}
    for k, v in all_results.items():
        combined[k] = np.stack(v, axis=0)

    return epsilons, combined
