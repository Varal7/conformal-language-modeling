import os
import numpy as np
import json
from tqdm.auto import tqdm

from clm.components import (
    get_oracle_size_for_C_inner,
    get_rouge_label_for_C_inner,
    get_rouge_with_refs,
    get_preds,
    get_random_preds,
    get_first_k_preds,
)

for split in ["train", "val", "test"]:
    base = f"data/cxr/{split}"
    out_dir = f"data/cxr/{split}/components"

    with open(os.path.join(base, "row_rouge.jsonl")) as f:
        row_rouge_scores = [json.loads(line) for line in tqdm(f)]
    with open(os.path.join(base, "row_generation.jsonl")) as f:
        row_generation_idx_to_row_idx = [json.loads(line) for line in tqdm(f)]
    with open(os.path.join(base, "row_reference.jsonl")) as f:
        row_reference_idx_to_row_idx = [json.loads(line) for line in tqdm(f)]
    with open(os.path.join(base, "image_component_scores.jsonl")) as f:
        image_scores = [json.loads(line) for line in tqdm(f)]
    with open(os.path.join(base, "normprob_component_scores.jsonl")) as f:
        prob_scores = [json.loads(line) for line in tqdm(f)]


    rouge_with_refs = get_rouge_with_refs(row_rouge_scores, row_reference_idx_to_row_idx)
    rouge_label = get_rouge_label_for_C_inner(rouge_with_refs)
    oracle_size = get_oracle_size_for_C_inner(rouge_with_refs)
    _, _, K = rouge_with_refs.shape

    image_preds = get_preds(image_scores, row_generation_idx_to_row_idx, K)
    probs_preds = get_preds(prob_scores, row_generation_idx_to_row_idx, K)
    random_preds = get_random_preds(row_generation_idx_to_row_idx, K)
    firstk_preds = get_first_k_preds(row_generation_idx_to_row_idx, K)

    mask = firstk_preds != -1
    assert (mask == (image_preds != -1)).all()
    assert (mask == (probs_preds != -1)).all()
    assert (mask == (random_preds != -1)).all()

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "rouge_with_refs.npy"), rouge_with_refs)
    np.save(os.path.join(out_dir, "rouge_label.npy"), rouge_label)
    np.save(os.path.join(out_dir, "oracle_size.npy"), oracle_size)
    np.save(os.path.join(out_dir, "image_preds.npy"), image_preds)
    np.save(os.path.join(out_dir, "probs_preds.npy"), probs_preds)
    np.save(os.path.join(out_dir, "random_preds.npy"), random_preds)
    np.save(os.path.join(out_dir, "firstk_preds.npy"), firstk_preds)
    np.save(os.path.join(out_dir, "mask.npy"), mask)
