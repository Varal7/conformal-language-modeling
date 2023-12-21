#! /bin/bash

SPLIT='test'

for score in 'probs'; do
    python scripts/run_trials.py \
           --train_score_file "data/cxr/train/${score}.npy" \
           --train_label_file "data/cxr/train/labels.npy" \
           --train_similarity_file "data/cxr/train/diversity_rouge.npy" \
           --test_score_file "data/cxr/${SPLIT}/${score}.npy" \
           --test_label_file "data/cxr/${SPLIT}/labels.npy" \
           --test_similarity_file "data/cxr/${SPLIT}/diversity_rouge.npy" \
           --output_file "results/cxr/${SPLIT}/${score}_results.npz"
done
