#! /bin/bash

SPLIT='test'

for score in 'probs'; do
    python scripts/run_trials.py \
           --train_score_file "data/triviaqa/train/${score}.npy" \
           --train_label_file "data/triviaqa/train/labels.npy" \
           --train_similarity_file "data/triviaqa/train/diversity.npy" \
           --test_score_file "data/triviaqa/${SPLIT}/${score}.npy" \
           --test_label_file "data/triviaqa/${SPLIT}/labels.npy" \
           --test_similarity_file "data/triviaqa/${SPLIT}/diversity.npy" \
           --output_file "results/triviaqa/${SPLIT}/${score}_results.npz"
done
