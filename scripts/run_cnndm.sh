#! /bin/bash

SPLIT='test'

for score in 'probs'; do
    python scripts/run_trials.py \
           --train_score_file "data/cnndm/train/${score}.npy" \
           --train_label_file "data/cnndm/train/labels.npy" \
           --train_similarity_file "data/cnndm/train/diversity.npy" \
           --test_score_file "data/cnndm/${SPLIT}/${score}.npy" \
           --test_label_file "data/cnndm/${SPLIT}/labels.npy" \
           --test_similarity_file "data/triviaqa/${SPLIT}/diversity.npy" \
           --output_file "results/cnndm/${SPLIT}/${score}_results.npz"
done
