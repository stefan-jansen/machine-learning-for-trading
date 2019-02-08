#!/usr/bin/env bash
python word2vec.py --language=en --file=ngrams_1 --embedding_size=300 --num_neg_samples=20 --starter_lr=.1 --target_lr=.05 --batch_size=10 --min_count=10 --window_size=10
python word2vec.py --language=es --file=ngrams_1 --embedding_size=300 --num_neg_samples=20 --starter_lr=.1 --target_lr=.05 --batch_size=10 --min_count=10 --window_size=10