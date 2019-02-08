# coding: utf-8

from pathlib import Path
from argparse import ArgumentParser
from time import time
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

np.random.seed(42)

LANGUAGES = ['en', 'es']


def combine_files():
    for language in LANGUAGES:
        source_dir = DATA_DIR / language / 'sentences'
        target_file = Path('wiki', language, 'wiki.txt')
        with target_file.open('a') as target:
            for source in source_dir.glob('*.txt'):
                for line in source.open('r'):
                    target.write(line)


def get_accuracy(acc, detail=False):
    results = [[c['section'], len(c['correct']), len(c['incorrect'])] for c in acc]
    results = pd.DataFrame(results, columns=['category', 'correct', 'incorrect'])
    results['average'] = results.correct.div(results[['correct', 'incorrect']].sum(1))
    results.sort_values('average', ascending=False)
    if detail:
        print(results)
    return results.iloc[-1, 1:].tolist()

language = 'es'
PROJECT_DIR = Path('/home/stefan/projects/odsc_2018/word2vec-translation')
ANALOGIES_PATH = PROJECT_DIR / 'data' / 'analogies' / 'analogies-{}.txt'.format(language)

gensim_path = Path('wiki', language)
if not gensim_path.exists():
    gensim_path.mkdir(parents=True, exist_ok=True)

sentence_path = gensim_path / 'wiki.txt'
sentences = LineSentence(str(sentence_path))
start = time()
model = Word2Vec(sentences,
                 sg=1,
                 size=300,
                 window=5,
                 min_count=5,
                 negative=10,
                 workers=8,
                 iter=1,
                 alpha=0.05)

print('Duration: {:,.1f}s'.format(time() - start))

model.wv.save(str(gensim_path / 'word_vectors.bin'))

acc = get_accuracy(model.wv.accuracy(str(ANALOGIES_PATH), case_insensitive=True))
print('Base Accuracy: Correct {:,d} | Wrong {:,d} | Avg {:,.2%}\n'.format(*acc))

accuracies = [acc]
for i in range(1, 11):
    start = time()
    model.train(sentences, epochs=1, total_examples=model.corpus_count)
    accuracies.append(get_accuracy(model.wv.accuracy(str(ANALOGIES_PATH))))
    print('{} | Duration: {:,.1f} | Accuracy: {:.2%} '.format(i, time() - start, accuracies[-1][-1]))

pd.DataFrame(accuracies, columns=['correct', 'wrong', 'average']).to_csv(gensim_path / 'accuracies.csv', index=False)
model.wv.save(str(gensim_path / 'word_vectors_final.bin'))
