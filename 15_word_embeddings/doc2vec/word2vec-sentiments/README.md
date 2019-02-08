
# Sentiment Analysis using Doc2Vec

Word2Vec is dope. In short, it takes in a corpus, and churns out vectors for each of those words. What's so special about these vectors you ask? Well, similar words are near each other. Furthermore, these vectors represent how we use the words. For example, `v_man - v_woman` is approximately equal to `v_king - v_queen`, illustrating the relationship that "man is to woman as king is to queen". This process, in NLP voodoo, is called **word embedding**. These representations have been applied widely. This is made even more awesome with the introduction of Doc2Vec that represents not only words, but entire sentences and documents. Imagine being able to represent an entire sentence using a fixed-length vector and proceeding to run all your standard classification algorithms. Isn't that amazing?

However, Word2Vec documentation is shit. The C-code is nigh unreadable (700 lines of highly optimized, and sometimes weirdly optimized code). I personally spent a lot of time untangling Doc2Vec and crashing into ~50% accuracies due to implementation mistakes. This tutorial aims to help other users get off the ground using Word2Vec for their own research. We use Word2Vec for **sentiment analysis** by attempting to classify the Cornell IMDB movie review corpus (http://www.cs.cornell.edu/people/pabo/movie-review-data/). The specific data set used is available for download at http://ai.stanford.edu/~amaas/data/sentiment/.

## Show Me The Code

The IPython Notebook (code + tutorial) can be found in `word2vec-sentiments.ipynb`

The code to just run the Doc2Vec and save the model as `imdb.d2v` can be found in `run.py`. Should be useful for running on computer clusters.

## What Does This Repo Contain

- `test-neg.txt` `test-pos.txt` `train-neg.txt` `train-pos.txt` `train-unsup.txt` Training and testing data. Explained in more detail in the notebook.
- `word2vec-sentiment.ipynb` The notebook (code + tutorial)
- `run.py` Just the code

## License

Copyright (c) 2015 [Linan Qiu](https://github.com/linanqiu)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
