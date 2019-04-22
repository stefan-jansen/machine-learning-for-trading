# Chapter 15: Word-Vector Embeddings

This chapter introduces uses neural networks to learn a vector representation of individual semantic units like a word or a paragraph. These vectors are dense rather than sparse as in the bag-of-words model and have a few hundred real-valued rather than tens of thousand binary or discrete entries. They are called embeddings because they assign each semantic unit a location in a continuous vector space.
 
Embeddings result from training a model to relate tokens to their context with the benefit that similar usage implies a similar vector. As a result, the embeddings encode semantic aspects like relationships among words by means of their relative location. They are powerful features for use in the deep learning models that we will introduce in the following chapters. More specifically, in this chapter, we will cover:
- What word embeddings are, how they work and capture semantic information
- How to use trained word vectors
- Which network architectures are useful to train word2vec models
- How to train a word2vec model using keras, gensim, and TensorFlow
- How to visualize and evaluate the quality of word vectors
- How to train a word2vec model using SEC filings
- How doc2vec extends word2vec using SEC filings
- How doc2vec extends word2vec

## How Word Embeddings encode Semantics

Word embeddings represent tokens as lower-dimensional vectors so that their relative location reflects their relationship in terms of how they are used in context. They embody the distributional hypothesis from linguistics that claims words are best defined by the company they keep.

Word vectors are capable of capturing numerous semantic aspects; not only are synonyms close to each other, but words can have multiple degrees of similarity, e.g. the word ‘driver’ could be similar to ‘motorist’ or to ‘factor’. Furthermore, embeddings reflect relationships among pairs of words like analogies (Tokyo is to Japan what Paris is to France, or went is to go what saw is to see).  

### How neural language models learn usage in context

Word embeddings result from a training a shallow neural network to predict a word given its context. Whereas traditional language models define context as the words preceding the target, word embedding models use the words contained in a symmetric window surrounding the target. 

In contrast, the bag-of-words model uses the entire documents as context and uses (weighted) counts to capture the co-occurrence of words rather than predictive vectors.

### The word2vec Model: scalable word and phrase embeddings

A word2vec model is a two-layer neural net that takes a text corpus as input and outputs a set of embedding vectors for words in that corpus. There are two different architectures to efficiently learn word vectors using shallow neural networks.
- The continuous-bag-of-words (CBOW) model predicts the target word using the average of the context word vectors as input so that their order does not matter. CBOW trains faster and tends to be slightly more accurate for frequent terms, but pays less attention to infrequent words.
- The skip-gram (SG) model, in contrast, uses the target word to predict words sampled from the context. It works well with small datasets and finds good representations even for rare words or phrases.

### Evaluating embeddings: vector arithmetic and analogies

The dimensions of the word and phrase vectors do not have an explicit meaning. However, the embeddings encode similar usage as proximity in the latent space in a way that carries over to semantic relationships. This results in the interesting properties that analogies can be expressed by adding and subtracting word vectors.

Just as words can be used in different contexts, they can be related to other words in different ways, and these relationships correspond to different directions in the latent space. Accordingly, there are several types of analogies that the embeddings should reflect if the training data permits.

The word2vec authors provide a list of several thousand relationships spanning aspects of geography, grammar and syntax, and family relationships to evaluate the quality of embedding vectors (see directory [analogies](data/analogies)).

## Working with embedding models

Similar to other unsupervised learning techniques, the goal of learning embedding vectors is to generate features for other tasks like text classification or sentiment analysis.
There are several options to obtain embedding vectors for a given corpus of documents:
- Use embeddings learned from a generic large corpus like Wikipedia or Google News
- Train your own model using documents that reflect a domain of interest

### Using trained word vectors

There are several sources for pre-trained word embeddings. Popular options include Stanford’s GloVE and spaCy’s built-in vectors (see the notebook [using_trained_vectors ](02_using_trained_vectors.ipynb) for details).

#### GloVe: Global Vectors for Word Representation

GloVe is an unsupervised algorithm developed at the Stanford NLP lab that learns vector representations for words from aggregated global word-word co-occurrence statistics (see references). Vectors pre-trained on the following web-scale sources are available:
- Common Crawl with 42B or 840B tokens and a vocabulary of 1.9M or 2.2M tokens
- Wikipedia 2014 + Gigaword 5 with 6B tokens and a vocabulary of 400K tokens
- Twitter using 2B tweets, 27B tokens and a vocabulary of 1.2M tokens

The following table shows the accuracy on the word2vec semantics test achieved by the GloVE vectors trained on Wikipedia:

| Category                 | Samples | Accuracy | Category              | Samples | Accuracy |
|--------------------------|---------|----------|-----------------------|---------|----------|
| capital-common-countries | 506     | 94.86%   | comparative           | 1332    | 88.21%   |
| capital-world            | 8372    | 96.46%   | superlative           | 1056    | 74.62%   |
| city-in-state            | 4242    | 60.00%   | present-participle    | 1056    | 69.98%   |
| currency                 | 752     | 17.42%   | nationality-adjective | 1640    | 92.50%   |
| family                   | 506     | 88.14%   | past-tense            | 1560    | 61.15%   |
| adjective-to-adverb      | 992     | 22.58%   | plural                | 1332    | 78.08%   |
| opposite                 | 756     | 28.57%   | plural-verbs          | 870     | 58.51%   |

### How to train your own word vector embeddings

Many tasks require embeddings or domain-specific vocabulary that pre-trained models based on a generic corpus may not represent well or at all. Standard word2vec models are not able to assign vectors to out-of-vocabulary words and instead use a default vector that reduces their predictive value.

E.g., when working with industry-specific documents, the vocabulary or its usage may change over time as new technologies or products emerge. As a result, the embeddings need to evolve as well. In addition, corporate earnings releases use nuanced language not fully reflected in Glove vectors pre-trained on Wikipedia articles.

## Word Vectors from SEC Filings using gensim

In this section, we will learn word and phrase vectors from annual SEC filings using gensim to illustrate the potential value of word embeddings for algorithmic trading. In the following sections, we will combine these vectors as features with price returns to train neural networks to predict equity prices from the content of security filings.

In particular, we use a dataset containing over 22,000 10-K annual reports from the period 2013-2016 that are filed by listed companies and contain both financial information and management commentary (see chapter 3 on Alternative Data). For about half of 11K filings for companies that we have stock prices to label the data for predictive modeling (see references about data source and the notebooks in the folder [sec-filings](sec-filings) for details). 

- [2013-2016 Cleaned/Parsed 10-K Filings with the SEC](https://data.world/jumpyaf/2013-2016-cleaned-parsed-10-k-filings-with-the-sec)
- [Stock Market Predictions with Natural Language Deep Learning](https://www.microsoft.com/developerblog/2017/12/04/predicting-stock-performance-deep-learning/)

## Sentiment Analysis with Doc2Vec

Text classification requires combining multiple word embeddings. A common approach is to average the embedding vectors for each word in the document. This uses information from all embeddings and effectively uses vector addition to arrive at a different location point in the embedding space. However, relevant information about the order of words is lost. 

In contrast, the state-of-the-art generation of embeddings for pieces of text like a paragraph or a product review is to use the document embedding model doc2vec. This model was developed by the word2vec authors shortly after publishing their original contribution. Similar to word2vec, there are also two flavors of doc2vec:
- The distributed bag of words (DBOW) model corresponds to the Word2Vec CBOW model. The document vectors result from training a network on the synthetic task of predicting a target word based on both the context word vectors and the document's doc vector.
- The distributed memory (DM) model corresponds to the word2wec skipgram architecture. The doc vectors result from training a neural net to predict a target word using the full document’s doc vector.

The notebook [yelp_sentiment](doc2vec/yelp_sentiment.ipynb) applied doc2vec to a random sample of 1mn Yelp reviews with their associated star ratings.

## Bonus: word2vec for translation

- [Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/abs/1309.4168), Tomas Mikolov, Quoc V. Le, Ilya Sutskever, arxiv 2013
- [Word and Phrase Translation with word2vec](https://arxiv.org/abs/1705.03127), Stefan Jansen, arxiv, 2017

### Resources

- [GloVe: Global Vectors for Word Representation](https://github.com/stanfordnlp/GloVe)
- [Common Crawl Data](http://commoncrawl.org/the-data/)
- [word2vec analogy samples](https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt)
- [spaCy word vectors and semantic similarity](https://spacy.io/usage/vectors-similarity)
- [2013-2016 Cleaned/Parsed 10-K Filings with the SEC](https://data.world/jumpyaf/2013-2016-cleaned-parsed-10-k-filings-with-the-sec)