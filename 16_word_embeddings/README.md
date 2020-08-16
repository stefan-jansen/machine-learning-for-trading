# Word Embeddings for Earnings Calls and SEC Filings 

This chapter introduces uses neural networks to learn a vector representation of individual semantic units like a word or a paragraph. These vectors are dense rather than sparse as in the bag-of-words model and have a few hundred real-valued rather than tens of thousand binary or discrete entries. They are called embeddings because they assign each semantic unit a location in a continuous vector space.
 
Embeddings result from training a model to relate tokens to their context with the benefit that similar usage implies a similar vector. As a result, the embeddings encode semantic aspects like relationships among words by means of their relative location. They are powerful features for use in the deep learning models that we will introduce in the following chapters.


## Content

1. [How Word Embeddings encode Semantics](#how-word-embeddings-encode-semantics)
    * [How neural language models learn usage in context](#how-neural-language-models-learn-usage-in-context)
    * [The word2vec Model: scalable word and phrase embeddings](#the-word2vec-model-scalable-word-and-phrase-embeddings)
    * [Evaluating embeddings: vector arithmetic and analogies](#evaluating-embeddings-vector-arithmetic-and-analogies)
2. [Code example: Working with embedding models](#code-example-working-with-embedding-models)
    * [Working with Global Vectors for Word Representation (GloVe)](#working-with-global-vectors-for-word-representation-glove)
    * [Evaluating embeddings using analogies](#evaluating-embeddings-using-analogies)
3. [Code example: training domain-specific embeddings using financial news](#code-example-training-domain-specific-embeddings-using-financial-news)
    * [Preprocessing financial news: sentence detection and n-grams](#preprocessing-financial-news-sentence-detection-and-n-grams)
    * [Skip-gram architecture in TensorFlow 2 and visualization with TensorBoard](#skip-gram-architecture-in-tensorflow-2-and-visualization-with-tensorboard)
    * [How to train embeddings faster with Gensim](#how-to-train-embeddings-faster-with-gensim)
4. [Code Example: word Vectors from SEC Filings using gensim](#code-example-word-vectors-from-sec-filings-using-gensim)
    * [Preprocessing: content selection, sentence detection, and n-grams](#preprocessing-content-selection-sentence-detection-and-n-grams)
    * [Model training and evaluation](#model-training-and-evaluation)
5. [Code example: sentiment Analysis with Doc2Vec](#code-example-sentiment-analysis-with-doc2vec)
6. [New Frontiers: Attention, Transformers, and Pretraining](#new-frontiers-attention-transformers-and-pretraining)
    * [Attention is all you need: transforming natural language generation](#attention-is-all-you-need-transforming-natural-language-generation)
    * [BERT: Towards a more universal, pretrained language model](#bert-towards-a-more-universal-pretrained-language-model)
    * [Using pretrained state-of-the-art models](#using-pretrained-state-of-the-art-models)
7. [Additional Resources](#additional-resources)

## How Word Embeddings encode Semantics

Word embeddings represent tokens as lower-dimensional vectors so that their relative location reflects their relationship in terms of how they are used in context. They embody the distributional hypothesis from linguistics that claims words are best defined by the company they keep.

Word vectors are capable of capturing numerous semantic aspects; not only are synonyms close to each other, but words can have multiple degrees of similarity, e.g. the word ‘driver’ could be similar to ‘motorist’ or to ‘factor’. Furthermore, embeddings reflect relationships among pairs of words like analogies (Tokyo is to Japan what Paris is to France, or went is to go what saw is to see).  

### How neural language models learn usage in context

Word embeddings result from a training a shallow neural network to predict a word given its context. Whereas traditional language models define context as the words preceding the target, word embedding models use the words contained in a symmetric window surrounding the target. 

In contrast, the bag-of-words model uses the entire documents as context and uses (weighted) counts to capture the co-occurrence of words rather than predictive vectors.

### The word2vec Model: scalable word and phrase embeddings

A word2vec model is a two-layer neural net that takes a text corpus as input and outputs a set of embedding vectors for words in that corpus. There are two different architectures to efficiently learn word vectors using shallow neural networks.
- The **continuous-bag-of-words** (CBOW) model predicts the target word using the average of the context word vectors as input so that their order does not matter. CBOW trains faster and tends to be slightly more accurate for frequent terms, but pays less attention to infrequent words.
- The **skip-gram** (SG) model, in contrast, uses the target word to predict words sampled from the context. It works well with small datasets and finds good representations even for rare words or phrases.

### Evaluating embeddings: vector arithmetic and analogies

The dimensions of the word and phrase vectors do not have an explicit meaning. However, the embeddings encode similar usage as proximity in the latent space in a way that carries over to semantic relationships. This results in the interesting properties that analogies can be expressed by adding and subtracting word vectors.

Just as words can be used in different contexts, they can be related to other words in different ways, and these relationships correspond to different directions in the latent space. Accordingly, there are several types of analogies that the embeddings should reflect if the training data permits.

The word2vec authors provide a list of several thousand relationships spanning aspects of geography, grammar and syntax, and family relationships to evaluate the quality of embedding vectors (see directory [analogies](data/analogies)).

## Code example: Working with embedding models

Similar to other unsupervised learning techniques, the goal of learning embedding vectors is to generate features for other tasks like text classification or sentiment analysis.
There are several options to obtain embedding vectors for a given corpus of documents:
- Use embeddings learned from a generic large corpus like Wikipedia or Google News
- Train your own model using documents that reflect a domain of interest

### Working with Global Vectors for Word Representation (GloVe)

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

There are several sources for pre-trained word embeddings. Popular options include Stanford’s GloVE and spaCy’s built-in vectors.
- The notebook [using_trained_vectors ](01_using_trained_vectors.ipynb) illustrates how to work with pretrained vectors.

### Evaluating embeddings using analogies

The notebook [evaluating_embeddings](02_evaluating_embeddings.ipynb) demonstrates how to test the quality of word vectors using analogies and other semantic relationships among words.

## Code example: training domain-specific embeddings using financial news

Many tasks require embeddings of domain-specific vocabulary that models pre-trained on a generic corpus may not be able to capture. Standard word2vec models are not able to assign vectors to out-of-vocabulary words and instead use a default vector that reduces their predictive value. 

For example, when working with industry-specific documents, the vocabulary or its usage may change over time as new technologies or products emerge. As a result, the embeddings need to evolve as well. In addition, documents like corporate earnings releases use nuanced language that GloVe vectors pre-trained on Wikipedia articles are unlikely to properly reflect.

See the [data](../data) directory for instructions on sourcing the financial news dataset.

### Preprocessing financial news: sentence detection and n-grams

The notebook [financial_news_preprocessing](03_financial_news_preprocessing.ipynb) demonstrates how to prepare the source data for our model

### Skip-gram architecture in TensorFlow 2 and visualization with TensorBoard

The notebook [financal_news_word2vec_tensorflow](04_financal_news_word2vec_tensorflow.ipynb) illustrates how to build a word2vec model using the Keras interface of TensorFlow 2 that we will introduce in much more detail in the next chapter. 

### How to train embeddings faster with Gensim

The TensorFlow implementation is very transparent in terms of its architecture, but it is not particularly fast. The natural language processing (NLP) library [gensim](https://radimrehurek.com/gensim/) that we also used for topic modeling in the last chapter, offers better performance and more closely resembles the C-based word2vec implementation provided by the original authors.

The notebook [inancial_news_word2vec_gensim](05_financial_news_word2vec_gensim.ipynb) shows how to learn word vectors more efficiently.

## Code Example: word Vectors from SEC Filings using gensim

In this section, we will learn word and phrase vectors from annual SEC filings using gensim to illustrate the potential value of word embeddings for algorithmic trading. In the following sections, we will combine these vectors as features with price returns to train neural networks to predict equity prices from the content of security filings.

In particular, we use a dataset containing over 22,000 10-K annual reports from the period 2013-2016 that are filed by listed companies and contain both financial information and management commentary (see Chapter 3 on [Alternative Data](../03_alternative_data)). For about half of 11K filings for companies that we have stock prices to label the data for predictive modeling (see references about data source and the notebooks in the folder [sec-filings](sec-filings) for details). 

- [2013-2016 Cleaned/Parsed 10-K Filings with the SEC](https://data.world/jumpyaf/2013-2016-cleaned-parsed-10-k-filings-with-the-sec)
- [Stock Market Predictions with Natural Language Deep Learning](https://www.microsoft.com/developerblog/2017/12/04/predicting-stock-performance-deep-learning/)

### Preprocessing: content selection, sentence detection, and n-grams

The notebook [sec_preprocessing](06_sec_preprocessing.ipynb) shows how to parse and tokenize the text using spaCy, similar to the approach in Chapter 14, [Text Data for Trading: Sentiment Analysis](../14_working_with_text_data). 

### Model training and evaluation

The notebook [sec_word2vec](07_sec_word2vec.ipynb) uses gensim's [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) implementation of the skip-gram architecture to learn word vectors for the SEC filings dataset.

## Code example: sentiment Analysis with Doc2Vec

Text classification requires combining multiple word embeddings. A common approach is to average the embedding vectors for each word in the document. This uses information from all embeddings and effectively uses vector addition to arrive at a different location point in the embedding space. However, relevant information about the order of words is lost. 

In contrast, the state-of-the-art generation of embeddings for pieces of text like a paragraph or a product review is to use the document embedding model doc2vec. This model was developed by the word2vec authors shortly after publishing their original contribution. Similar to word2vec, there are also two flavors of doc2vec:
- The distributed bag of words (DBOW) model corresponds to the Word2Vec CBOW model. The document vectors result from training a network on the synthetic task of predicting a target word based on both the context word vectors and the document's doc vector.
- The distributed memory (DM) model corresponds to the word2wec skipgram architecture. The doc vectors result from training a neural net to predict a target word using the full document’s doc vector.

The notebook [doc2vec_yelp_sentiment](08_doc2vec_yelp_sentiment.ipynb) applies doc2vec to a random sample of 1mn Yelp reviews with their associated star ratings.

## New Frontiers: Attention, Transformers, and Pretraining

Word2vec and GloVe embeddings capture more semantic information than the bag-of-words approach, but only allow for a single fixed-length representation of each token that does not differentiate between context-specific usages. To address unsolved problems like multiple meanings for the same word, called polysemy, several new models have emerged that build on the attention mechanism designed to learn more contextualized word embeddings ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)). Key characteristics of these models are 
- the use of bidirectional language models that process text both left-to-right and right-to-left for a richer context representation, and
- the use of semi-supervised pretraining on a large generic corpus to learn universal language aspects in the form of embeddings and network weights that can be used end fine-tuned for specific tasks

### Attention is all you need: transforming natural language generation

In 2018, Google released the BERT model, which stands for Bidirectional Encoder Representations from Transformers ([Devlin et al. 2019](https://arxiv.org/abs/1810.04805)). In a major breakthrough for NLP research, it achieved groundbreaking results on eleven natural language understanding tasks ranging from question answering and named entity recognition to paraphrasing and sentiment analysis as measured by the General Language Understanding Evaluation (GLUE) [benchmark](https://gluebenchmark.com/).

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

### BERT: Towards a more universal, pretrained language model

The BERT model builds on two key ideas, namely the transformer architecture described in the previous section and unsupervised pre-training so that it doesn’t need to be trained from scratch for each new task; rather, its weights are fine-tuned.
- BERT takes the attention mechanism to a new (deeper) level by using 12 or 24 layers depending on the architecture, each with 12 or 16 attention heads, resulting in up to 24 x 16 = 384 attention mechanisms to learn context-specific embeddings.  
- BERT uses unsupervised, bidirectional pre-training to learn its weights in advance on two tasks: masked language modeling (predicting a missing word given the left and right context) and next sentence prediction (predicting whether one sentence follows another).

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/)
- [The General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/leaderboard)
- [Financial NLP at S&P Global ](https://www.youtube.com/watch?v=rdmaR4WRYEM&list=PLBmcuObd5An4UC6jvK_-eSl6jCvP1gwXc&index=9)

### Using pretrained state-of-the-art models

- [Huggingface Transformers](https://github.com/huggingface/transformers)
    - Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, T5, CTRL...) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over thousands of pretrained models in 100+ languages and deep interoperability between PyTorch & TensorFlow 2.0.
- [spacy-transformers](https://github.com/explosion/spacy-transformers)
    - This package (previously spacy-pytorch-transformers) provides spaCy model pipelines that wrap Hugging Face's transformers package, so you can use them in spaCy. The result is convenient access to state-of-the-art transformer architectures, such as BERT, GPT-2, XLNet, etc. For more details and background.
- [Allen NLP](https://allennlp.org/)
    - Deep learning for NLP: AllenNLP makes it easy to design and evaluate new deep learning models for nearly any NLP problem, along with the infrastructure to easily run them in the cloud or on your laptop.
    - State of the art models: AllenNLP includes reference implementations of high quality models for both core NLP problems (e.g. semantic role labeling) and NLP applications (e.g. textual entailment).
- [Sentence Transformers: Multilingual Sentence Embeddings using BERT / RoBERTa / XLM-RoBERTa & Co. with PyTorch]
     -BERT / RoBERTa / XLM-RoBERTa produces out-of-the-box rather bad sentence embeddings. This repository fine-tunes BERT / RoBERTa / DistilBERT / ALBERT / XLNet with a siamese or triplet network structure to produce semantically meaningful sentence embeddings that can be used in unsupervised scenarios: Semantic textual similarity via cosine-similarity, clustering, semantic search.

## Additional Resources

- [GloVe: Global Vectors for Word Representation](https://github.com/stanfordnlp/GloVe)
- [Common Crawl Data](http://commoncrawl.org/the-data/)
- [word2vec analogy samples](https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt)
- [spaCy word vectors and semantic similarity](https://spacy.io/usage/vectors-similarity)
- [2013-2016 Cleaned/Parsed 10-K Filings with the SEC](https://data.world/jumpyaf/2013-2016-cleaned-parsed-10-k-filings-with-the-sec)
- [Stanford Sentiment Tree Bank](https://nlp.stanford.edu/sentiment/treebank.html)
- [Word embeddings | TensorFlow Core](https://www.tensorflow.org/tutorials/text/word_embeddings)
- [Visualizing Data using the Embedding Projector in TensorBoard](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin)
