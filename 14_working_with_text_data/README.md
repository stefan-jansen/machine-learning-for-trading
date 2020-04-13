# Text Data for Trading: Sentiment Analysis

This is the first of three chapters dedicated to extracting signals for algorithmic trading strategies from text data using natural language processing (NLP) and machine learning.

Text data is very rich in content but highly unstructured so that it requires more preprocessing to enable an ML algorithm to extract relevant information. A key challenge consists of converting text into a numerical format without losing its meaning. We will cover several techniques capable of capturing nuances of language so that they can be used as input for ML algorithms.

In this chapter, we will introduce fundamental feature extraction techniques that focus on individual semantic units, i.e. words or short groups of words called tokens. We will show how to represent documents as vectors of token counts by creating a document-term matrix and then proceed to use it as input for news classification and sentiment analysis. We will also introduce the Naive Bayes algorithm that is popular for this purpose.

In the following two chapters, we build on these techniques and use ML algorithms like topic modeling and word-vector embeddings to capture the information contained in a broader context. 

In particular, in this chapter we will cover:
- What the fundamental NLP workflow looks like
- How to build a multilingual feature extraction pipeline using spaCy and Textblob
- How to perform NLP tasks like part-of-speech tagging or named entity recognition
- How to convert tokens to numbers using the document-term matrix
- How to classify text using the Naive Bayes model
- How to perform sentiment analysis


## How to extract features from text data
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf), Daniel Jurafsky & James H. Martin, 3rd edition, draft, 2018
- [Statistical natural language processing and corpus-based computational linguistics](https://nlp.stanford.edu/links/statnlp.html), Annotated list of resources, Stanford University
- [NLP Data Sources](https://github.com/niderhoff/nlp-datasets)

### Challenges of Natural Language Processing

The conversion of unstructured text into a machine-readable format requires careful preprocessing to preserve the valuable semantic aspects of the data. How humans derive meaning from and comprehend the content of language is not fully understood and improving language understanding by machines remains an area of very active research. 

NLP is challenging because the effective use of text data for machine learning requires an understanding of the inner workings of language as well as knowledge about the world to which it refers. Key challenges include:
- ambiguity due to polysemy, i.e. a word or phrase can have different meanings that depend on context (‘Local High School Dropouts Cut in Half’)
- non-standard and evolving use of language, especially in social media
- idioms: ‘throw in the towel’
- entity names can be tricky : ‘Where is A Bug's Life playing?’
- the need for knowledge about the world: ‘Mary and Sue are sisters’ vs ‘Mary and Sue are mothers’

### Use Cases

Key NLP use cases include:

| Use Case  | Description  | Examples  |
|---|---|---|
| Chatbots | Understand natural language from the user and return intelligent responses | [Api.ai](https://api.ai/) |
| Information retrieval | Find relevant results and similar results | [Google](https://www.google.com/) |
| Information extraction | Structured information from unstructured documents | [Events from Gmail](https://support.google.com/calendar/answer/6084018?hl=en) |
| Machine translation | One language to another | [Google Translate](https://translate.google.com/) |
| Text simplification | Preserve the meaning of text, but simplify the grammar and vocabulary | [Rewordify](https://rewordify.com/), [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) |
| Predictive text input | Faster or easier typing  | [Phrase completion](https://justmarkham.shinyapps.io/textprediction/), [A much better application](https://farsite.shinyapps.io/swiftkey-cap/) |
| Sentiment analysis | Attitude of speaker | [Hater News](https://medium.com/@KevinMcAlear/building-hater-news-62062c58325c) |
| Automatic summarization | Extractive or abstractive summarization | [reddit's autotldr algo](https://smmry.com/about), [autotldr example](https://www.reddit.com/r/technology/comments/35brc8/21_million_people_still_use_aol_dialup/cr2zzj0)  |
| Natural language generation | Generate text from data | [How a computer describes a sports match](http://www.bbc.com/news/technology-34204052), [Publishers withdraw more than 120 gibberish papers](http://www.nature.com/news/publishers-withdraw-more-than-120-gibberish-papers-1.14763) |
| Speech recognition and generation | Speech-to-text, text-to-speech | [Google's Web Speech API demo](https://www.google.com/intl/en/chrome/demos/speech.html), [Vocalware Text-to-Speech demo](https://www.vocalware.com/index/demo) |
| Question answering | Determine the intent of the question, match query with knowledge base, evaluate hypotheses | [How did Watson beat Jeopardy champion Ken Jennings?](http://blog.ted.com/how-did-supercomputer-watson-beat-jeopardy-champion-ken-jennings-experts-discuss/), [Watson Trivia Challenge](http://www.nytimes.com/interactive/2010/06/16/magazine/watson-trivia-game.html), [The AI Behind Watson](http://www.aaai.org/Magazine/Watson/watson.php)

## From text to tokens – the NLP pipeline

The following table summarizes the key tasks of an NLP pipeline:


| Feature                     | Description                                                       |
|-----------------------------|-------------------------------------------------------------------|
| Tokenization                | Segment text into words, punctuations marks etc.                  |
| Part-of-speech tagging      | Assign word types to tokens, like a verb or noun.                 |
| Dependency parsing          | Label syntactic token dependencies, like subject <=> object.      |
| Stemming & Lemmatization    | Assign the base forms of words: "was" => "be", "rats" => "rat".   |
| Sentence boundary detection | Find and segment individual sentences.                            |
| Named Entity Recognition    | Label "real-world" objects, like persons, companies or locations. |
| Similarity                  | Evaluate similarity of words, text spans, and documents.          |


### NLP pipeline with spaCy and textacy

The notebook [nlp_pipeline_with_spaCy](01_nlp_pipeline_with_spaCy.ipynb) demonstrates how to construct an NLP pipeline using the open-source python library [spaCy]((https://spacy.io/)). The [textacy](https://chartbeat-labs.github.io/textacy/index.html) library builds on spaCy and provides easy access to spaCy attributes and additional functionality.

- spaCy [docs](https://spacy.io/) and installation [instructions](https://spacy.io/usage/#installation)
- textacy relies on `spaCy` to solve additional NLP tasks - see [documentation](https://chartbeat-labs.github.io/textacy/index.html)

#### Code Examples

The code for this section is in the notebook `nlp_pipeline_with_spaCy`

#### Data
- [BBC Articles](http://mlg.ucd.ie/datasets/bbc.html), use raw text files
- [TED2013](http://opus.nlpl.eu/TED2013.php), a parallel corpus of TED talk subtitles in 15 langugages

### NLP with TextBlob

The `TextBlob` library provides a simplified interface for common NLP tasks including part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and others.

The notebook [nlp_with_textblob](02_nlp_with_textblob.ipynb) illustrates its functionality.

- [Documentation](https://textblob.readthedocs.io/en/dev/)
- [Sentiment Analysis](https://github.com/sloria/TextBlob/blob/dev/textblob/en/en-sentiment.xml)

A good alternative is NLTK, a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

- Natural Language ToolKit (NLTK) [Documentation](http://www.nltk.org/)

## From tokens to numbers – the document-term matrix

This section introduces the bag-of-words model that converts text data into a numeric vector space representation that permits the comparison of documents using their distance. We demonstrate how to create a document-term matrix using the sklearn library.

- [TF-IDF is about what matters](https://planspace.org/20150524-tfidf_is_about_what_matters/)

### Document-term matrix with sklearn

The scikit-learn preprocessing module offers two tools to create a document-term matrix. 
1. The [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) uses binary or absolute counts to measure the term frequency tf(d, t) for each document d and token t.
2. The [TfIDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), in contrast, weighs the (absolute) term frequency by the inverse document frequency (idf). As a result, a term that appears in more documents will receive a lower weight than a token with the same frequency for a given document but lower frequency across all documents

The notebook [document_term_matrix](03_document_term_matrix.ipynb) demonstrate usage and configuration.

## Text classification and sentiment analysis

Once text data has been converted into numerical features using the natural language processing techniques discussed in the previous sections, text classification works just like any other classification task.

In this section, we will apply these preprocessing technique to news articles, product reviews, and Twitter data and teach various classifiers to predict discrete news categories, review scores, and sentiment polarity.

First, we will introduce the Naive Bayes model, a probabilistic classification algorithm that works well with the text features produced by a bag-of-words model.

- [Daily Market News Sentiment and Stock Prices](https://www.econstor.eu/handle/10419/125094), David E. Allen & Michael McAleer & Abhay K. Singh, 2015, Tinbergen Institute Discussion Paper
- [Predicting Economic Indicators from Web Text Using Sentiment Composition](http://www.ijcce.org/index.php?m=content&c=index&a=show&catid=39&id=358), Abby Levenberg, et al, 2014
- [JP Morgan NLP research results](https://www.jpmorgan.com/global/research/machine-learning)

### The Naive Bayes classifier

The Naive Bayes algorithm is very popular for text classification because low computational cost and memory requirements facilitate training on very large, high-dimensional datasets. Its predictive performance can compete with more complex models, provides a good baseline, and is best known for successful spam detection.

The model relies on Bayes theorem and the assumption that the various features are independent of each other given the outcome class. In other words, for a given outcome, knowing the value of one feature (e.g. the presence of a token in a document) does not provide any information about the value of another feature.


### News article classification

We start with an illustration of the Naive Bayes model to classify 2,225 BBC news articles that we know belong to five different categories.

The notebook [text_classification](04_text_classification.ipynb) contains the relevant examples.

### Sentiment Analysis

Sentiment analysis is one of the most popular uses of natural language processing and machine learning for trading because positive or negative perspectives on assets or other price drivers are likely to impact returns. 

Generally, modeling approaches to sentiment analysis rely on dictionaries as the TextBlob library or models trained on outcomes for a specific domain. The latter is preferable because it permits more targeted labeling, e.g. by tying text features to subsequent price changes rather than indirect sentiment scores.

#### Twitter Dataset

We illustrate machine learning for sentiment analysis using a [Twitter dataset](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) with binary polarity labels, and a large Yelp business review dataset with a five-point outcome scale.

The notebook [sentiment_analysis_twitter](05_sentiment_analysis_twitter.ipynb) contains the relevant example.

- [Cheng-Caverlee-Lee September 2009 - January 2010 Twitter Scrape](https://archive.org/details/twitter_cikm_2010)

#### Yelp Dataset

To illustrate text processing and classification at larger scale, we also use the [Yelp Dataset](https://www.yelp.com/dataset).

The notebook [sentiment_analysis_yelp](06_sentiment_analysis_yelp.ipynb) contains the relevant example.

- [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge)

