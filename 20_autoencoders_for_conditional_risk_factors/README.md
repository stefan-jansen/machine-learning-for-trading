# Autoencoders for Conditional Risk Factors and Asset Pricing

This chapter shows how unsupervised learning can leverage deep learning for trading. More specifically, we’ll discuss autoencoders that have been around for decades but recently attracted fresh interest.

An autoencoder is a neural network trained to reproduce the input while learning a new representation of the data, encoded by the parameters of a hidden layer. 
Autoencoders have long been used for nonlinear dimensionality reduction and manifold learning (see [Chapter 13](../13_unsupervised_learning)). 
A variety of designs leverage the feedforward, convolutional, and recurrent network architectures we covered in the last three chapters. 

We will also see how autoencoders can underpin a trading strategy by building a deep neural network that uses an [autoencoder to extract risk factors](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) and predict equity returns, conditioned on a range of equity attributes (Gu, Kelly, and Xiu 2020).

## Content

1. [Autoencoders for nonlinear feature extraction](#autoencoders-for-nonlinear-feature-extraction)
    * [Code example: Generalizing PCA with nonlinear dimensionality reduction](#code-example-generalizing-pca-with-nonlinear-dimensionality-reduction)
    * [Code example: convolutional autoencoders to compress and denoise images](#code-example-convolutional-autoencoders-to-compress-and-denoise-images)
    * [Seq2seq autoencoders to extract time-series features for trading](#seq2seq-autoencoders-to-extract-time-series-features-for-trading)
    * [Code example: Variational autoencoders - learning how to generate the input data](#code-example-variational-autoencoders---learning-how-to-generate-the-input-data)
2. [Code example: A conditional autoencoder for return forecasts and trading](#code-example-a-conditional-autoencoder-for-return-forecasts-and-trading)
    * [Creating a new dataset with stock price and metadata information](#creating-a-new-dataset-with-stock-price-and-metadata-information)
    * [Computing predictive asset characteristics](#computing-predictive-asset-characteristics)
    * [Creating and training the conditional autoencoder architecture](#creating-and-training-the-conditional-autoencoder-architecture)
    * [Evaluating the results](#evaluating-the-results)

## Autoencoders for nonlinear feature extraction

In Chapter 17, [Deep Learning for Trading](../17_deep_learning), we saw how neural networks succeed at supervised learning by extracting a hierarchical feature representation useful for the given task. Convolutional neural networks, e.g., learn and synthesize increasingly complex patterns from grid-like data, for example, to identify or detect objects in an image or to classify time series. 
An autoencoder, in contrast, is a neural network designed exclusively to learn a new representation that encodes the input in a way that helps solve another task. To this end, the training forces the network to reproduce the input. Since autoencoders typically use the same data as input and output, they are also considered an instance of self-supervised learning. 
In the process, the parameters of a hidden layer h become the code that represents the input, similar to the word2vec model covered in [Chapter 16](../16_word_embeddings). 

For a good overview, see Chapter 14 in Deep Learning:
- [Autoencoders](http://www.deeplearningbook.org/contents/autoencoders.html), Ian Goodfellow, Yoshua Bengio and Aaron Courville, Deep Learning Book, MIT Press 2016

The TensorFlow's Keras interfacte makes it fairly straightforward to build various types of autoencoders and the following examples are adapted from Keras' tutorials.

- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)

### Code example: Generalizing PCA with nonlinear dimensionality reduction

A traditional use case includes dimensionality reduction, achieved by limiting the size of the hidden layer so that it performs lossy compression. Such an autoencoder is called undercomplete and the purpose is to force it to learn the most salient properties of the data by minimizing a loss function. In addition to feedforward architectures, autoencoders can also use convolutional layers to learn hierarchical feature representations.

The notebook [deep_autoencoders](01_deep_autoencoders.ipynb) illustrates how to implement several of autoencoder models using TensorFlow, including autoencoders using deep feedforward nets and sparsity constraints. 
 
### Code example: convolutional autoencoders to compress and denoise images

As discussed in Chapter 18, [CNNs: Time Series as Images and Satellite Image Classification](../18_convolutional_neural_nets), fully-connected feedforward architectures are not well suited to capture local correlations typical to data with a grid-like structure. Instead, autoencoders can also use convolutional layers to learn a hierarchical feature representation. Convolutional autoencoders leverage convolutions and parameter sharing to learn hierarchical patterns and features irrespective of their location, translation, or changes in size.

The notebook [convolutional_denoising_autoencoders](02_convolutional_denoising_autoencoders.ipynb) goes on to demonstrate how to implement convolutional and denoising autencoders to recover corrupted image inputs.

### Seq2seq autoencoders to extract time-series features for trading

Sequence-to-sequence autoencoders are based on RNN components, such as Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRUs). They learn a compressed representation of sequential data and have been applied to video, text, audio, and time-series data.

- [A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html), Francois Chollet, September 2017
- [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681), Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov, 2016
- [Gradient Trader Part 1: The Surprising Usefulness of Autoencoders](https://rickyhan.com/jekyll/update/2017/09/14/autoencoders.html)
    - [Code examples](https://github.com/0b01/recurrent-autoencoder)
- [Deep Learning Financial Market Data](http://wp.doc.ic.ac.uk/hipeds/wp-content/uploads/sites/78/2017/01/Steven_Hutt_Deep_Networks_Financial.pdf)
    - Motivation: Regulators identify prohibited patterns of trading activity detrimental to orderly markets. Financial Exchanges are responsible for maintaining orderly markets. (e.g. Flash Crash and Hound of Hounslow.)
    - Challenge: Identify prohibited trading patterns quickly and efficiently.
    - **Goal**: Build a trading pattern search function using Deep Learning. Given a sample trading pattern identify similar patterns in historical LOB data.

### Code example: Variational autoencoders - learning how to generate the input data

Variational Autoencoders (VAE) are more recent developments focused on generative modeling. More specifically, VAEs are designed to learn a latent variable model for the input data. Note that we encountered latent variables in Chapter 14, Topic Modeling.

Hence, VAEs do not let the network learn arbitrary functions as long as it faithfully reproduces the input. Instead, they aim to learn the parameters of a probability distribution that generates the input data. In other words, VAEs are generative models because, if successful, you can generate new data points by sampling from the distribution learned by the VAE.

The notebook [variational_autoencoder](03_variational_autoencoder.ipynb) shows how to build a Variational Autoencoder using Keras.

- [Auto-encoding variational bayes](https://arxiv.org/abs/1312.6114), Diederik P Kingma, Max Welling, 2014
- [Tutorial: What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
- [Variational Autoencoder / Deep Latent Gaussian Model in tensorflow and pytorch](https://github.com/altosaar/variational-autoencoder)

## Code example: A conditional autoencoder for return forecasts and trading

Recent research by [Gu, Kelly, and Xiu](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) develops an asset pricing model based on the exposure of securities to risk factors. It builds on the concept of data-driven risk factors that we discussed in Chapter 13 when introducing PCA as well as the risk factor models covered in Chapter 4, Financial Feature Engineering: How to Research Alpha Factors. 
The authors aim to show that the asset characteristics used by factor models to capture the systematic drivers of ‘anomalies’ are just proxies for the time-varying exposure to risk factors that cannot be directly measured. 
In this context, anomalies are returns in excess of those explained by the exposure to aggregate market risk (see the discussion of the capital asset pricing model in [Chapter 5](../05_strategy_evaluation)).

### Creating a new dataset with stock price and metadata information

The reference implementation uses stock price and firm characteristic data for over 30,000 US equities from the Center for Research in Security Prices (CRSP) from 1957-2016 at monthly frequency. It computes 94 metrics that include a broad range of asset attributes suggested as predictive of returns in previous academic research and listed in Green, Hand, and Zhang (2017), who set out to verify these claims.
Since we do not have access to the high-quality but costly CRSP data, we leverage [yfinance](https://github.com/ranaroussi/yfinance) (see Chapter 2, [Market and Fundamental Data: Sources and Techniques](../02_market_and_fundamental_data)) to download price and metadata from Yahoo Finance. There are downsides to choosing free data, including:
- the lack of quality control regarding adjustments, 
- survivorship bias because we cannot get data for stocks that are no longer listed, and
- a smaller scope in terms of both the number of equities and the length of their history. 

The notebook [build_us_stock_dataset](04_build_us_stock_dataset.ipynb) contains the relevant code examples for this section.

### Computing predictive asset characteristics

The authors test 94 asset attributes and identify the 20 most influential metrics while asserting that feature importance drops off quickly thereafter. The top 20 stock characteristics fall into three categories, namely:
- Price trend, including (industry) momentum, short- and long-term reversal, or the recent maximum return
- Liquidity such as turnover, dollar volume, or market capitalization
- Risk measures, for instance, total and idiosyncratic return volatility or market beta

Of these 20, we limit the analysis to 16 for which we have or can approximate the relevant inputs. The notebook [conditional_autoencoder_for_trading_data](05_conditional_autoencoder_for_trading_data.ipynb) demonstrates how to calculate the relevant metrics.

### Creating and training the conditional autoencoder architecture

The conditional autoencoder proposed by the authors allows for time-varying return distributions that take into account changing asset characteristics. 
To this end, they extend standard autoencoder architectures that we discussed in the first section of this chapter to allow for features to shape the encoding.

The notebook [conditional_autoencoder_for_asset_pricing_model](06_conditional_autoencoder_for_asset_pricing_model.ipynb) demonstrates how to create and train this architecture.

### Evaluating the results

The notebook [alphalens_analysis](07_alphalens_analysis.ipynb) measures the financial performance of the model's prediction.


