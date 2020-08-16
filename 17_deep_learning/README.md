# Deep Learning for Trading

This chapter kicks off part four, which covers how several deep learning (DL) modeling techniques can be useful for investment and trading. DL has achieved numerous breakthroughs in many domains ranging from image and speech recognition to robotics and intelligent agents that have drawn widespread attention and revived large-scale research into Artificial Intelligence (AI). The expectations are high that the rapid development will continue and many more solutions to difficult practical problems will emerge.

In this chapter, we will present feedforward neural networks to introduce key elements of working with neural networks relevant to the various DL architectures covered in the following chapters. More specifically, we will demonstrate how to train large models efficiently using the backpropagation algorithm and manage the risks of overfitting. We will also show how to use the popular Keras, TensorFlow 2.0, and PyTorch frameworks, which we will leverage throughout part four.

In the following chapters, we will build on this foundation to design various architectures suitable for different investment applications with a particular focus on alternative text and image data. These include recurrent neural networks (RNNs) tailored to sequential data such as time series or natural language, and Convolutional Neural Networks (CNNs), which are particularly well suited to image data but can also be used with time-series data. We will also cover deep unsupervised learning, including autoencoders and Generative Adversarial Networks (GANs) as well as reinforcement learning to train agents that interactively learn from their environment.

## Content

1. [Deep learning: How it differs and why it matters](#deep-learning-how-it-differs-and-why-it-matters)
    * [How hierarchical features help tame high-dimensional data](#how-hierarchical-features-help-tame-high-dimensional-data)
    * [Automating feature extraction: DL as representation learning](#automating-feature-extraction-dl-as-representation-learning)
    * [How DL relates to machine learning and artificial intelligence](#how-dl-relates-to-machine-learning-and-artificial-intelligence)
2. [Code example: Designing a neural network](#code-example-designing-a-neural-network)
    * [Key design choices](#key-design-choices)
    * [How to regularize deep neural networks](#how-to-regularize-deep-neural-networks)
    * [Training faster: Optimizations for deep learning](#training-faster-optimizations-for-deep-learning)
3. [Popular Deep Learning libraries](#popular-deep-learning-libraries)
    * [How to Leverage GPU Optimization](#how-to-leverage-gpu-optimization)
    * [How to use Tensorboard](#how-to-use-tensorboard)
    * [Code example: how to use PyTorch](#code-example-how-to-use-pytorch)
    * [Code example: How to use TensorFlow](#code-example-how-to-use-tensorflow)
4. [Code example: Optimizing a neural network for a long-short trading strategy](#code-example-optimizing-a-neural-network-for-a-long-short-trading-strategy)
    * [Optimizing the NN architecture](#optimizing-the-nn-architecture)
    * [Backtesting a long-short strategy based on ensembled signals](#backtesting-a-long-short-strategy-based-on-ensembled-signals)


## Deep learning: How it differs and why it matters

The machine learning (ML) algorithms covered in Part 2 work well on a wide variety of important problems, including on text data as demonstrated in Part 3. They have been less successful, however, in solving central AI problems such as recognizing speech or classifying objects in images. These limitations have motivated the development of DL, and the recent DL breakthroughs have greatly contributed to a resurgence of interest in AI. F

or a comprehensive introduction that includes and expands on many of the points in this section, see Goodfellow, Bengio, and Courville (2016), or for a much shorter version, see LeCun, Bengio, and Hinton (2015).

- [Deep Learning](https://www.deeplearningbook.org/), Ian Goodfellow, Yoshua Bengio and Aaron Courville, MIT Press, 2016
- [Deep learning](https://www.nature.com/articles/nature14539), Yann LeCun, Yoshua Bengio, and Geoffrey Hinton, Nature 2015
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), Michael A. Nielsen, Determination Press, 2015
- [The Quest for Artificial Intelligence - A History of Ideas and Achievements](https://ai.stanford.edu/~nilsson/QAI/qai.pdf), Nils J. Nilsson, Cambridge University Press, 2010
- [One Hundred Year Study on Artificial Intelligence (AI100)](https://ai100.stanford.edu/)
- [TensorFlow Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.71056&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false), Interactive, browser-based Deep Learning platform

### How hierarchical features help tame high-dimensional data

As discussed throughout Part 2, the key challenge of supervised learning is to generalize from training data to new samples. Generalization becomes exponentially more difficult as the dimensionality of the data increases. We encountered the root causes of these difficulties as the curse of dimensionality in Chapter 13, [Unsupervised Learning: From Data-Driven Risk Factors to Hierarchical Risk Parity](../13_unsupervised_learning).

### Automating feature extraction: DL as representation learning

Many AI tasks like image or speech recognition require knowledge about the world. One of the key challenges is to encode this knowledge so a computer can utilize it. For decades, the development of ML systems required considerable domain expertise to transform the raw data (such as image pixels) into an internal representation that a learning algorithm could use to detect or classify patterns.

### How DL relates to machine learning and artificial intelligence

AI has a long history, going back at least to the 1950s as an academic field and much longer as a subject of human inquiry, but has experienced several waves of ebbing and flowing enthusiasm since (see [The Quest for Artificial Intelligence](https://ai.stanford.edu/~nilsson/QAI/qai.pdf), Nilsson, 2009 for an in-depth survey). 
- ML is an important subfield with a long history in related disciplines such as statistics and became prominent in the 1980s. 
- DL is a form of representation learning and itself a subfield of ML.

## Code example: Designing a neural network

To gain a better understanding of how NN work, the notebook [01_build_and_train_feedforward_nn](build_and_train_feedforward_nn.ipynb) formulates as simple feedforward architecture and forward propagation computations using matrix algebra and implements it using Numpy, the Python counterpart of linear algebra.

<p align="center">
<img src="https://i.imgur.com/UKCr9zi.png" width="85%">
</p>

### Key design choices

Some NN design choices resemble those for other supervised learning models. For example, the output is dictated by the type of the ML problem such as regression, classification, or ranking. Given the output, we need to select a cost function to measure prediction success and failure, and an algorithm that optimizes the network parameters to minimize the cost. 

NN-specific choices include the numbers of layers and nodes per layer, the connections between nodes of different layers, and the type of activation functions.

### How to regularize deep neural networks

The downside of the capacity of NN to approximate arbitrary functions is the greatly increased risk of overfitting. The best protection against overfitting is to train the model on a larger dataset. Data augmentation, e.g. by creating slightly modified versions of images, is a powerful alternative approach. The generation of synthetic financial training data for this purpose is an active research area that we will address in [Chapter 21](../21_gans_for_synthetic_time_series)

### Training faster: Optimizations for deep learning

Backprop refers to the computation of the gradient of the cost function with respect to the internal parameter we wish to update and the use of this information to update the parameter values. The gradient is useful because it indicates the direction of parameter change that causes the maximal increase in the cost function. Hence, adjusting the parameters according to the negative gradient produces an optimal cost reduction, at least for a region very close to the observed samples. See Ruder (2016) for an excellent overview of key gradient descent optimization algorithms.

- [Gradient Checking & Advanced Optimization](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization), Unsupervised Feature Learning and Deep Learning, Stanford University
- [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#momentum), Sebastian Ruder, 2016

## Popular Deep Learning libraries

Currently, the most popular DL libraries are [TensorFlow](https://www.tensorflow.org/) (supported by Google) and [PyTorch](https://pytorch.org/) (supported by Facebook). 

Development is very active with PyTorch at version 1.4 and TensorFlow at 2.2 as of March 2020. TensorFlow 2.0 adopted [Keras](https://keras.io/) as its main interface, effectively combining both libraries into one.
Additional options include:

- [Microsoft Cognitive Toolkit (CNTK)](https://github.com/Microsoft/CNTK)
- [Caffe](http://caffe.berkeleyvision.org/)
- [Thenao](http://www.deeplearning.net/software/theano/), developed at University of Montreal since 2007
- [Apache MXNet](https://mxnet.apache.org/), used by Amazon
- [Chainer](https://chainer.org/), developed by the Japanese company Preferred Networks
- [Torch](http://torch.ch/), uses Lua, basis for PyTorch
- [Deeplearning4J](https://deeplearning4j.org/), uses Java

### How to Leverage GPU Optimization

All popular Deep Learning libraries support the use of GPU, and some also allow for parallel training on multiple GPU. The most common types of GPU are produced by NVIDA, and configuration requires installation and setup of the CUDA environment. The process continues to evolve and can be somewhat challenging depending on your computational environment. 

A more straightforward way to leverage GPU is via the the Docker virtualization platform. There are numerous images available that you can run in local container managed by Docker that circumvents many of the driver and version conflicts that you may otherwise encounter. Tensorflow provides docker images on its website that can also be used with Keras. 

- [Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](http://timdettmers.com/2018/11/05/which-gpu-for-deep-learning/), Tim Dettmers
- [A Full Hardware Guide to Deep Learning](http://timdettmers.com/2018/12/16/deep-learning-hardware-guide/), Tim Dettmers

### How to use Tensorboard

Tensorboard is a great visualization tool that comes with TensorFlow. It includes a suite of visualization tools to simplify the understanding, debugging, and optimization of neural networks.

You can use it to visualize the computational graph, plot various execution and performance metrics, and even visualize image data processed by the network. It also permits comparisons of different training runs.
When you run the how_to_use_keras notebook, and with TensorFlow installed, you can launch Tensorboard from the command line:

```python
tensorboard --logdir=/full_path_to_your_logs ## e.g. ./tensorboard
```
- [TensorBoard: Visualizing Learning](https://www.tensorflow.org/guide/summaries_and_tensorboard)

### Code example: how to use PyTorch

Pytorch has been developed at the Facebook AI Research group led by Yann LeCunn and the first alpha version released in September 2016. It provides deep integration with Python libraries like Numpy that can be used to extend its functionality, strong GPU acceleration, and automatic differentiation using its autograd system. It provides more granular control than Keras through a lower-level API and is mainly used as a deep learning research platform but can also replace NumPy while enabling GPU computation.

It employs eager execution, in contrast to the static computation graphs used by, e.g., Theano or TensorFlow. Rather than initially defining and compiling a network for fast but static execution, it relies on its autograd package for automatic differentiation of Tensor operations, i.e., it computes gradients ‘on the fly’ so that network structures can be partially modified more easily. This is called define-by-run, meaning that backpropagation is defined by how your code runs, which in turn implies that every single iteration can be different. The PyTorch documentation provides a detailed tutorial on this.

The notebook [how_to_use_pytorch](03_how_to_use_pytorch.ipynb) illustrates how to use the 1.4 release.

- [PyTorch Documentation](https://pytorch.org/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials)
- [PyTorch Ecosystem](https://pytorch.org/ecosystem)
    - [AllenNLP](https://allennlp.org/), state-of-the-art NLP platform developed by the Allen Institute for Artificial Intelligence
    - [Flair](https://github.com/zalandoresearch/flair),  simple framework for state-of-the-art NLP developed at Zalando
    - [fst.ai](http://www.fast.ai/), simplifies training NN using modern best practices; offers online training

### Code example: How to use TensorFlow

TensorFlow has become the leading deep learning library shortly after its release in September 2015, one year before PyTorch. TensorFlow 2.0 aims to simplify the API that has grown increasingly complex over time by making the Keras API, integrated into TensorFlow as part of the contrib package since 2017 its principal interface, and adopting eager execution. It will continue to focus on a robust implementation across numerous platforms but will make it easier to experiment and do research.

The notebook [how_to_use_tensorflow](04_how_to_use_tensorflow.ipynb) illustrates how to use the 2.0 release.

- [TensorFlow.org](https://www.tensorflow.org/)
- [Standardizing on Keras: Guidance on High-level APIs in TensorFlow 2.0](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a)
- [TensorFlow.js](https://js.tensorflow.org/), A JavaScript library for training and deploying ML models in the browser and on Node.js

## Code example: Optimizing a neural network for a long-short trading strategy

In practice, we need to explore variations for the design options for the NN architecture and how we train it from those we outlined previously because we can never be sure from the outset which configuration best suits the data. 

This code example explores various architectures for a simple feedforward neural network to predict daily stock returns using the dataset developed in [Chapter 12](../12_gradient_boosting_machines) (see the notebook [preparing_the_model_data](../12_gradient_boosting_machines/04_preparing_the_model_data.ipynb)).

To this end, we will define a function that returns a TensorFlow model based on several architectural input parameters and cross-validate alternative designs using the MultipleTimeSeriesCV we introduced in Chapter 7. To assess the signal quality of the model predictions, we build a simple ranking-based long-short strategy based on an ensemble of the models that perform best during the in-sample cross-validation period. To limit the risk of false discoveries, we then evaluate the performance of this strategy for an out-of-sample test period.

### Optimizing the NN architecture

The notebook [how_to_optimize_a_NN_architecure](04_how_to_use_tensorflow.ipynb) explores various options to build a simple feedforward Neural Network to predict asset returns. To develop our trading strategy, we use the daily stock returns for 995 US stocks for the eight-year period from 2010 to 2017. 

### Backtesting a long-short strategy based on ensembled signals

To translate our NN model into a trading strategy, we generate predictions, evaluate their signal quality, create rules that define how to trade on these predictions, and backtest the performance of a strategy that implements these rules. 

The notebook [backtesting_with_zipline](05_backtesting_with_zipline.ipynb) contains the code examples for this section.
