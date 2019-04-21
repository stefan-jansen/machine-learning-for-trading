# Chapter 16: Deep Learning

The chapter presents feedforward neural networks (NN) to demonstrate how to efficiently train large models using backpropagation, and manage the risks of overfitting. It also shows how to use of the frameworks Keras, TensorFlow 2.0, and PyTorch.

In the following chapters, we will build on this foundation to design and train a variety of architectures suitable for different investment applications with a particular focus on alternative data sources. These include recurrent NN tailored to sequential data like time series or natural language and convolutional NN particularly well suited to image data. We will also cover deep unsupervised learning, including Generative Adversarial Networks (GAN) to create synthetic data and reinforcement learning to train agents that interactively learn from their environment. In particular, this chapter will cover
- How DL solves AI challenges in complex domains
- How key innovations have propelled DL to its current popularity
- How feed-forward networks learn representations from data
- How to design and train deep neural networks in Python
- How to implement deep NN using Keras, TensorFlow, and PyTorch
- How to build and tune a deep NN to predict asset price moves

## How Deep Learning Works

- [Deep Learning](https://www.deeplearningbook.org/), Ian Goodfellow, Yoshua Bengio and Aaron Courville, MIT Press, 2016
- [Deep learning](https://www.nature.com/articles/nature14539), Yann LeCun, Yoshua Bengio, and Geoffrey Hinton, Nature 2015
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), Michael A. Nielsen, Determination Press, 2015
- [The Quest for Artificial Intelligence - A History of Ideas and Achievements](https://ai.stanford.edu/~nilsson/QAI/qai.pdf), Nils J. Nilsson, Cambridge University Press, 2010
- [One Hundred Year Study on Artificial Intelligence (AI100)](https://ai100.stanford.edu/)
- [TensorFlow Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.71056&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false), Interactive, browser-based Deep Learning platform


### Backpropagation

- [Gradient Checking & Advanced Optimization](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization), Unsupervised Feature Learning and Deep Learning, Stanford University
- [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#momentum), Sebastian Ruder, 2016

## How to build a Neural Network using Python

To gain a better understanding of how NN work, the notebook [01_build_and_train_feedforward_nn](build_and_train_feedforward_nn.ipynb) formulates as simple feedforward architecture and forward propagation computations using matrix algebra and implements it using Numpy, the Python counterpart of linear algebra.


## Popular Deep Learning libraries

Currently, the most popular DL libraries are TensorFlow (supported by Google), Keras (led by Francois Chollet, now at Google), and PyTorch (supported by Facebook). Development is very active with PyTorch just releasing version 1.0 and TensorFlow 2.0 expected in early Spring 2019 when it is expected to adopt Keras as its main interface.

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

### How to use Keras

Keras was designed as a high-level or meta API to accelerate the iterative workflow when designing and training deep neural networks with computational backends like TensorFlow, Theano, or CNTK. It has been integrated into TensorFlow in 2017 and is set to become the principal TensorFlow interface with the 2.0 release. You can also combine code from both libraries to leverage Keras’ high-level abstractions as well as customized TensorFlow graph operations.

The notebook [how_to_use_keras](02_how_to_use_keras.ipynb) demonstrates the functionality.

- [A Full Hardware Guide to Deep Learning](http://timdettmers.com/2018/12/16/deep-learning-hardware-guide/), Tim Dettmers
- [Keras documentation](https://keras.io/)

### How to use Tensorboard

Tensorboard is a great visualization tool that comes with TensorFlow. It includes a suite of visualization tools to simplify the understanding, debugging, and optimization of neural networks.

You can use it to visualize the computational graph, plot various execution and performance metrics, and even visualize image data processed by the network. It also permits comparisons of different training runs.
When you run the how_to_use_keras notebook, and with TensorFlow installed, you can launch Tensorboard from the command line:

```python
tensorboard --logdir=/full_path_to_your_logs ## e.g. ./tensorboard
```

- [TensorBoard: Visualizing Learning](https://www.tensorflow.org/guide/summaries_and_tensorboard)

### How to use PyTorch 1.0

Pytorch has been developed at the Facebook AI Research group led by Yann LeCunn and the first alpha version released in September 2016. It provides deep integration with Python libraries like Numpy that can be used to extend its functionality, strong GPU acceleration, and automatic differentiation using its autograd system. It provides more granular control than Keras through a lower-level API and is mainly used as a deep learning research platform but can also replace NumPy while enabling GPU computation.

It employs eager execution, in contrast to the static computation graphs used by, e.g., Theano or TensorFlow. Rather than initially defining and compiling a network for fast but static execution, it relies on its autograd package for automatic differentiation of Tensor operations, i.e., it computes gradients ‘on the fly’ so that network structures can be partially modified more easily. This is called define-by-run, meaning that backpropagation is defined by how your code runs, which in turn implies that every single iteration can be different. The PyTorch documentation provides a detailed tutorial on this.

- [PyTorch Documentation](https://pytorch.org/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials)
- [PyTorch Ecosystem](https://pytorch.org/ecosystem)
    - [AllenNLP](https://allennlp.org/), state-of-the-art NLP platform developed by the Allen Institute for Artificial Intelligence
    - [Flair](https://github.com/zalandoresearch/flair),  simple framework for state-of-the-art NLP developed at Zalando
    - [fst.ai](http://www.fast.ai/), simplifies training NN using modern best practices; offers online training

### How to use TensorFlow

TensorFlow has become the leading deep learning library shortly after its release in September 2015, one year before PyTorch. TensorFlow 2.0 aims to simplify the API that has grown increasingly complex over time by making the Keras API, integrated into TensorFlow as part of the contrib package since 2017 its principal interface, and adopting eager execution. It will continue to focus on a robust implementation across numerous platforms but will make it easier to experiment and do research.

The notebook [how_to_use_tensorflow](04_how_to_use_tensorflow.ipynb) will  illustrateshow to use the 2.0 release (updated as the interface stabilizes).

- [TensorFlow.org](https://www.tensorflow.org/)
- [Standardizing on Keras: Guidance on High-level APIs in TensorFlow 2.0](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a)
- [TensorFlow.js](https://js.tensorflow.org/), A JavaScript library for training and deploying ML models in the browser and on Node.js

## How to optimize Neural Network Architectures

In practice, we need to explore variations of the design options outlined above because we can rarely be sure from the outset which network architecture best suits the data.
The GridSearchCV class provided by scikit-learn that we encountered in Chapter 6, The Machine Learning Workflow conveniently automates this process. Just be mindful of the risk of false discoveries and keep track of how many experiments you are running to adjust the results accordingly.

The notebook [how_to_optimize_a_NN_architecure](04_how_to_use_tensorflow.ipynb) explores various options to build a simple feedforward Neural Network to predict asset price moves for a one-month horizon. The python script of the same name aims to facilitate running the code on a server in order to speed up computation.
