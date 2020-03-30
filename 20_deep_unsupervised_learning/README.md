# Chapter 19: Autoencoders - Unsupervised Deep Learning

This chapter presents two unsupervised learning techniques that leverage deep learning: autoencoders, which have been around for decades, and Generative Adversarial Networks (GANs), which were introduced by Ian Goodfellow in 2014 and which Yann LeCun has called the most exciting idea in AI in the last ten years. 
- An autoencoder is a neural network trained to reproduce the input while learning a new representation of the data, encoded by the parameters of a hidden layer. Autoencoders have long been used for nonlinear dimensionality reduction and manifold learning. More recently, autoencoders have been designed as generative models that learn probability distributions over observed and latent variables. A variety of designs leverage the feedforward network, Convolutional Neural Network (CNN), and recurrent neural network (RNN) architectures we covered in the last three chapters.
- GANs are a recent innovation that train two neural nets—a generator and a discriminator—in a competitive setting. The generator aims to produce samples that the discriminator is unable to distinguish from a given class of training data. The result is a generative model capable of producing new (fake) samples that are representative of a certain target distribution. GANs have produced a wave of research and can be successfully applied in many domains. An example from the medical domain that could potentially be highly relevant for trading is the generation of time-series data that simulates alternative trajectories and can be used to train supervised or reinforcement algorithms.

More specifically, this chapter covers:

- Which types of autoencoders are of practical use and how they work
- How to build and train autoencoders using Python
- How GANs work, why they're useful, and how they could be applied to trading
- How to build GANs using Python

- [Unsupervised Learning](https://cilvr.nyu.edu/lib/exe/fetch.php?media=deeplearning:2016:lecun-20160308-unssupervised-learning-nyu.pdf), Yann LeCun, 2016

## How Autoencoders work

An autoencoder, in contrast, is a neural network designed exclusively to learn a new representation, that is, an encoding of the input. To this end, the training forces the network to faithfully reproduce the input. Since autoencoders typically use the same data as input and output, they are also considered an instance of self-supervised learning. In the process, the parameters of a hidden layer become the code that represents the input. 

- [Autoencoders](http://www.deeplearningbook.org/contents/autoencoders.html), Ian Goodfellow, Yoshua Bengio and Aaron Courville, Deep Learning Book, Chapter 14, MIT Press 2016

### Nonlinear dimensionality reduction

A traditional use case includes dimensionality reduction, achieved by limiting the size of the hidden layer so that it performs lossy compression. Such an autoencoder is called undercomplete and the purpose is to force it to learn the most salient properties of the data by minimizing a loss function. In addition to feedforward architectures, autoencoders can also use convolutional layers to learn hierarchical feature representations. 

The powerful capabilities of neural networks to represent complex functions require tight limitations of the capacity of the encoder and decoder to force the extraction of a useful signal rather than noise. In other words, when it is too easy for the network to recreate the input, it fails to learn only the most interesting aspects of the data. This challenge is similar to the overfitting phenomenon that frequently occurs when using models with a high capacity for supervised learning. Just as in these settings, regularization can help by adding constraints to the autoencoder that facilitate the learning of a useful representation.

### Sequence-to-Sequence Autoencoders

Sequence-to-sequence autoencoders are based on RNN components, such as Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRUs). They learn a compressed representation of sequential data and have been applied to video, text, audio, and time-series data.

- [A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html), Francois Chollet, September 2017
- [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681), Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov, 2016

### Variational Autoencoders

Variational Autoencoders (VAE) are more recent developments focused on generative modeling. More specifically, VAEs are designed to learn a latent variable model for the input data. Note that we encountered latent variables in Chapter 14, Topic Modeling.

Hence, VAEs do not let the network learn arbitrary functions as long as it faithfully reproduces the input. Instead, they aim to learn the parameters of a probability distribution that generates the input data. In other words, VAEs are generative models because, if successful, you can generate new data points by sampling from the distribution learned by the VAE.

- [Auto-encoding variational bayes](https://arxiv.org/abs/1312.6114), Diederik P Kingma, Max Welling, 2014

## How to build autoencoders using Python

The Keras library makes it fairly straightforward to build various types of autoencoders and the following examples are adapted from Keras' tutorials.

- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)

### Feedforward Autoencoders with Sparsity Constraints

The notebook [deep_autoencoders](01_deep_autoencoders.ipynb) illustrates how to implement several of the autoencoder models introduced in the preceding section using Keras. This includes autoencoders using deep feedforward nets and sparsity constraints. 

### Convolutional & Denoising Autoencoders

The notebook [convolutional_denoising_autoencoders](02_convolutional_denoising_autoencoders.ipynb) goes on to demonstrate how to implement convolutionals and denoising autencoders to recover corrupted image inputs.

### Variational Autoencoders

The notebook [variational_autoencoder](03_variational_autoencoder.ipynb) shows how to build a Variational Autoencoder using Keras.

## Generative Adversarial Networks

The supervised learning algorithms that we focused on for most of this book receive input data that's typically complex and predicts a numerical or categorical label that we can compare to the ground truth to evaluate its performance. These algorithms are also called discriminative models because they learn to differentiate between different output classes.

The goal of generative models is to produce complex output, such as realistic images, given simple input, which can even be random numbers. They achieve this by modeling a probability distribution over the possible output. This probability distribution can have many dimensions, for example, one for each pixel in an image or its character or token in a document. As a result, the model can generate output that are very likely representative of the class of output. 

- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf), Ian Goodfellow, 2017
- [Why is unsupervised learning important?](https://www.quora.com/Why-is-unsupervised-learning-important), Yoshua Bengio on Quora, 2018

### How GANs work

- [GAN Lab: Understanding Complex Deep Generative Models using Interactive Visual Experimentation](https://www.groundai.com/project/gan-lab-understanding-complex-deep-generative-models-using-interactive-visual-experimentation/), Minsuk Kahng, Nikhil Thorat, Duen Horng (Polo) Chau, Fernanda B. Viégas, and Martin Wattenberg, IEEE Transactions on Visualization and Computer Graphics, 25(1) (VAST 2018), Jan. 2019
    - [GitHub](https://poloclub.github.io/ganlab/)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), Ian Goodfellow, et al, 2014
- [Generative Adversarial Networks: an Overview](https://arxiv.org/pdf/1710.07035.pdf), Antonia Creswell, et al, 2017
- [Generative Models](https://blog.openai.com/generative-models/), OpenAI Blog

### Evolution of GAN Architectures

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf), Luke Metz et al, 2016
- [Conditional Generative Adversarial Net](https://arxiv.org/pdf/1411.1784.pdf), Medhi Mirza and Simon Osindero, 2014
- [Infogan: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657.pdf), Xi Chen et al, 2016
- [Stackgan: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242.pdf), Shaoting Zhang et al, 2016
- [Photo-realistic Single Image Super-resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf), Alejando Acosta et al, 2016
- [Unpaired Image-to-image Translation Using Cycle-consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf), Juan-Yan Zhu et al, 2018
- [Learning What and Where to Draw](https://arxiv.org/abs/1610.02454), Scott Reed, et al 2016
- [Fantastic GANs and where to find them](http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them)

### Applications of GANs

- [Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633), Cristóbal Esteban, Stephanie L. Hyland, Gunnar Rätsch, 2016
    - [GitHub Repo](https://github.com/ratschlab/RGAN)
- [MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks](https://arxiv.org/pdf/1901.04997.pdf), Dan Li, Dacheng Chen, Jonathan Goh, and See-Kiong Ng, 2019
    - [GitHub Repo](https://github.com/LiDan456/MAD-GANs)
- [GAN — Some cool applications](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900), Jonathan Hui, 2018
- [gans-awesome-applications](https://github.com/nashory/gans-awesome-applications), curated list of awesome GAN applications

### How to build GANs using Python

The notebook [deep_convolutional_generative_adversarial_network](04_deep_convolutional_generative_adversarial_network.ipynb) illustrates the implementation of a GAN using Python. It uses the Deep Convolutional GAN (DCGAN) example to synthesize images from the fashion MNIST dataset

- [Kears-GAN](https://github.com/eriklindernoren/Keras-GAN), numerous Keras GAN implementations
- [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN), numerous PyTorch GAN implementations