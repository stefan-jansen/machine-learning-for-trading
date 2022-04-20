# Convolutional Neural Networks: Time Series as Images

In this chapter, we introduce the first specialized Deep Learning architectures that we will cover in part 4. Deep Convolutional Neural Networks, also ConvNets or CNN, have enabled superhuman performance in classifying images, video, speech, and audio. Recurrent nets, the subject of the following chapter, have performed exceptionally well on sequential data such as text and speech.

CNNs are named after the linear algebra operation called convolution that replaces the general matrix multiplication typical of feed-forward networks (discussed in the last chapter on Deep Learning) in at least one of their layers. We will discuss how convolutions work and why they are particularly useful to data with a certain regular structure like images or time series.

Research into CNN architectures has proceeded very rapidly and new architectures that improve benchmark performance continue to emerge. We will describe a set of building blocks that consistently appears in successful applications and illustrate their application to image data and financial time series. We will also demonstrate how transfer learning can speed up learning by using pre-trained weights for some of the CNN layers.

## Content

1. [How CNNs learn to model grid-like data](#how-cnns-learn-to-model-grid-like-data)
    * [Code example: From hand-coding to learning and synthesizing filters from data](#code-example-from-hand-coding-to-learning-and-synthesizing-filters-from-data)
    * [How the key elements of a convolutional layer operate](#how-the-key-elements-of-a-convolutional-layer-operate)
    * [Computer Vision Tasks](#computer-vision-tasks)
    * [The evolution of CNN architectures: key innovations](#the-evolution-of-cnn-architectures-key-innovations)
2. [CNN for Images: From Satellite Data to Object Detection](#cnn-for-images-from-satellite-data-to-object-detection)
    * [Code example: LeNet5: The first CNN with industrial applications](#code-example-lenet5-the-first-cnn-with-industrial-applications)
    * [Code example: AlexNet - reigniting deep learning research](#code-example-alexnet---reigniting-deep-learning-research)
    * [Code example: transfer learning with VGG16 in practice](#code-example-transfer-learning-with-vgg16-in-practice)
        - [How to extract bottleneck features](#how-to-extract-bottleneck-features)
        - [How to fine-tune a pre-trained model](#how-to-fine-tune-a-pre-trained-model)
    * [Code example: identifying land use with satellite images using transfer learning](#code-example-identifying-land-use-with-satellite-images-using-transfer-learning)
    * [Code example: object detection in practice with Google Street View House Numbers](#code-example-object-detection-in-practice-with-google-street-view-house-numbers)
        - [Preprocessing the source images](#preprocessing-the-source-images)
        - [Transfer learning with a custom final layer for multiple outputs](#transfer-learning-with-a-custom-final-layer-for-multiple-outputs)
3. [CNN for time series data: predicting stock returns](#cnn-for-time-series-data-predicting-stock-returns)
    * [Code example: building an autoregressive CNN with 1D convolutions](#code-example-building-an-autoregressive-cnn-with-1d-convolutions)
    * [Code example: CNN-TA - clustering financial time series in 2D image format](#code-example-cnn-ta---clustering-financial-time-series-in-2d-image-format)
        - [Creating the 2D time series of financial indicators](#creating-the-2d-time-series-of-financial-indicators)
        - [Select and cluster the most relevant features](#select-and-cluster-the-most-relevant-features)
        - [Create and train a convolutional neural network](#create-and-train-a-convolutional-neural-network)
        - [Backtesting a long-short trading strategy](#backtesting-a-long-short-trading-strategy)

## How CNNs learn to model grid-like data

CNNs are conceptually similar to the feedforward NNs we covered in the previous chapter. They consist of units that contain parameters called weights and biases, and the training process adjusts these parameters to optimize the network’s output for a given input. Each unit applies its parameters to a linear operation on the input data or activations received from other units, possibly followed by a non-linear transformation. 

CNNs differ because they encode the assumption that the input has a structure most commonly found in image data where pixels form a two-dimensional grid, typically with several channels to represent the components of the color signal, such as the red, green and blue channels of the RGB color model.

The most important element to encode the assumption of a grid-like topology is the convolution operation that gives CNNs their name, combined with pooling. We will see that the specific assumptions about the functional relationship between input and output data implies that CNNs need far fewer parameters and compute more efficiently.

### Code example: From hand-coding to learning and synthesizing filters from data

For image data, this local structure has traditionally motivated the development of hand-coded filters that extract such patterns for the use as features in machine learning models.
- The notebook [filter_example](01_filter_example.ipynb) illustrates how to use hand-coded filters in a convolutional network and visualize the resulting transformation of the image.
- See [Interpretability of Deep Learning Models with Tensorflow 2.0](https://www.sicara.ai/blog/2019-08-28-interpretability-deep-learning-tensorflow) for an example visualization of the patterns learned by CNN filters.

### How the key elements of a convolutional layer operate

Fully-connected feedforwardNNs make no assumptions about the topology, or local structure of the input data so that arbitrarily reordering the features has no impact on the training result.

For many data sources, however, local structure is quite significant. Examples include autocorrelation in time series or the spatial correlation among pixel values due to common patterns like edges or corners. For image data, this local structure has traditionally motivated the development of hand-coded filter methods that extract local patterns for the use as features in machine learning models.

- [Deep Learning](http://www.deeplearningbook.org/contents/convnets.html), Chapter 9, Convolutional Networks, Ian Goodfellow et al, MIT Press, 2016
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/syllabus.html), Stanford’s deep learning course. Helpful for building foundations, with engaging lectures and illustrative problem sets.
- [Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/#conv), Module 2 in CS231n Convolutional Neural Networks for Visual Recognition, Lecture Notes by Andrew Karpathy, Stanford, 2016
- [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](http://www.image-net.org/challenges/LSVRC/)
- [Convnet Benchmarks](https://github.com/soumith/convnet-benchmarks), Benchmarking of all publicly accessible implementations of convnets
- [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html), ConvNetJS CIFAR-10 demo in the browser by Andrew Karpathy
- [An Interactive Node-Link Visualization of Convolutional Neural Networks](http://scs.ryerson.ca/~aharley/vis/), interactive CNN visualization
- [GradientBased Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf), Yann LeCun Leon Bottou Yoshua Bengio and Patrick, IEEE, 1998
- [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/), Christopher Olah, 2014
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122), Fisher Yu, Vladlen Koltun, ICLR 2016

### Computer Vision Tasks

Image classification is a fundamental computer vision task that requires labeling an image based on certain objects it contains. Many practical applications, including investment and trading strategies, require additional information. 
- The object detection task requires not only the identification but also the spatial location of all objects of interest, typically using bounding boxes. Several algorithms have been developed to overcome the inefficiency of brute-force sliding-window approaches, including region proposal methods (R-CNN) and the You Only Look Once (YOLO) real-time object detection algorithm (see references on GitHub).
- The object segmentation task goes a step further and requires a class label and an outline of every object in the input image. This may be useful to count objects in an image and evaluate a level of activity. 
- Semantic segmentation, also called scene parsing, makes dense predictions to assign a class label to each pixel in the image. As a result, the image is divided into semantic regions and each pixel is assigned to its enclosing object or region.

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/), You Only Look Once real-time object detection
- [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf), Girshick et al, Berkely, arxiv 2014
- [Playing around with RCNN](https://cs.stanford.edu/people/karpathy/rcnn/), Andrew Karpathy, Stanford
- [R-CNN, Fast R-CNN, Faster R-CNN, YOLO — Object Detection Algorithms](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e), Rohith Ghandi, 2018

### The evolution of CNN architectures: key innovations

Several CNN architectures have pushed performance boundaries over the past two decades by introducing important innovations. Predictive performance growth accelerated dramatically with the arrival of big data in the form of ImageNet (Fei-Fei 2015) with 14 million images assigned to 20,000 classes by humans via Amazon’s Mechanical Turk. The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) became the focal point of CNN progress around a slightly smaller set of 1.2 million images from 1,000 classes.

- [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf), Long et al, Berkeley
- [Mask R-CNN](https://arxiv.org/abs/1703.06870), Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, arxiv, 2017
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf), Olaf Ronneberger, Philipp Fischer, and Thomas Brox, arxiv 2015
- [U-Net Tutorial](http://deeplearning.net/tutorial/unet.html)
- [Very Deep Convolutional Networks for Large-Scale Visual Recognition](http://www.robots.ox.ac.uk/~vgg/research/very_deep/), Karen Simonyan and Andrew Zisserman on VGG16 that won the ImageNet ILSVRC-2014 competition
- [Benchmarks for popular CNN models](https://github.com/jcjohnson/cnn-benchmarks)
- [Analysis of deep neural networks](https://medium.com/@culurciello/analysis-of-deep-neural-networks-dcf398e71aae), Alfredo Canziani, Thomas Molnar, Lukasz Burzawa, Dawood Sheik, Abhishek Chaurasia, Eugenio Culurciello, 2018
- [LeNet-5 Demos](http://yann.lecun.com/exdb/lenet/index.html)
- [Neural Network Architectures](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf), Kaiming He et al, Microsoft Research, 2015
- [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567), Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, arxiv 2015
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261), Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi, arxiv, 2016
- [Network In Network](https://arxiv.org/pdf/1312.4400v3.pdf), Min Lin et al, arxiv 2014
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), Sergey Ioffe, Christian Szegedy, arxiv 2015
- [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035), Vincent Fung, 2017

## CNN for Images: From Satellite Data to Object Detection

This section demonstrates how to solve key computer vision tasks such as image classification and object detection. As mentioned in the introduction and in Chapter 3 on alternative data, image data can inform a trading strategy by providing clues about future trends, changing fundamentals, or specific events relevant for a target asset class or investment universe. Popular examples include exploiting satellite images for clues about the supply of agricultural commodities, consumer and economic activity, or the status of manufacturing or raw material supply chains. Specific tasks might include, for example: 
- Image classification: identify whether cultivated land for certain crops is expanding or predict harvest quality and quantities, or 
- Object detection: count the number of oil tankers on a certain transport route or the number of cars in a parking lot, or identify the location of shoppers in a mall.

### Code example: LeNet5: The first CNN with industrial applications

All libraries we introduced in the last chapter provide support for convolutional layers. 

The notebook [digit_classification_with_lenet5](02_digit_classification_with_lenet5.ipynb) illustrates the LeNet5 architecture using the most basic MNIST handwritten digit dataset,

### Code example: AlexNet - reigniting deep learning research

Fast-forward to 2012, and we move on to the deeper and more modern AlexNet architecture. We will use the CIFAR10 dataset that uses 60,000 ImageNet samples, compressed to 32x32 pixel resolution (from the original 224x224), but still with three color channels. There are only 10 of the original 1,000 classes. 

See the notebook [image_classification_with_alexnet](03_image_classification_with_alexnet.ipynb) for implementation, including the use of data augmentation.

### Code example: transfer learning with VGG16 in practice

In practice, we often do not have enough data to train a CNN from scratch with random initialization. Transfer learning is a machine learning technique that repurposes a model trained on one set of data for another task. Naturally, it works if the learning from the first task carries over to the task of interest. If successful, it can lead to better performance and faster training that requires less labeled data than training a neural network from scratch on the target task.

Tensorflow 2, for example, contains pre-trained models for several of the reference architectures discussed previously, namely VGG16 and its larger version VGG19, ResNet50, InceptionV3, and InceptionResNetV2, as well as MobileNet, DenseNet, NASNet, and MobileNetV2.

The transfer learning approach to CNN relies on pre-training on a very large dataset like ImageNet. The goal is that the convolutional filters extract a feature representation that generalizes to new images. In a second step, it leverages the result to either initialize and retrain a new CNN or as inputs to in a new network that tackles the task of interest.

CNN architectures typically use a sequence of convolutional layers to detect hierarchical patterns, adding one or more fully-connected layers to map the convolutional activations to the outcome classes or values. The output of the last convolutional layer that feeds into the fully-connected part is called bottleneck features. We can use the bottleneck features of a pre-trained network as inputs into a new fully-connected network, usually after applying a ReLU activation function. 

In other words, we freeze the convolutional layers and replace the dense part of the network. An additional benefit is that we can then use inputs of different sizes because it is the dense layers that constrain the input size. 

Alternatively, we can use the bottleneck features as inputs into a different machine learning algorithm. In the AlexNet architecture, e.g., the bottleneck layer computes a vector with 4096 entries for each 224 x 224 input image. We then use this vector as features for a new model.

Alternatively, we can go a step further and not only replace and retrain the classifier on top of the CNN using new data but to also fine-tune the weights of the pre-trained CNN. To achieve this, we continue training, either only for later layers while freezing the weights of some earlier layers. The motivation is to preserve presumably more generic patterns learned by lower layers, such as edge or color blob detectors while allowing later layers of the CNN to adapt to the details of a new task. ImageNet, e.g., contains a wide variety of dog breeds which may lead to feature representations specifically useful for differentiating between these classes.

- [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [How transferable are features in deep neural networks?](https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf), Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson, NIPS, 2014
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

#### How to extract bottleneck features

The notebook [bottleneck_features](09_bottleneck_features.ipynb) illustrates how to download the pre-trained VGG16 model, either with the final layers to generate predictions or without the final layers to extract the outputs produced by the bottleneck features.

#### How to fine-tune a pre-trained model

The notebook [transfer_learning](10_transfer_learning.ipynb), adapted from a TensorFlow 2 tutorial, demonstrates how to freeze some or all of the layers of a pre-trained model and continue training using a new fully-connected set of layers and data with a different format.

### Code example: identifying land use with satellite images using transfer learning

Satellite images figure prominently among alternative data (see [Chapter 3](../03_alternative_data)). For instance, commodity traders may rely on satellite images to predict the supply of certain crops or activity at mining sites, oil or tanker traffic. 

To illustrate working with this type of data, we load the [EuroSat dataset](https://arxiv.org/abs/1709.00029) included in the TensorFlow 2 datasets (Helber et al. 2017). The EuroSat dataset includes around 27,000 images in 64x64 format that represent 10 different types of land uses.
 
The notebook [satellite_images](11_satellite_images.ipynb) downloads the [DenseNet201](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet201) architecture from `tensorflow.keras.applications` and replace its final layers.

We use 10 percent of the training images for validation purposes and achieve the best out-of-sample classification accuracy of 97.96 percent after ten epochs. This exceeds the performance cited in the original paper for the best performing ResNet-50 architecture with 90-10 split.

### Code example: object detection in practice with Google Street View House Numbers

Object detection requires the ability to distinguish between several classes of objects and to decide how many and which of these objects are present in an image.

A prominent example is Ian Goodfellow’s identification of house numbers from Google’s street view dataset. It requires to identify 
- how many of up to five digits make up the house number, 
- The correct digit for each component, and
- The proper order of the constituent digits.

See the [data](../data) directory for instructions on obtaining the dataset.

#### Preprocessing the source images

The notebooks [svhn_preprocessing](12_svhn_preprocessing.ipynb) contains code to produce a simplified, cropped dataset that uses bounding box information to create regularly shaped 32x32 images containing the digits; the original images are of arbitrary shape.

#### Transfer learning with a custom final layer for multiple outputs

The notebook [svhn_object_detection](13_svhn_object_detection.ipynb) goes on to illustrate how to build a deep CNN using Keras’ functional API to generate multiple outputs: one to predict how many digits are present, and five for the value of each in the order they appear.

## CNN for time series data: predicting stock returns

CNN were originally developed to process image data and have achieved superhuman performance on various computer vision tasks. As discussed in the first section, time series data has a grid-like structure similar to that of images, and CNN have been successfully applied to one-, two- and three dimensional representations of temporal data. 

The application of CNN to time series will most likely bear fruit if the data meets the model’s key assumption that local patterns or relationships help predict the outcome. In the time-series context, local patterns could be autocorrelation or similar non-linear relationships at relevant intervals. Along the second and third dimension, local patterns imply systematic relationships among different components of a multivariate series or among these series for different tickers. Since locality matters, it is important that the data is organized accordingly in contrast to feed-forward networks where shuffling the elements of any dimension does not negatively affect the learning process.

### Code example: building an autoregressive CNN with 1D convolutions

We will introduce the time series use case for CNN with a univariate autoregressive asset return model. More specifically, the model receives the most recent 12 months of returns and uses a single layer of one-dimensional convolutions to predict the subsequent month.

The notebook [time_series_prediction](04_time_series_prediction.ipynb) illustrates the time series use case with the univariate asset price forecast example we introduced in the last chapter. Recall that we create rolling monthly stock returns and use the 24 lagged returns alongside one-hot-encoded month information to predict whether the subsequent monthly return is positive or negative.

### Code example: CNN-TA - clustering financial time series in 2D image format

To exploit the grid-like structure of time-series data, we can use CNN architectures for univariate and multivariate time series. In the latter case, we consider different time series as channels, similar to the different color signals.

An alternative approach converts a time series of alpha factors into a two-dimensional format to leverage the ability of CNNs to detect local patterns. [Sezer and Ozbayoglu](https://www.sciencedirect.com/science/article/abs/pii/S1568494618302151) (2018) propose [CNN-TA](https://github.com/omerbsezer/CNN-TA) that computes 15 technical indicators for different intervals and uses hierarchical clustering (see Chapter 13) to locate indicators that behave similarly close to each other in a 2D grid.

#### Creating the 2D time series of financial indicators

The notebook [engineer_cnn_features](05_cnn_for_trading_feature_engineering.ipynb) creates technical indicators at different intervals.

#### Select and cluster the most relevant features
 
The notebook [convert_cnn_features_to_image_format](06_cnn_for_trading_features_to_clustered_image_format.ipynb) selects the 15 most relevant features from the 20 candidates to fill the 15⨉15 input grid and then applies hierarchical clustering.

#### Create and train a convolutional neural network

Now we are ready to design, train and evaluate a CNN following the steps outlined in the previous section. The notebook [cnn_for_trading](07_cnn_for_trading.ipynb) contains the relevant code examples.

#### Backtesting a long-short trading strategy

To get a sense of the signal quality, we compute the spread between equal-weighted portfolios invested in stocks selected according to the signal quintiles using [Alphalens](https://github.com/quantopian/alphalens) (see [Chapter 4](../04_alpha_factor_research)).

<p align="center">
<img src="https://i.imgur.com/JlKttDL.png" width="80%">
</p>




