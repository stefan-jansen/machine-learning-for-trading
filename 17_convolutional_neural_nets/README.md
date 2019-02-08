# Chapter 17: Convolutional Neural Networks

## How to build a Deep ConvNet

### How Convolutional Layers work

- [Deep Learning](http://www.deeplearningbook.org/contents/convnets.html), Chapter 9, Convolutional Networks, Ian Goodfellow et al, MIT Press, 2016
- [Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/#conv), Module 2 in CS231n Convolutional Neural Networks for Visual Recognition, Lecture Notes by Andrew Karpathy, Stanford, 2016
- [Convnet Benchmarks](https://github.com/soumith/convnet-benchmarks), Benchmarking of all publicly accessible implementations of convnets
- [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html), ConvNetJS CIFAR-10 demo in the browser by Andrew Karpathy
- [An Interactive Node-Link Visualization of Convolutional Neural Networks](http://scs.ryerson.ca/~aharley/vis/), interactive CNN visualization
- [GradientBased Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf), Yann LeCun Leon Bottou Yoshua Bengio and Patrick, IEEE, 1998
- [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/), Christopher Olah, 2014
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122), Fisher Yu, Vladlen Koltun, ICLR 2016

#### Code examples



### Computer Vision Tasks

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/), You Only Look Once real-time object detection
- [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf), Girshick et al, Berkely, arxiv 2014
- [Playing around with RCNN](https://cs.stanford.edu/people/karpathy/rcnn/), Andrew Karpathy, Stanford
- [R-CNN, Fast R-CNN, Faster R-CNN, YOLO — Object Detection Algorithms](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e), Rohith Ghandi, 2018


### Reference Architectures & Benchmarks

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


## CNN with Keras and PyTorch


## Transfer Learning

- [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [How transferable are features in deep neural networks?](https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf), Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson, NIPS, 2014
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## Capsule Nets
- [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829), Sara Sabour, Nicholas Frosst, Geoffrey E Hinton, arxiv, 2017

## Resources

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/syllabus.html), Stanford’s deep learning course. Helpful for building foundations, with engaging lectures and illustrative problem sets.
- [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](http://www.image-net.org/challenges/LSVRC/)


docker run -it -p 8889:8888 -v /home/stefan/projects/machine-learning-for-trading/17_convolutional_neural_nets:/cnn --name tensorflow tensorflow/tensorflow:latest-gpu-py3 bash


