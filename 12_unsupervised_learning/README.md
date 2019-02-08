# Chapter 12: Unsupervised Learning

## Dimensionality reduction

### The curse of dimensionality

#### Code Examples

The notebook `curse_of_dimensionality` contains the simulation used in this section.

### Linear Dimensionality Reduction

- [Dimension Reduction: A Guided Tour](https://www.microsoft.com/en-us/research/publication/dimension-reduction-a-guided-tour-2/), Chris J.C. Burges, Foundations and Trends in Machine Learning, January 2010

#### PCA

- [Mixtures of Probabilistic Principal Component Analysers](http://www.miketipping.com/papers/met-mppca.pdf), Michael E. Tipping and Christopher M. Bishop, Neural Computation 11(2), pp 443–482. MIT Press
- [Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions](http://users.cms.caltech.edu/~jtropp/papers/HMT11-Finding-Structure-SIREV.pdf), N. Halko†, P. G. Martinsson, J. A. Tropp, SIAM REVIEW, Vol. 53, No. 2, pp. 217–288
- [Relationship between SVD and PCA. How to use SVD to perform PCA?](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca), excellent technical CrossValidated StackExchange answer with visualization

##### Visualizing PCA in 2D


#### ICA

- [Independent Component Analysis: Algorithms and Applications](https://www.sciencedirect.com/science/article/pii/S0893608000000265), Aapo Hyvärinen and Erkki Oja, Neural Networks, 2000
- [Independent Components Analysis](http://cs229.stanford.edu/notes/cs229-notes11.pdf), CS229 Lecture Notes, Andrew Ng
- [Common factors in prices, order flows, and liquidity](https://www.sciencedirect.com/science/article/pii/S0304405X0000091X), Hasbrouck and Seppi, Journal of Financial Economics, 2001
- [Volatility Modelling of Multivariate Financial Time Series by Using ICA-GARCH Models](https://link.springer.com/chapter/10.1007/11508069_74), Edmond H. C. Wu, Philip L. H. Yu, in: Gallagher M., Hogan J.P., Maire F. (eds) Intelligent Data Engineering and Automated Learning - IDEAL 2005
- [The Prediction Performance of Independent Factor Models](http://www.cs.cuhk.hk/~lwchan/papers/icapred.pdf), Chan, In: proceedings of the 2002 IEEE International Joint Conference on Neural Networks
- [An Overview of Independent Component Analysis and Its Applications](http://www.informatica.si/ojs-2.4.3/index.php/informatica/article/download/334/333), Ganesh R. Naik, Dinesh K Kumar, Informatica 2011

#### PCA and Risk Factor Models

- [Characteristics Are Covariances: A Unified Model of Risk and Return](http://www.nber.org/2018LTAM/kelly.pdf), Kelly, Pruitt and Su, NBER, 2018

#### PCA and Eigen Portfolios

- [Statistical Arbitrage in the U.S. Equities Market](https://math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf), Marco Avellaneda and Jeong-Hyun Lee, 2008


### Manifold Learning

#### Data

- [MNIST Data](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)

#### Local Linear Embedding

- [Locally Linear Embedding](https://cs.nyu.edu/~roweis/lle/), Sam T. Roweis and Lawrence K. Saul (LLE author website)

#### t-SNE

- [Visualizing Data using t-SNE](http://www.cs.toronto.edu/~hinton/absps/tsne.pdf), van der Maaten, Hinton, Journal of Machine Learning Research, 2008
- [Visualizing Time-Dependent Data Using Dynamic t-SNE](http://www.cs.rug.nl/~alext/PAPERS/EuroVis16/paper.pdf), Rauber, Falcão, Telea, Eurographics Conference on Visualization (EuroVis) 2016
- [t-Distributed Stochastic Neighbor Embedding Wins Merck Viz Challenge](http://blog.kaggle.com/2012/11/02/t-distributed-stochastic-neighbor-embedding-wins-merck-viz-challenge/), Kaggle Blog 2016
- [t-SNE: Google Tech Talk](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw), van der Maaten, 2013
- [Parametric t-SNE](https://github.com/kylemcdonald/Parametric-t-SNE), fast t-SNE implementation using Keras by Kyle McDonald

#### UMAP

- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426), Leland McInnes, John Healy, 2018


### Hierarchical Risk Parity

- [Building Diversified Portfolios that Outperform Out-of-Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678), Lopez de Prado, Journal of Portfolio Management, 2015
- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Raffinot 2016




