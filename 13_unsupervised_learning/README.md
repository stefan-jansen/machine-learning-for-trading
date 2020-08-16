# Unsupervised Learning: From Data-Driven Risk Factors to Hierarchical Risk Parity 

Unsupervised learning is useful when a dataset contains only features and no measurement of the outcome, or when we want to extract information independent from the outcome. Instead of predicting future outcomes, the goal is to learn an informative representation of the data that is useful for solving another task, including the exploration of a data set. Examples include the identification of topics to summarize documents (see [Chapter 14](../14_topic_modeling), the reduction of the number of features to lower the risk of overfitting and computational cost for supervised learning, or to group similar observations as illustrated by the use of clustering for asset allocation at the end of this chapter.

Dimensionality reduction and clustering are the main tasks for unsupervised learning: 
- Dimensionality reduction transforms the existing features into a new, smaller set while minimizing the loss of information. A broad range of algorithms exists that differ by how they measure the loss of information, whether they apply linear or non-linear transformations or the constraints they impose on the new feature set. 
- Clustering algorithms identify and group similar observations or features instead of identifying new features. Algorithms differ in how they define the similarity of observations and their assumptions about the resulting groups.

More specifically, this chapter covers:
- How principal and independent component analysis (PCA and ICA) perform linear dimensionality reduction
- Identifying data-driven risk factors and eigenportfolios from asset returns using PCA
- Effectively visualizing nonlinear, high-dimensional data using manifold learning
- Using T-SNE and UMAP to explore high-dimensional image data
- How k-means, hierarchical, and density-based clustering algorithms work
- Using agglomerative clustering to build robust portfolios with hierarchical risk parity

## Content

1. [Code Example: the curse of dimensionality](#code-example-the-curse-of-dimensionality)
2. [Linear Dimensionality Reduction](#linear-dimensionality-reduction)
    * [Code Example: Principal Component Analysis](#code-example-principal-component-analysis)
        - [Visualizing key ideas behind PCA ](#visualizing-key-ideas-behind-pca-)
        - [How the PCA algorithm works](#how-the-pca-algorithm-works)
    * [References](#references)
3. [Code Examples: PCA for Trading ](#code-examples-pca-for-trading-)
    * [Data-driven risk factors](#data-driven-risk-factors)
    * [Eigenportfolios](#eigenportfolios)
    * [References](#references-2)
4. [Independent Component Analysis](#independent-component-analysis)
5. [Manifold Learning](#manifold-learning)
    * [Code Example: what a manifold looks like ](#code-example-what-a-manifold-looks-like-)
    * [Code Example: Local Linear Embedding](#code-example-local-linear-embedding)
    * [References](#references-3)
6. [Code Examples: visualizing high-dimensional image and asset price data with manifold learning](#code-examples-visualizing-high-dimensional-image-and-asset-price-data-with-manifold-learning)
    * [t-distributed stochastic neighbor embedding (t-SNE)](#t-distributed-stochastic-neighbor-embedding-t-sne)
    * [UMAP](#umap)
7. [Cluster Algorithms](#cluster-algorithms)
    * [Code example: comparing cluster algorithms](#code-example-comparing-cluster-algorithms)
    * [Code example: k-Means](#code-example-k-means)
        - [The algorithm](#the-algorithm)
        - [Evaluating the results](#evaluating-the-results)
    * [Code example: Hierarchical Clustering](#code-example-hierarchical-clustering)
    * [Code example: Density-Based Clustering](#code-example-density-based-clustering)
    * [Code example: Gaussian Mixture Models](#code-example-gaussian-mixture-models)
    * [Code example: Hierarchical Risk Parity](#code-example-hierarchical-risk-parity)
        - [The algorithm](#the-algorithm-2)
        - [Backtest comparison with alternatives](#backtest-comparison-with-alternatives)
    * [References](#references-4)

## Code Example: the curse of dimensionality

The number of dimensions of a dataset matter because each new dimension can add signal concerning an outcome. However, there is also a downside known as the curse of dimensionality: as the number of independent features grows while the number of observations remains constant, the average distance between data points also grows, and the density of the feature space drops exponentially. The implications for machine learning are dramatic because prediction becomes much harder when observations are more distant, i.e., different from each other.

The notebook [curse_of_dimensionality](01_linear_dimensionality_reduction/00_curse_of_dimensionality.ipynb) simulates how the average and minimum distances between data points increase as the number of dimensions grows.

## Linear Dimensionality Reduction

Linear dimensionality reduction algorithms compute linear combinations that translate, rotate, and rescale the original features to capture significant variation in the data, subject to constraints on the characteristics of the new features.

This section introduces these two algorithms and then illustrates how to apply PCA to asset returns to learn risk factors from the data, and to build so-called eigen portfolios for systematic trading strategies.

- [Dimension Reduction: A Guided Tour](https://www.microsoft.com/en-us/research/publication/dimension-reduction-a-guided-tour-2/), Chris J.C. Burges, Foundations and Trends in Machine Learning, January 2010

### Code Example: Principal Component Analysis

PCA finds principal components as linear combinations of the existing features and uses these components to represent the original data. The number of components is a hyperparameter that determines the target dimensionality and needs to be equal to or smaller than the number of observations or columns, whichever is smaller.

#### Visualizing key ideas behind PCA 

The notebook [pca_key_ideas](01_linear_dimensionality_reduction/01_pca_key_ideas.ipynb) visualizes principal components in 2D and 3D.

PCA aims to capture most of the variance in the data, to make it easy to recover the original features, and that each component adds information. It reduces dimensionality by projecting the original data into the principal component space. PCA makes several assumptions that are important to keep in mind. These include:
- high variance implies a high signal-to-noise ratio
- the data is standardized so that the variance is comparable across features
- linear transformations capture the relevant aspects of the data, and
- higher-order statistics beyond the first and second moment do not matter, which implies that the data has a normal distribution

The emphasis on the first and second moments align with standard risk/return metrics, but the normality assumption may conflict with the characteristics of market data.

#### How the PCA algorithm works

The notebook [the_math_behind_pca](01_linear_dimensionality_reduction/02_the_math_behind_pca.ipynb) illustrate the computation of principal components.

### References

- [Mixtures of Probabilistic Principal Component Analysers](http://www.miketipping.com/papers/met-mppca.pdf), Michael E. Tipping and Christopher M. Bishop, Neural Computation 11(2), pp 443–482. MIT Press
- [Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions](http://users.cms.caltech.edu/~jtropp/papers/HMT11-Finding-Structure-SIREV.pdf), N. Halko†, P. G. Martinsson, J. A. Tropp, SIAM REVIEW, Vol. 53, No. 2, pp. 217–288
- [Relationship between SVD and PCA. How to use SVD to perform PCA?](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca), excellent technical CrossValidated StackExchange answer with visualization

## Code Examples: PCA for Trading 

PCA is useful for algorithmic trading in several respects, including the data-driven derivation of risk factors by applying PCA to asset returns, and the construction of uncorrelated portfolios based on the principal components of the correlation matrix of asset returns.
 
### Data-driven risk factors

In [Chapter 07 - Linear Models](../07_linear_models/02_fama_macbeth.ipynb), we explored risk factor models used in quantitative finance to capture the main drivers of returns. These models explain differences in returns on assets based on their exposure to systematic risk factors and the rewards associated with these factors.
 
In particular, we explored the Fama-French approach that specifies factors based on prior knowledge about the empirical behavior of average returns, treats these factors as observable, and then estimates risk model coefficients using linear regression. An alternative approach treats risk factors as latent variables and uses factor analytic techniques like PCA to simultaneously estimate the factors and how the drive returns from historical returns.

- The notebook [pca_and_risk_factor_models](01_linear_dimensionality_reduction/03_pca_and_risk_factor_models.ipynb) demonstrates how this method derives factors in a purely statistical or data-driven way with the advantage of not requiring ex-ante knowledge of the behavior of asset returns.
 
### Eigenportfolios

Another application of PCA involves the covariance matrix of the normalized returns. The principal components of the correlation matrix capture most of the covariation among assets in descending order and are mutually uncorrelated. Moreover, we can use standardized principal components as portfolio weights. 

The notebook [pca_and_eigen_portfolios](01_linear_dimensionality_reduction/04_pca_and_eigen_portfolios.ipynb) illustrates how to create Eigenportfolios.

### References

- [Characteristics Are Covariances: A Unified Model of Risk and Return](http://www.nber.org/2018LTAM/kelly.pdf), Kelly, Pruitt and Su, NBER, 2018
- [Statistical Arbitrage in the U.S. Equities Market](https://math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb20090616.pdf), Marco Avellaneda and Jeong-Hyun Lee, 2008

## Independent Component Analysis

Independent component analysis (ICA) is another linear algorithm that identifies a new basis to represent the original data but pursues a different objective than PCA. See [Hyvärinen and Oja](https://www.sciencedirect.com/science/article/pii/S0893608000000265) (2000) for a detailed introduction.
 
ICA emerged in signal processing, and the problem it aims to solve is called blind source separation. It is typically framed as the cocktail party problem where a given number of guests are speaking at the same time so that a single microphone would record overlapping signals. ICA assumes there are as many different microphones as there are speakers, each placed at different locations so that it records a different mix of the signals. ICA then aims to recover the individual signals from the different recordings.

- [Independent Component Analysis: Algorithms and Applications](https://www.sciencedirect.com/science/article/pii/S0893608000000265), Aapo Hyvärinen and Erkki Oja, Neural Networks, 2000
- [Independent Components Analysis](http://cs229.stanford.edu/notes/cs229-notes11.pdf), CS229 Lecture Notes, Andrew Ng
- [Common factors in prices, order flows, and liquidity](https://www.sciencedirect.com/science/article/pii/S0304405X0000091X), Hasbrouck and Seppi, Journal of Financial Economics, 2001
- [Volatility Modelling of Multivariate Financial Time Series by Using ICA-GARCH Models](https://link.springer.com/chapter/10.1007/11508069_74), Edmond H. C. Wu, Philip L. H. Yu, in: Gallagher M., Hogan J.P., Maire F. (eds) Intelligent Data Engineering and Automated Learning - IDEAL 2005
- [The Prediction Performance of Independent Factor Models](http://www.cs.cuhk.hk/~lwchan/papers/icapred.pdf), Chan, In: proceedings of the 2002 IEEE International Joint Conference on Neural Networks
- [An Overview of Independent Component Analysis and Its Applications](http://www.informatica.si/ojs-2.4.3/index.php/informatica/article/download/334/333), Ganesh R. Naik, Dinesh K Kumar, Informatica 2011

## Manifold Learning

The manifold hypothesis emphasizes that high-dimensional data often lies on or near a lower-dimensional non-linear manifold that is embedded in the higher dimensional space. 

[Manifold learning](https://scikit-learn.org/stable/modules/manifold.html) aims to find the manifold of intrinsic dimensionality and then represent the data in this subspace. A simplified example uses a road as one-dimensional manifolds in a three-dimensional space and identifies data points using house numbers as local coordinates.

### Code Example: what a manifold looks like 

The notebook [manifold_learning_intro](02_manifold_learning/01_manifold_learning_intro.ipynb) contains several exampoles, including the two-dimensional swiss roll that illustrates the topological structure of manifolds. 

### Code Example: Local Linear Embedding

Several techniques approximate a lower dimensional manifold. One example is [locally-linear embedding](https://cs.nyu.edu/~roweis/lle/) (LLE) that was developed in 2000 by Sam Roweis and Lawrence Saul.
 
- The notebook [manifold_learning_lle](02_manifold_learning/02_manifold_learning_lle.ipynb) demonstrates how it ‘unrolls’ the swiss roll. For each data point, LLE identifies a given number of nearest neighbors and computes weights that represent each point as a linear combination of its neighbors. It finds a lower-dimensional embedding by linearly projecting each neighborhood on global internal coordinates on the lower-dimensional manifold and can be thought of as a sequence of PCA applications.

The generic examples use the following datasets:

- [MNIST Data](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)

### References

- [Locally Linear Embedding](https://cs.nyu.edu/~roweis/lle/), Sam T. Roweis and Lawrence K. Saul (LLE author website)

## Code Examples: visualizing high-dimensional image and asset price data with manifold learning

### t-distributed stochastic neighbor embedding (t-SNE)

[t-SNE](https://lvdmaaten.github.io/tsne/) is an award-winning algorithm developed in 2010 by Laurens van der Maaten and Geoff Hinton to detect patterns in high-dimensional data. It takes a probabilistic, non-linear approach to locating data on several different, but related low-dimensional manifolds. The algorithm emphasizes keeping similar points together in low dimensions, as opposed to maintaining the distance between points that are apart in high dimensions, which results from algorithms like PCA that minimize squared distances. 

- [Visualizing Data using t-SNE](http://www.cs.toronto.edu/~hinton/absps/tsne.pdf), van der Maaten, Hinton, Journal of Machine Learning Research, 2008
- [Visualizing Time-Dependent Data Using Dynamic t-SNE](http://www.cs.rug.nl/~alext/PAPERS/EuroVis16/paper.pdf), Rauber, Falcão, Telea, Eurographics Conference on Visualization (EuroVis) 2016
- [t-Distributed Stochastic Neighbor Embedding Wins Merck Viz Challenge](http://blog.kaggle.com/2012/11/02/t-distributed-stochastic-neighbor-embedding-wins-merck-viz-challenge/), Kaggle Blog 2016
- [t-SNE: Google Tech Talk](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw), van der Maaten, 2013
- [Parametric t-SNE](https://github.com/kylemcdonald/Parametric-t-SNE), fast t-SNE implementation using Keras by Kyle McDonald

### UMAP

[UMAP](https://github.com/lmcinnes/umap)) is a more recent algorithm for visualization and general dimensionality reduction. It assumes the data is uniformly distributed on a locally connected manifold and looks for the closest low-dimensional equivalent using fuzzy topology. It uses a neighbors parameter that impacts the result similarly as perplexity above.

It is faster and hence scales better to large datasets than t-SNE, and sometimes preserves global structure than better than t-SNE. It can also work with different distance functions, including, e.g., cosine similarity that is used to measure the distance between word count vectors.

- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426), Leland McInnes, John Healy, 2018

- The notebooks [manifold_learning_tsne_umap](02_manifold_learning/03_manifold_learning_tsne_umap.ipynb) and [manifold_learning_asset_prices](02_manifold_learning/04_manifold_learning_asset_prices.ipynb) demonstrate the usage of both t-SNE and UMAP with various data sets, including equity returns.

## Cluster Algorithms

Both clustering and dimensionality reduction summarize the data. Dimensionality reduction compresses the data by representing it using new, fewer features that capture the most relevant information. Clustering algorithms, in contrast, assign existing observations to subgroups that consist of similar data points.

Clustering can serve to better understand the data through the lens of categories learned from continuous variables. It also permits automatically categorizing new objects according to the learned criteria. Examples of related applications include hierarchical taxonomies, medical diagnostics, or customer segmentation. Alternatively, clusters can be used to represent groups as prototypes, using e.g. the midpoint of a cluster as the best representatives of learned grouping. An example application includes image compression.

Clustering algorithms differ with respect to their strategy of identifying groupings:
- Combinatorial algorithms select the most coherent of different groupings of observations
- Probabilistic modeling estimates distributions that most likely generated the clusters
- Hierarchical clustering finds a sequence of nested clusters that optimizes coherence at any given stage

Algorithms also differ by the notion of what constitutes a useful collection of objects that needs to match the data characteristics, domain and the goal of the applications. Types of groupings include:
- Clearly separated groups of various shapes
- Prototype- or center-based, compact clusters
- Density-based clusters of arbitrary shape
- Connectivity- or graph-based clusters

Important additional aspects of a clustering algorithm include whether 
- it requires exclusive cluster membership, 
- makes hard, i.e., binary, or soft, probabilistic assignment, and 
- is complete and assigns all data points to clusters.

### Code example: comparing cluster algorithms

The notebook [clustering_algos](03_clustering_algorithms/01_clustering_algos.ipynb) compares the clustering results for several algorithm using toy dataset designed to test clustering algorithms.

### Code example: k-Means

k-Means is the most well-known clustering algorithm and was first proposed by Stuart Lloyd at Bell Labs in 1957. 

#### The algorithm

The algorithm finds K centroids and assigns each data point to exactly one cluster with the goal of minimizing the within-cluster variance (called inertia). It typically uses Euclidean distance but other metrics can also be used. k-Means assumes that clusters are spherical and of equal size and ignores the covariance among features.

- The notebook [kmeans_implementation](03_clustering_algorithms/02_kmeans_implementation.ipynb) demonstrates how the k-Means algorithm works.

#### Evaluating the results

Cluster quality metrics help select among alternative clustering results. 

- The notebook [kmeans_evaluation ](03_clustering_algorithms/03_kmeans_evaluation.ipynb) illustrates how to evaluate clustering quality using inertia and the [silhouette score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

### Code example: Hierarchical Clustering

Hierarchical clustering avoids the need to specify a target number of clusters because it assumes that data can successively be merged into increasingly dissimilar clusters. It does not pursue a global objective but decides incrementally how to produce a sequence of nested clusters that range from a single cluster to clusters consisting of the individual data points.

While hierarchical clustering does not have hyperparameters like k-Means, the measure of dissimilarity between clusters (as opposed to individual data points) has an important impact on the clustering result. The options differ as follows:

- Single-link: distance between nearest neighbors of two clusters
- Complete link: maximum distance between respective cluster members
- Group average
- Ward’s method: minimize within-cluster variance

The notebook [hierarchical_clustering](03_clustering_algorithms/04_hierarchical_clustering.ipynb) demonstrates how this algorithm works, and how to visualize and evaluate the results.  

### Code example: Density-Based Clustering

Density-based clustering algorithms assign cluster membership based on proximity to other cluster members. They pursue the goal of identifying dense regions of arbitrary shapes and sizes. They do not require the specification of a certain number of clusters but instead rely on parameters that define the size of a neighborhood and a density threshold.

The notebook [density_based_clustering](03_clustering_algorithms/05_density_based_clustering.ipynb) demonstrates how DBSCAN and hierarchical DBSCAN work.

- [Pairs Trading with density-based clustering and cointegration](https://www.quantopian.com/posts/pairs-trading-with-machine-learning)

### Code example: Gaussian Mixture Models

Gaussian mixture models (GMM) are a generative model that assumes the data has been generated by a mix of various multivariate normal distributions. The algorithm aims to estimate the mean & covariance matrices of these distributions.

It generalizes the k-Means algorithm: it adds covariance among features so that clusters can be ellipsoids rather than spheres, while the centroids are represented by the means of each distribution. The GMM algorithm performs soft assignments because each point has a probability to be a member of any cluster. 

The notebook [gaussian_mixture_models](03_clustering_algorithms/06_gaussian_mixture_models.ipynb) demonstrates the application of a GMM clustering model.

### Code example: Hierarchical Risk Parity

The key idea of hierarchical risk parity (HRP) is to use hierarchical clustering on the covariance matrix to be able to group assets with similar correlations together and reduce the number of degrees of freedom by only considering 'similar' assets as substitutes when constructing the portfolio. 

#### The algorithm

The notebook [hierarchical_risk_parity](04_hierarchical_risk_parity/01_hierarchical_risk_parity.ipynb) in the subfolder [hierarchical_risk_parity](04_hierarchical_risk_parity) illustrate its application. 

#### Backtest comparison with alternatives

The notebook [pf_optimization_with_hrp_zipline_benchmark](04_hierarchical_risk_parity/02_pf_optimization_with_hrp_zipline_benchmark.ipynb) in the subfolder [hierarchical_risk_parity](04_hierarchical_risk_parity) compares HRP with other portfolio optimization methods discussed in [Chapter 5](../05_strategy_evaluation). 

### References

- [Building Diversified Portfolios that Outperform Out-of-Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678), Lopez de Prado, Journal of Portfolio Management, 2015
- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Raffinot 2016
- [Visualizing the Stock Market Structure](https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html)



