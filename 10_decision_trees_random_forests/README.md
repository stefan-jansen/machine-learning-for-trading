# Chapter 10: Decision Trees & Random Forests

This chapter introduces decision trees and random forests. In short, it covers the following:
- How to use decision trees for regression and classification
- How to gain insights from decision trees and visualize the decision rules learned from the data
- Why ensemble models tend to deliver superior results
- How bootstrap aggregation addresses the overfitting challenges of decision trees
- How to train, tune, and interpret random forests

## Decision trees for regression and classification

Decision trees are a machine learning algorithm that predicts the value of a target variable based on decision rules learned from data. The algorithm can be applied to regression and classification problems by changing the objective that governs how the algorithm learns the rules.

We will discuss how decision trees use rules to make predictions, how to train them to predict (continuous) returns as well as (categorical) directions of price movements, and how to interpret, visualize, and tune them effectively.

### How to use decision trees in practice

In this section, we illustrate how to use tree-based models to gain insight and make predictions. To demonstrate regression trees we predict returns, and for the classification case, we return to the example of positive and negative asset price moves. The code examples for this section are in the notebook [decision_trees](01_decision_trees.ipynb) unless otherwise noted.

#### How to prepare the data

We use a simplified version of the data set constructed in [Chapter 4, Alpha factor research](../04_alpha_factor_research). It consists of daily stock prices provided by Quandl for the 2010-2017 period and various engineered features. The details can be found in the [data_prep](00_data_prep.ipynb) notebook. 

The decision tree models in this chapter are not equipped to handle missing or categorical variables, so we will apply dummy encoding to the latter after dropping any of the former.

#### How to build a regression tree

Regression trees make predictions based on the mean outcome value for the training samples assigned to a given node and typically rely on the mean-squared error to select optimal rules during recursive binary splitting.

#### How to build a classification tree

A classification tree works just like the regression version, except that categorical nature of the outcome requires a different approach to making predictions and measuring the loss. While a regression tree predicts the response for an observation assigned to a leaf node using the mean outcome of the associated training samples, a classification tree instead uses the mode, that is, the most common class among the training samples in the relevant region. A classification tree can also generate probabilistic predictions based on relative class frequencies.

#### How to visualize a decision tree

You can visualize the tree using the [graphviz](https://graphviz.gitlab.io/download/) library because sklearn can output a description of the tree using the .dot language used by that library. You can configure the output to include feature and class labels and limit the number of levels to keep the chart readable, as follows:

### Overfitting and regularization

Decision trees have a strong tendency to overfit, especially when a dataset has a large number of features relative to the number of samples. The notebook [decision_trees](01_decision_trees.ipynb) explains relevant regularization hyperparameters and illustrates their use.

### How to tune the hyperparameters

The notebook also demonstrates the use of cross-validation including `sklearn`'s [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) class for exhaustive search over hyperparameter combinations.

## Random forests

Decision trees are not only useful for their transparency and interpretability but are also fundamental building blocks for much more powerful ensemble models that combine many individual trees with strategies to randomly vary their design to address the overfitting and high variance problems discussed in the preceding section.


### Ensemble models

Ensemble learning involves combining several machine learning models into a single new model that aims to make better predictions than any individual model. More specifically, an ensemble integrates the predictions of several base estimators trained using one or more given learning algorithms to reduce the generalization error that these models may produce on their own.

### How bagging lowers model variance

Bagging refers to the aggregation of bootstrap samples, which are random samples with replacement. Such a random sample has the same number of observations as the original dataset but may contain duplicates due to replacement. 

Bagging reduces the variance of the base estimators by randomizing how, for example, each tree is grown and then averages the predictions to reduce their generalization error. It is often a straightforward approach to improve on a given model without the need to change the underlying algorithm. It works best with complex models that have low bias and high variance, such as deep decision trees, because its goal is to limit overfitting. Boosting methods, in contrast, work best with weak models, such as shallow decision trees.

### How to build a random forest

The random forest algorithm expands on the randomization introduced by the bootstrap samples generated by bagging to reduce variance further and improve predictive performance.
In addition to training each ensemble member on bootstrapped training data, random forests also randomly sample from the features used in the model (without replacement). Depending on the implementation, the random samples can be drawn for each tree or each split. As a result, the algorithm faces different options when learning new rules, either at the level of a tree or for each split.

### How to train and tune a random forest

The notebook [random_forest](02_random_forest.ipynb) contains implementation details for this section.