# Random Forests - A Long-Short Strategy for Japanese Stocks

In this chapter, we will learn how to use two new classes of machine learning models for trading: decision trees and random forests. We will see how decision trees learn rules from data that encode non-linear relationships between the input and the output variables. We will illustrate how to train a decision tree and use it for prediction for regression and classification problems such as asset returns and price moves. We will also visualize and interpret the rules learned by the model, and tune the model's hyperparameters to optimize the bias-variance tradeoff and prevent overfitting. 

Decision trees are not only important standalone models but are also frequently used as components in other models. In the second part of this chapter, we will introduce ensemble models that combine multiple individual models to produce a single aggregate prediction with lower prediction-error variance. 

We will illustrate bootstrap aggregation, often called bagging, as one of several methods to randomize the construction of individual models and reduce the correlation of the prediction errors made by an ensemble's components. We will illustrate how bagging effectively reduces the variance, and learn how to configure, train, and tune random forests. We will see how random forests as an ensemble of a large number of decision trees, can dramatically reduce prediction errors, at the expense of some loss in interpretation. 

Then we will proceed and build a long-short trading strategy that uses a Random Forest ensemble to generate profitable signals for large-cap Japanese equities over the last three years. We will source and prepare the stock price data, tune the hyperparameters of a Random Forest model, and backtest trading rules based on the models’ signals. The resulting long-short strategy uses machine learning rather than the cointegration relationship we saw in Chapter 9 on Time Series models to identify and trade baskets of securities whose prices will likely move in opposite directions over a given investment horizon.

## Content

1. [Decision trees: Learning rules from data](#decision-trees-learning-rules-from-data)
2. [Code Example: Decision trees in practice](#code-example-decision-trees-in-practice)
    * [The data: monthly stock returns and features](#the-data-monthly-stock-returns-and-features)
    * [Building a regression tree with time-series data](#building-a-regression-tree-with-time-series-data)
    * [Building a classification tree](#building-a-classification-tree)
    * [Visualizing a decision tree](#visualizing-a-decision-tree)
    * [Overfitting and regularization](#overfitting-and-regularization)
    * [How to tune the hyperparameters](#how-to-tune-the-hyperparameters)
3. [Random forests: Better predictions with ensembles](#random-forests-better-predictions-with-ensembles)
    * [Why ensemble models perform better](#why-ensemble-models-perform-better)
    * [Code Example: How bagging lowers model variance](#code-example-how-bagging-lowers-model-variance)
    * [Code Example: How to train and tune a random forest](#code-example-how-to-train-and-tune-a-random-forest)
4. [Code Example: Long-short signals for Japanese stocks with LightGBM](#code-example-long-short-signals-for-japanese-stocks-with-lightgbm)
    * [Custom Zipline Bundle](#custom-zipline-bundle)
    * [Feature Engineering](#feature-engineering)
    * [LightGBM Random Forest Model Tuning](#lightgbm-random-forest-model-tuning)
    * [Signal Evaluation with Alphalens](#signal-evaluation-with-alphalens)
    * [Backtest with Zipline](#backtest-with-zipline)

## Decision trees: Learning rules from data

Decision trees are a machine learning algorithm that predicts the value of a target variable based on decision rules learned from data. The algorithm can be applied to regression and classification problems by changing the objective that governs how the algorithm learns the rules.

We will discuss how decision trees use rules to make predictions, how to train them to predict (continuous) returns as well as (categorical) directions of price movements, and how to interpret, visualize, and tune them effectively.

## Code Example: Decision trees in practice

The notebook [decision_trees](01_decision_trees.ipynb) illustrates how to use tree-based models to gain insight and make predictions. We'll predict returns to demonstrate how to use regression trees, and positive or negative asset price moves for the classification case. 

### The data: monthly stock returns and features

We use a variation of the data set constructed in [Chapter 4, Alpha factor research](../04_alpha_factor_research). It consists of daily stock prices provided by Quandl for the 2010-2017 period and various engineered features. 
- The details can be found in the [data_prep](00_data_prep.ipynb) notebook. 

### Building a regression tree with time-series data

Regression trees make predictions based on the mean outcome value for the training samples assigned to a given node and typically rely on the mean-squared error to select optimal rules during recursive binary splitting.

### Building a classification tree

A classification tree works just like the regression version, except that categorical nature of the outcome requires a different approach to making predictions and measuring the loss. While a regression tree predicts the response for an observation assigned to a leaf node using the mean outcome of the associated training samples, a classification tree instead uses the mode, that is, the most common class among the training samples in the relevant region. A classification tree can also generate probabilistic predictions based on relative class frequencies.

### Visualizing a decision tree

You can visualize the tree using the [graphviz](https://graphviz.gitlab.io/download/) library because sklearn can output a description of the tree using the .dot language used by that library. You can configure the output to include feature and class labels and limit the number of levels to keep the chart readable, as follows:

### Overfitting and regularization

Decision trees have a strong tendency to overfit, especially when a dataset has a large number of features relative to the number of samples. The notebook [decision_trees](01_decision_trees.ipynb) explains relevant regularization hyperparameters and illustrates their use.

### How to tune the hyperparameters

The notebook also demonstrates the use of cross-validation including `sklearn`'s [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) class for exhaustive search over hyperparameter combinations.

## Random forests: Better predictions with ensembles

Decision trees are not only useful for their transparency and interpretability but are also fundamental building blocks for much more powerful ensemble models that combine many individual trees with strategies to randomly vary their design to address the overfitting and high variance problems discussed in the preceding section.

### Why ensemble models perform better

Ensemble learning involves combining several machine learning models into a single new model that aims to make better predictions than any individual model. More specifically, an ensemble integrates the predictions of several base estimators trained using one or more given learning algorithms to reduce the generalization error that these models may produce on their own.

### Code Example: How bagging lowers model variance

Bagging refers to the aggregation of bootstrap samples, which are random samples with replacement. Such a random sample has the same number of observations as the original dataset but may contain duplicates due to replacement. 

Bagging reduces the variance of the base estimators by randomizing how, for example, each tree is grown and then averages the predictions to reduce their generalization error. It is often a straightforward approach to improve on a given model without the need to change the underlying algorithm. It works best with complex models that have low bias and high variance, such as deep decision trees, because its goal is to limit overfitting. Boosting methods, in contrast, work best with weak models, such as shallow decision trees.

The notebook [bagged_decision_trees](02_bagged_decision_trees.ipynb) demonstrates the bias-variance tradeoff and how bagging reduce variance compared to an individual decision tree.

### Code Example: How to train and tune a random forest

The random forest algorithm expands on the randomization introduced by the bootstrap samples generated by bagging to reduce variance further and improve predictive performance.

In addition to training each ensemble member on bootstrapped training data, random forests also randomly sample from the features used in the model (without replacement). Depending on the implementation, the random samples can be drawn for each tree or each split. As a result, the algorithm faces different options when learning new rules, either at the level of a tree or for each split.

The notebook [random_forest_tuning](03_random_forest_tuning.ipynb) contains implementation details for this section.

## Code Example: Long-short signals for Japanese stocks with LightGBM

In [Chapter 9](../09, we used cointegration tests to identify pairs of stocks with a long-term equilibrium relationship in the form of a common trend to which their prices revert. 

In this chapter, we will use the predictions of a machine learning model to identify assets that are likely to go up or down so that we can enter market-neutral long and short positions accordingly. The approach is similar to our initial trading strategy that used linear regression in Chapter 7, Linear Models, and Chapter 8, Strategy Workflow: End-to-End Algo Trading.

Instead of the scikit-learn random forest implementation, we will use the [LightGBM](https://lightgbm.readthedocs.io/en/latest/) package that has been primarily designed for gradient boosting. One of several advantages is LightGBM’s ability to efficiently encode categorical variables as numeric features rather than using one-hot dummy encoding (Fisher 1958). We’ll provide a more detailed introduction in the next chapter, but the code samples should be easy to follow as the logic is similar to the scikit-learn version.

### Custom Zipline Bundle

- The directory [custom_bundle](00_custom_bundle) contains instruction on how to obtain the data and create a custom Zipline bundle.

### Feature Engineering

- The notebook [japanese_equity_features](04_japanese_equity_features.ipynb) shows how to generate model features.

### LightGBM Random Forest Model Tuning

- The notebook [random_forest_return_signals](05_random_forest_return_signals.ipynb) contains the code to train and tune a [LightGBM](https://lightgbm.readthedocs.io/en/latest/) random forest model

### Signal Evaluation with Alphalens

- The notebook [alphalens_signals_quality](06_alphalens_signals_quality.ipynb) shows how to evaluate the model predictions using [Alphalens](https://github.com/quantopian/alphalens).

### Backtest with Zipline

- The notebook [backtesting_with_zipline](07_backtesting_with_zipline.ipynb) evaluates the model predictions using a long-short strategy simulated using [Zipline](https://www.zipline.io/).

 
