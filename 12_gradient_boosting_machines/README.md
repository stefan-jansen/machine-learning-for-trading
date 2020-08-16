# Boosting your Trading Strategy

This chapter explores boosting, another ensemble learning algorithm typically based on decision trees that often produces even better results the [Random Forests](../10_decision_trees_random_forests). 

The key difference is that boosting, in its original AdaBoost version, modifies the training data for each tree based on the cumulative errors made by the model before adding the new tree. Random Forests, in contrast, use bagging to train many trees independently from each other using randomized versions of the training set. While Random Forests can be trained in parallel, boosting proceeds sequentially using reweighted versions of the data. State-of-the-art boosting implementations also adopt the randomization strategies of random forests.

In this chapter, we will see how boosting has evolved into one of the most successful machine learning algorithms over the last three decades. At the time of writing, it has come to dominate machine learning competitions for structured data (as opposed to high-dimensional images or speech, for example, where the relationship between the input and output is more complex, and deep learning excels at). More specifically, in this chapter we will cover the following topics:

## Content

1. [Getting started: adaptive boosting](#getting-started-adaptive-boosting)
    * [The AdaBoost algorithm](#the-adaboost-algorithm)
    * [Code Example: AdaBoost with sklearn](#code-example-adaboost-with-sklearn)
2. [Gradient boosting - ensembles for most tasks ](#gradient-boosting---ensembles-for-most-tasks-)
    * [How to train and tune GBM models](#how-to-train-and-tune-gbm-models)
    * [Code Example: Gradient boosting with scikit-learn](#code-example-gradient-boosting-with-scikit-learn)
3. [Using XGBoost, LightGBM and CatBoost](#using-xgboost-lightgbm-and-catboost)
4. [Code Example: A long-short trading strategy with gradient boosting](#code-example-a-long-short-trading-strategy-with-gradient-boosting)
    * [Preparing the data](#preparing-the-data)
    * [How to generate signals with LightGBM and CatBoost models](#how-to-generate-signals-with-lightgbm-and-catboost-models)
    * [Evaluating the trading signals](#evaluating-the-trading-signals)
    * [Creating out-of-sample predictions](#creating-out-of-sample-predictions)
    * [Defining and backtesting the long-short strategy](#defining-and-backtesting-the-long-short-strategy)
5. [A peek into the black box: How to interpret GBM results](#a-peek-into-the-black-box-how-to-interpret-gbm-results)
    * [Code example: attributing feature importance with LightGBM](#code-example-attributing-feature-importance-with-lightgbm)
        - [Feature importance](#feature-importance)
        - [Partial dependence plots](#partial-dependence-plots)
        - [SHapley Additive exPlanations (SHAP Values)](#shapley-additive-explanations-shap-values)
6. [An intraday strategy with Algoseek and LightGBM](#an-intraday-strategy-with-algoseek-and-lightgbm)
    * [Code example: engineering intraday features](#code-example-engineering-intraday-features)
    * [Code example: tuning a LightGBM model and evaluating the forecasts](#code-example-tuning-a-lightgbm-model-and-evaluating-the-forecasts)
7. [Resources](#resources)
    * [XGBoost](#xgboost)
    * [LightGBM](#lightgbm)
    * [CatBoost](#catboost)


## Getting started: adaptive boosting

Like bagging, boosting combines base learners into an ensemble. Boosting was initially developed for classification problems, but can also be used for regression, and has been called one of the most potent learning ideas introduced in the last 20 years (as described in [Elements of Statistical Learning](http://web.stanford.edu/~hastie/ElemStatLearn/) by Trevor Hastie, et al). Like bagging, it is a general method or metamethod that can be applied to many statistical learning models.

The following sections briefly introduce AdaBoost and then focus on the gradient boosting model, as well as several state-of-the-art implementations of this algorithm. 

### The AdaBoost algorithm

AdaBoost is a significant departure from bagging, which builds ensembles on very deep trees to reduce bias. AdaBoost, in contrast, grows shallow trees as weak learners, often producing superior accuracy with stumps—that is, trees formed by a single split. The algorithm starts with an equal-weighted training set and then successively alters the sample distribution. After each iteration, AdaBoost increases the weights of incorrectly classified observations and reduces the weights of correctly predicted samples so that subsequent weak learners focus more on particularly difficult cases. Once trained, the new decision tree is incorporated into the ensemble with a weight that reflects its contribution to reducing the training error.

- [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf), Y. Freund, R. Schapire, 1997.

### Code Example: AdaBoost with sklearn

As part of its ensemble module, sklearn provides an [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) implementation that supports two or more classes. 

The code examples for this section are in the notebook [gbm_baseline](01_gbm_baseline.ipynb) that compares the performance of various algorithms with a dummy classifier that always predicts the most frequent class.

The algorithms in this chapter use a dataset generated by the notebook [feature-engineering](../04_alpha_factor_research/00_data/feature_engineering.ipynb) from [Chapter 4 on Alpha Factor Research](../04_alpha_factor_research) that needs to be executed first.

- `sklearn` AdaBoost [docs](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

## Gradient boosting - ensembles for most tasks 

The main idea behind the resulting Gradient Boosting Machines (GBM) algorithm is the training of the base learners to learn the negative gradient of the current loss function of the ensemble. As a result, each addition to the ensemble directly contributes to reducing the overall training error given the errors made by prior ensemble members. Since each new member represents a new function of the data, gradient boosting is also said to optimize over the functions hm in an additive fashion. 

- [Greedy function approximation: A gradient boosting machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf), Jerome H. Friedman, 1999

### How to train and tune GBM models

The two key drivers of gradient boosting performance are the size of the ensemble and the complexity of its constituent decision trees. The control of complexity for decision trees aims to avoid learning highly specific rules that typically imply a very small number of samples in leaf nodes. We covered the most effective constraints used to limit the ability of a decision tree to overfit to the training data in [Chapter 4 on Decision Trees and Random Forests](../10_decision_trees_random_forests).

In addition to directly controlling the size of the ensemble, there are various regularization techniques, such as shrinkage, that we encountered in the context of the Ridge and Lasso linear regression models in [Chapter 7, Linear Models – Regression and Classification](../07_linear_models). Furthermore, the randomization techniques used in the context of random forests are also commonly applied to gradient boosting machines.

### Code Example: Gradient boosting with scikit-learn

The ensemble module of sklearn contains an implementation of gradient boosting trees for regression and classification, both binary and multiclass.

The notebook [boosting_baseline](./01_boosting_baseline.ipynb) demonstrates how to run cross-validation for the [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

The notebook [sklearn_gbm_tuning](02_sklearn_gbm_tuning.ipynb) shows how to [GridSearchCV]() to search for the best set of parameters. This can be very time-consuming to run. 

The notebook [sklearn_gbm_tuning_results](03_sklearn_gbm_tuning_results.ipynb) displays some of the results that can be obtained.

- `scikit-klearn` Gradient Boosting [docs](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)

## Using XGBoost, LightGBM and CatBoost

Over the last few years, several new gradient boosting implementations have used various innovations that accelerate training, improve resource efficiency, and allow the algorithm to scale to very large datasets. The new implementations and their sources are as follows:
- [XGBoost](https://github.com/dmlc/xgboost) (extreme gradient boosting), started in 2014 by Tianqi Chen at the University of Washington 
- [LightGBM](https://github.com/Microsoft/LightGBM), first released in January 2017, by Microsoft
- [CatBoost](https://tech.yandex.com/catboost/), first released in April 2017 by Yandex

The book reviews the numerous algorithmic innovations that have emerged over time and subsequently converged (so that most features are available for all implementations) before illustrating their implementation.



## Code Example: A long-short trading strategy with gradient boosting

In this section, we’ll design, implement, and evaluate a trading strategy for US equities driven by daily return forecasts produced by gradient boosting models. 

As in the previous examples, we’ll lay out a framework and build a specific example that you can adapt to run your own experiments. There are numerous aspects that you can vary, from the asset class and investment universe to more granular aspects like the features, holding period, or trading rules. See, for example, the Alpha Factor Library in the Appendix for numerous additional features.

### Preparing the data

We use the Quandl Wiki data to engineer a few simple features (see notebook [preparing_the_model_data](04_preparing_the_model_data.ipynb) for details) and select a model using 2015/16 as validation period and run an out-of-sample test for 2017.

### How to generate signals with LightGBM and CatBoost models

We’ll keep the trading strategy simple and only use a single machine learning (ML) signal; a real-life application will likely use multiple signals from different sources, such as complementary ML models trained on different datasets or with different lookahead or lookback periods. It would also use sophisticated risk management from simple stop-loss to value-at-risk analysis.

XGBoost, LightGBM, and CatBoost offer interfaces for multiple languages, including Python, and have both a sklearn interface that is compatible with other sklearn features, such as GridSearchCV and their own methods to train and predict gradient boosting models. The notebook [gbm_baseline](01_gbm_baseline.ipynb) illustrates the use of the sklearn interface for each implementation. The library methods are often better documented and are also easy to use, so we'll use them to illustrate the use of these models.

The process entails the creation of library-specific data formats, the tuning of various hyperparameters, and the evaluation of results that we will describe in the following sections. 

- The notebook [trading_signals_with_lightgbm_and_catboost](05_trading_signals_with_lightgbm_and_catboost.ipynb) cross-validates a range of hyperparameter options to optimize the models' predictive performance.

### Evaluating the trading signals

The notebook [evaluate_trading_signals](06_evaluate_trading_signals.ipynb) demonstrates how to evaluate the trading signals.

### Creating out-of-sample predictions

The notebook [making_out_of_sample_predictions](08_making_out_of_sample_predictions.ipynb) shows how to create predictions for the best-performing models.

### Defining and backtesting the long-short strategy

The notebook [backtesting_with_zipline](09_backtesting_with_zipline.ipynb) creates a strategy based on the model predictions, simulates its historical performance using [Zipline](https://www.zipline.io/, and evaluates the result using [Pyfolio](https://github.com/quantopian/pyfolio. 

## A peek into the black box: How to interpret GBM results

Understanding why a model predicts a certain outcome is very important for several reasons, including trust, actionability, accountability, and debugging. Insights into the nonlinear relationship between features and the outcome uncovered by the model, as well as interactions among features, are also of value when the goal is to learn more about the underlying drivers of the phenomenon under study.

### Code example: attributing feature importance with LightGBM

The notebook [model_interpretation](06_model_interpretation.ipynb) illustrates the following methods.

#### Feature importance

There are three primary ways to compute global feature importance values:
- Gain: This classic approach introduced by Leo Breiman in 1984 uses the total reduction of loss or impurity contributed by all splits for a given feature. The motivation is largely heuristic, but it is a commonly used method to select features.
- Split count: This is an alternative approach that counts how often a feature is used to make a split decision, based on the selection of features for this purpose based on the resultant information gain.
- Permutation: This approach randomly permutes the feature values in a test set and measures how much the model's error changes, assuming that an important feature should create a large increase in the prediction error. Different permutation choices lead to alternative implementations of this basic approach.

#### Partial dependence plots

In addition to the summary contribution of individual features to the model's prediction, partial dependence plots visualize the relationship between the target variable and a set of features. The nonlinear nature of gradient boosting trees causes this relationship to depend on the values of all other features.

#### SHapley Additive exPlanations (SHAP Values)

At the 2017 NIPS conference, Scott Lundberg and Su-In Lee from the University of Washington presented a new and more accurate approach to explaining the contribution of individual features to the output of tree ensemble models called [SHapley Additive exPlanations](https://github.com/slundberg/shap), or SHAP values.

This new algorithm departs from the observation that feature-attribution methods for tree ensembles, such as the ones we looked at earlier, are inconsistent — that is, a change in a model that increases the impact of a feature on the output can lower the importance values for this feature.

SHAP values unify ideas from collaborative game theory and local explanations, and have been shown to be theoretically optimal, consistent, and locally accurate based on expectations. Most importantly, Lundberg and Lee have developed an algorithm that manages to reduce the complexity of computing these model-agnostic, additive feature-attribution methods from O(TLDM) to O(TLD2), where T and M are the number of trees and features, respectively, and D and L are the maximum depth and number of leaves across the trees. 

This important innovation permits the explanation of predictions from previously intractable models with thousands of trees and features in a fraction of a second. An open source implementation became available in late 2017 and is compatible with XGBoost, LightGBM, CatBoost, and sklearn tree models. 

Shapley values originated in game theory as a technique for assigning a value to each player in a collaborative game that reflects their contribution to the team's success. SHAP values are an adaptation of the game theory concept to tree-based models and are calculated for each feature and each sample. They measure how a feature contributes to the model output for a given observation. For this reason, SHAP values provide differentiated insights into how the impact of a feature varies across samples, which is important given the role of interaction effects in these nonlinear models.

SHAP values provide granular feature attribution at the level of each individual prediction, and enable much richer inspection of complex models through (interactive) visualization. The SHAP summary scatterplot displayed at the beginning of this section offers much more differentiated insights than a global feature-importance bar chart. Force plots of individual clustered predictions allow for more detailed analysis, while SHAP dependence plots capture interaction effects and, as a result, provide more accurate and detailed results than partial dependence plots.

## An intraday strategy with Algoseek and LightGBM

This section and the notebooks will be updated once Algoseek makes the sample data available.

### Code example: engineering intraday features

The notebook [intraday_features](10_intraday_features.ipynb) creates features from minute-bar trade and quote data.

### Code example: tuning a LightGBM model and evaluating the forecasts

The notebook [intraday_model](11_intraday_model.ipynb) optimizes a LightGBM boosting model, generates out-of-sample predictions, and evaluates the result.

## Resources

- [xgboost - LightGBM Parameter Comparison](https://sites.google.com/view/lauraepp/parameters)
- [xgboost vs LightGBM Benchmarks](https://sites.google.com/view/lauraepp/new-benchmarks)
- [Depth- vs Leaf-wise growth](https://datascience.stackexchange.com/questions/26699/decision-trees-leaf-wise-best-first-and-level-wise-tree-traverse)
- [Rashmi Korlakai Vinayak, Ran Gilad-Bachrach. “DART: Dropouts meet Multiple Additive Regression Trees.”](http://www.jmlr.org/proceedings/papers/v38/korlakaivinayak15.pdf)

### XGBoost

- [GitHub Repo](https://github.com/dmlc/xgboost)
- [Documentation](https://xgboost.readthedocs.io)
- [Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Accelerating the XGBoost algorithm using GPU computing. Mitchell R, Frank E., 2017, PeerJ Computer Science 3:e127](https://peerj.com/articles/cs-127/)
- [XGBoost: Scalable GPU Accelerated Learning, Rory Mitchell, Andrey Adinets, Thejaswi Rao, 2018](http://arxiv.org/abs/1806.11248)
- [Nvidia Parallel Forall: Gradient Boosting, Decision Trees and XGBoost with CUDA, Rory Mitchell, 2017](https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/)
- [Awesome XGBoost](https://github.com/dmlc/xgboost/tree/master/demo)

### LightGBM

- [GitHub Repo](https://github.com/Microsoft/LightGBM)
- [Documentation](https://lightgbm.readthedocs.io/en/latest/index.html)
- [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- [On Grouping for Maximum Homogeneity](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.1958.10501479#.W8_3pXX24UE)

### CatBoost

- [Python API](https://tech.yandex.com/catboost/doc/dg/concepts/python-quickstart-docpage/)
- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
- [CatBoost: gradient boosting with categorical features](http://learningsys.org/nips17/assets/papers/paper_11.pdf)