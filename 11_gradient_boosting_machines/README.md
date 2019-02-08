# Chapter 10: Gradient Boosting Machines

## Adaptive Boosting
### The AdaBoost Algorithm
- [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf), Y. Freund, R. Schapire, 1997.

### AdaBoost with sklearn

- `sklearn` AdaBoost [docs](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

#### Code Examples

The notebook `gbm_baseline` containts the code for this section.

## Gradient Boosting Machines

- [Greedy function approximation: A gradient boosting machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf), Jerome H. Friedman, 1999

### How to train and tune GBM

### Gradient Boosting with sklearn

- `scikit-klearn` Gradient Boosting [docs](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)

#### Code Examples

The notebook `gbm_baseline` containts the code for this section.


- [Rashmi Korlakai Vinayak, Ran Gilad-Bachrach. “DART: Dropouts meet Multiple Additive Regression Trees.”](http://www.jmlr.org/proceedings/papers/v38/korlakaivinayak15.pdf)

## Fast, scalable GBM implementations

- [xgboost - LightGBM Parameter Comparison](https://sites.google.com/view/lauraepp/parameters)
- [xgboost vs LightGBM Benchmarks](https://sites.google.com/view/lauraepp/new-benchmarks)
- [Depth- vs Leaf-wise growth](https://datascience.stackexchange.com/questions/26699/decision-trees-leaf-wise-best-first-and-level-wise-tree-traverse)

#### sklearn


#### XGBoost

- [GitHub Repo](https://github.com/dmlc/xgboost)
- [Documentation](https://xgboost.readthedocs.io)
- [Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Accelerating the XGBoost algorithm using GPU computing. Mitchell R, Frank E., 2017, PeerJ Computer Science 3:e127](https://peerj.com/articles/cs-127/)
- [XGBoost: Scalable GPU Accelerated Learning, Rory Mitchell, Andrey Adinets, Thejaswi Rao, 2018](http://arxiv.org/abs/1806.11248)
- [Nvidia Parallel Forall: Gradient Boosting, Decision Trees and XGBoost with CUDA, Rory Mitchell, 2017](https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/)
- [Awesome XGBoost](https://github.com/dmlc/xgboost/tree/master/demo)

#### LightGBM

- [GitHub Repo](https://github.com/Microsoft/LightGBM)
- [Documentation](https://lightgbm.readthedocs.io/en/latest/index.html)
- [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- [On Grouping for Maximum Homogeneity](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.1958.10501479#.W8_3pXX24UE)

#### CatBoost

- [Python API](https://tech.yandex.com/catboost/doc/dg/concepts/python-quickstart-docpage/)
- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
- [CatBoost: gradient boosting with categorical features](http://learningsys.org/nips17/assets/papers/paper_11.pdf)