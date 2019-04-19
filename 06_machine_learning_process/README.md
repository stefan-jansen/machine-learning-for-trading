# Chapter 06: Machine Learning

In this introductory chapter, we will start to illustrate how you can use a broad range of supervised and unsupervised machine learning (ML) models for algorithmic trading. 

 We cover aspects that apply across model categories so that we can focus on model-specific usage in the following chapters. These aspects include the goal of learning a functional relationship from data by optimizing an objective or loss function. They also include the closely related methods of measuring model performance.
 
 We distinguish between unsupervised and supervised learning, and supervised regression and classification problems. We also outline use cases for algorithmic trading.  

## Learning from Data

## The Machine Learning Workflow

Developing an ML solution requires a systematic approach to maximize the chances of success while proceeding efficiently. It is also important to make the process transparent and replicable to facilitate collaboration, maintenance, and subsequent refinements.

The process is iterative throughout, and the effort at different stages will vary according to the project. Nonethelesee, this process should generally include the following steps:

1. Frame the problem, identify a target metric, and define success
2. Source, clean, and validate the data
3. Understand your data and generate informative features
4. Pick one or more machine learning algorithms suitable for your data
5. Train, test, and tune your models
6. Use your model to solve the original problem


### Basic Walkthrough: K-nearest neighbors

The notebook [machine_learning_workflow](01_machine_learning_workflow.ipynb) contains several examples that illustrate the machine learning workflow using a simple dataset of house prices.

- sklearn [Documentation](http://scikit-learn.org/stable/documentation.html)

### Frame the problem: goals & metrics

The starting point for any machine learning exercise is the ultimate use case it aims to address. Sometimes, this goal will be statistical inference in order to identify an association between variables or even a causal relationship. Most frequently, however, the goal will be the direct prediction of an outcome to yield a trading signal.

### Collect & prepare the data

We addressed the sourcing of market and fundamental data in [Chapter 2](../02_market_and_fundamental_data), and for alternative data in [Chapter 3](../03_alternative_data). We will continue to work with various examples of these sources as we illustrate the application of the various models in later chapters. 

### How to explore, extract and engineer features

Understanding the distribution of individual variables and the relationships among outcomes and features is the basis for picking a suitable algorithm. This typically starts with visualizations such as scatter plots, as illustrated in the companion notebook (and shown in the following image), but also includes numerical evaluations ranging from linear metrics, such as the correlation, to nonlinear statistics, such as the Spearman rank correlation coefficient that we encountered when we introduced the information coefficient. It also includes information-theoretic measures, such as mutual information

#### Code Example: Mutual Information

The notebook [mutual_information](02_mutual_information.ipynb) applies information theory to the financial data we created in the notebook [feature_engineering](../04_alpha_factor_research/00_data/feature_engineering.ipynb), in the chapter [Alpha Factors – Research and Evaluation]((../04_alpha_factor_research).

### Select an ML algorithm

The remainder of this book will introduce several model families, ranging from linear models, which make fairly strong assumptions about the nature of the functional relationship between input and output variables, to deep neural networks, which make very few assumptions.

### Design and tune the model

The ML process includes steps to diagnose and manage model complexity based on estimates of the model's generalization error. An unbiased estimate requires a statistically sound and efficient procedure, as well as error metrics that align with the output variable type, which also determines whether we are dealing with a regression, classification, or ranking problem.

#### Bias-Variance Trade-Off

The errors that an ML model makes when predicting outcomes for new input data can be broken down into reducible and irreducible parts. The irreducible part is due to random variation (noise) in the data that is not measured, such as relevant but missing variables or natural variation. 

The notebook [bias_variance](03_bias_variance.ipynb) demonstrates overfitting by approximating a cosine function using increasingly complex polynomials and measuring the in-sample error.  It draws 10 random samples with some added noise (n = 30) to learn a polynomial of varying complexity. Each time, the model predicts new data points and we capture the mean-squared error for these predictions, as well as the standard deviation of these errors.

It goes on to illustrate the impact of overfitting versus underfitting by trying to learn a Taylor series approximation of the cosine function of ninth degree with some added noise. In the following diagram, we draw random samples of the true function and fit polynomials that underfit, overfit, and provide an approximately correct degree of flexibility.

### How to use cross-validation for model selection

When several candidate models (that is, algorithms) are available for your use case, the act of choosing one of them is called the model selection problem. Model selection aims to identify the model that will produce the lowest prediction error given new data.

#### How to implement cross-validation in Python

 The script [cross_validation](04_cross_validation.py) illustrates various options for splitting data into training and test sets by showing how the indices of a mock dataset with ten observations are assigned to the train and test set.
 
### Parameter tuning with scikit-learn

Model selection typically involves repeated cross-validation of the out-of-sample performance of models using different algorithms (such as linear regression and random forest) or different configurations. Different configurations may involve changes to hyperparameters or the inclusion or exclusion of different variables.

#### Learning and Validation curves with yellowbricks

The notebook [machine_learning_workflow](01_machine_learning_workflow.ipynb)) demonstrates the use of learning and validation  illustrates the use of various model selection techniques. 

- Yellowbrick: Machine Learning Visualization [docs](http://www.scikit-yb.org/en/latest/)

#### Parameter tuning using GridSearchCV and pipeline

Since hyperparameter tuning is a key ingredient of the machine learning workflow, there are tools to automate this process. The sklearn library includes a GridSearchCV interface that cross-validates all combinations of parameters in parallel, captures the result, and automatically trains the model using the parameter setting that performed best during cross-validation on the full dataset.

In practice, the training and validation sets often require some processing prior to cross-validation. Scikit-learn offers the Pipeline to also automate any requisite feature-processing steps in the automated hyperparameter tuning facilitated by GridSearchCV.

The implementation examples in the included machine_learning_workflow.ipynb notebook to see these tools in action.

The notebook [machine_learning_workflow](01_machine_learning_workflow.ipynb)) also the use of these tools.

### Challenges with cross-validation in finance

A key assumption for the cross-validation methods discussed so far is the independent and identical (iid) distribution of the samples available for training.
For financial data, this is often not the case. On the contrary, financial data is neither independently nor identically distributed because of serial correlation and time-varying standard deviation, also known as heteroskedasticity

#### Purging, embargoing, and combinatorial CV

For financial data, labels are often derived from overlapping data points as returns are computed from prices in multiple periods. In the context of trading strategies, the results of a model's prediction, which may imply taking a position in an asset, may only be known later, when this decision is evaluated—for example, when a position is closed out. 

The resulting risks include the leaking of information from the test into the training set, likely leading to an artificially inflated performance that needs to be addressed by ensuring that all data is point-in-time—that is, truly available and known at the time it is used as the input for a model. Several methods have been proposed by Marcos Lopez de Prado in [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) to address these challenges of financial data for cross-validation:

- Purging: Eliminate training data points where the evaluation occurs after the prediction of a point-in-time data point in the validation set to avoid look-ahead bias.
- Embargoing: Further eliminate training samples that follow a test period.
