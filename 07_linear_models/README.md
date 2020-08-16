# Linear Models: From Risk Factors to Asset Return Forecasts

The family of linear models represents one of the most useful hypothesis classes. Many learning algorithms that are widely applied in algorithmic trading rely on linear predictors because they can be efficiently trained, are relatively robust to noisy financial data and have strong links to the theory of finance. Linear predictors are also intuitive, easy to interpret, and often fit the data reasonably well or at least provide a good baseline.

Linear regression has been known for over 200 years since Legendre and Gauss applied it to astronomy and began to analyze its statistical properties. Numerous extensions have since adapted the linear regression model and the baseline ordinary least squares (OLS) method to learn its parameters:

- **Generalized linear models** (GLM) expand the scope of applications by allowing for response variables that imply an error distribution other than the normal distribution. GLMs include the probit or logistic models for categorical response variables that appear in classification problems.
- More **robust estimation methods** enable statistical inference where the data violates baseline assumptions due to, for example, correlation over time or across observations. This is often the case with panel data that contains repeated observations on the same units such as historical returns on a universe of assets.
- **Shrinkage methods** aim to improve the predictive performance of linear models. They use a complexity penalty that biases the coefficients learned by the model with the goal of reducing the model's variance and improving out-of-sample predictive performance.

In practice, linear models are applied to regression and classification problems with the goals of inference and prediction. Numerous asset pricing models have been developed by academic and industry researchers that leverage linear regression. Applications include the identification of significant factors that drive asset returns for better risk and performance management, as well as the prediction of returns over various time horizons. Classification problems, on the other hand, include directional price forecasts. In this chapter, we will cover the following topics:

## Content

1. [Linear regression: From inference to prediction](#linear-regression-from-inference-to-prediction)
2. [The baseline model: Multiple linear regression](#the-baseline-model-multiple-linear-regression)
    * [Code Example: Simple and multiple linear regression with `statsmodels` and `scikit-learn`](#code-example-simple-and-multiple-linear-regression-with-statsmodels-and-scikit-learn)
3. [How to build a linear factor model](#how-to-build-a-linear-factor-model)
    * [From the CAPM to the Fama—French five-factor model](#from-the-capm-to-the-famafrench-five-factor-model)
    * [Obtaining the risk factors](#obtaining-the-risk-factors)
    * [Code Example: Fama—Macbeth regression](#code-example-famamacbeth-regression)
4. [Shrinkage methods: Regularization for linear regression](#shrinkage-methods-regularization-for-linear-regression)
    * [Hedging against overfitting – regularization in linear models](#hedging-against-overfitting--regularization-in-linear-models)
    * [Ridge regression](#ridge-regression)
    * [Lasso regression](#lasso-regression)
5. [How to predict stock returns with linear regression](#how-to-predict-stock-returns-with-linear-regression)
    * [Code Examples: inference and prediction for stock returns ](#code-examples-inference-and-prediction-for-stock-returns-)
6. [Linear classification](#linear-classification)
    * [The logistic regression model](#the-logistic-regression-model)
    * [Code Example: how to conduct inference with statsmodels](#code-example-how-to-conduct-inference-with-statsmodels)
    * [Code examples: how to use logistic regression for prediction](#code-examples-how-to-use-logistic-regression-for-prediction)
7. [References](#references)


## Linear regression: From inference to prediction

This section introduces the baseline cross-section and panel techniques for linear models and important enhancements that produce accurate estimates when key assumptions are violated. It continues to illustrate these methods by estimating factor models that are ubiquitous in the development of algorithmic trading strategies. Lastly, it focuses on regularization methods.

- [Introductory Econometrics](http://economics.ut.ac.ir/documents/3030266/14100645/Jeffrey_M._Wooldridge_Introductory_Econometrics_A_Modern_Approach__2012.pdf), Wooldridge, 2012

## The baseline model: Multiple linear regression

This section introduces the model's specification and objective function, methods to learn its parameters, statistical assumptions that allow for inference and diagnostics of these assumptions, as well as extensions to adapt the model to situations where these assumptions fail. Content includes:

- How to formulate and train the model
- The Gauss-Markov Theorem
- How to conduct statistical inference
- How to diagnose and remedy problems
- How to run linear regression in practice

### Code Example: Simple and multiple linear regression with `statsmodels` and `scikit-learn`

The notebook [linear_regression_intro](01_linear_regression_intro.ipynb) demonstrates the simple and multiple linear regression model, the latter using both OLS and gradient descent based on `statsmodels` and `scikit-learn`. 

## How to build a linear factor model

Algorithmic trading strategies use linear factor models to quantify the relationship between the return of an asset and the sources of risk that represent the main drivers of these returns. Each factor risk carries a premium, and the total asset return can be expected to correspond to a weighted average of these risk premia.

### From the CAPM to the Fama—French five-factor model

Risk factors have been a key ingredient to quantitative models since the Capital Asset Pricing Model (CAPM) explained the expected returns of all assets using their respective exposure to a single factor, the expected excess return of the overall market over the risk-free rate.

This differs from classic fundamental analysis a la Dodd and Graham where returns depend on firm characteristics. The rationale is that, in the aggregate, investors cannot eliminate this so-called systematic risk through diversification. Hence, in equilibrium, they require compensation for holding an asset commensurate with its systematic risk. The model implies that, given efficient markets where prices immediately reflect all public information, there should be no superior risk-adjusted returns.

### Obtaining the risk factors

The [Fama—French risk factors](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) are computed as the return difference on diversified portfolios with high or low values according to metrics that reflect a given risk factor. These returns are obtained by sorting stocks according to these metrics and then going long stocks above a certain percentile while shorting stocks below a certain percentile. The metrics associated with the risk factors are defined as follows:

- Size: Market Equity (ME) 
- Value: Book Value of Equity (BE) divided by ME
- Operating Profitability (OP): Revenue minus cost of goods sold/assets
- Investment: Investment/assets

Fama and French make updated risk factor and research portfolio data available through their [website]((http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)), and you can use the [pandas_datareader](https://pandas-datareader.readthedocs.io/en/latest/) library to obtain the data. 

### Code Example: Fama—Macbeth regression

To address the inference problem caused by the correlation of the residuals, Fama and MacBeth proposed a two-step methodology for a cross-sectional regression of returns on factors. The two-stage Fama—Macbeth regression is designed to estimate the premium rewarded for the exposure to a particular risk factor by the market. The two stages consist of:
- **First stage**: N time-series regression, one for each asset or portfolio, of its excess returns on the factors to estimate the factor loadings.
- **Second stage**: T cross-sectional regression, one for each time period, to estimate the risk premium.

The notebook [fama_macbeth](02_fama_macbeth.ipynb) illustrates how to run a Fama-Macbeth regression, including using the [LinearModels](https://bashtage.github.io/linearmodels/doc/) library.

## Shrinkage methods: Regularization for linear regression

When a linear regression model contains many correlated variables, their coefficients will be poorly determined because the effect of a large positive coefficient on the RSS can be canceled by a similarly large negative coefficient on a correlated variable. Hence, the model will have a tendency for high variance due to this wiggle room of the coefficients that increases the risk that the model overfits to the sample.

### Hedging against overfitting – regularization in linear models

One popular technique to control overfitting is that of regularization, which involves the addition of a penalty term to the error function to discourage the coefficients from reaching large values. In other words, size constraints on the coefficients can alleviate the resultant potentially negative impact on out-of-sample predictions. We will encounter regularization methods for all models since overfitting is such a pervasive problem.

In this section, we will introduce shrinkage methods that address two motivations to improve on the approaches to linear models discussed so far:
- Prediction accuracy: The low bias but high variance of least squares estimates suggests that the generalization error could be reduced by shrinking or setting some coefficients to zero, thereby trading off a slightly higher bias for a reduction in the variance of the model.
- Interpretation: A large number of predictors may complicate the interpretation or communication of the big picture of the results. It may be preferable to sacrifice some detail to limit the model to a smaller subset of parameters with the strongest effects.

### Ridge regression

The ridge regression shrinks the regression coefficients by adding a penalty to the objective function that equals the sum of the squared coefficients, which in turn corresponds to the L2 norm of the coefficient vector.

### Lasso regression

The lasso, known as basis pursuit in signal processing, also shrinks the coefficients by adding a penalty to the sum of squares of the residuals, but the lasso penalty has a slightly different effect. The lasso penalty is the sum of the absolute values of the coefficient vector, which corresponds to its L1 norm.

## How to predict stock returns with linear regression

In this section, we will use linear regression with and without shrinkage to predict returns and generate trading signals. To this end, we first create a dataset and then apply the linear regression models discussed in the previous section to illustrate their usage with statsmodels and sklearn.

### Code Examples: inference and prediction for stock returns 

- The notebook [preparing_the_model_data](03_preparing_the_model_data.ipynb) selects a universe of US equities and creates several features to predict daily returns.
- The notebook [statistical_inference_of_stock_returns_with_statsmodels](04_statistical_inference_of_stock_returns_with_statsmodels.ipynb) estimates several linear regression models using OLS and the `statsmodels` library.
- The notebook [predicting_stock_returns_with_linear_regression](05_predicting_stock_returns_with_linear_regression.ipynb) shows how to predict daily stock return using linear regression, as well as ridge and lasso models with  `scikit-klearn`.
- The notebook [evaluating_signals_using_alphalens](06_evaluating_signals_using_alphalens.ipynb) evaluates the model predictions using `alphalens`.

## Linear classification

There are many different classification techniques to predict a qualitative response. In this section, we will introduce the widely used logistic regression which is closely related to linear regression. We will address more complex methods in the following chapters, on generalized additive models that include decision trees and random forests, as well as gradient boosting machines and neural networks.

### The logistic regression model

The logistic regression model arises from the desire to model the probabilities of the output classes given a function that is linear in x, just like the linear regression model, while at the same time ensuring that they sum to one and remain in the [0, 1] as we would expect from probabilities.

In this section, we introduce the objective and functional form of the logistic regression model and describe the training method. We then illustrate how to use logistic regression for statistical inference with macro data using statsmodels, and how to predict price movements using the regularized logistic regression implemented by sklearn.

### Code Example: how to conduct inference with statsmodels

The notebook [logistic_regression_macro_data](07_logistic_regression_macro_data.ipynb)` illustrates how to run a logistic regression on macro data and conduct statistical inference using [statsmodels](https://www.statsmodels.org/stable/index.html).

### Code examples: how to use logistic regression for prediction

The lasso L1 penalty and the ridge L2 penalty can both be used with logistic regression. They have the same shrinkage effect as we have just discussed, and the lasso can again be used for variable selection with any linear regression model.

Just as with linear regression, it is important to standardize the input variables as the regularized models are scale sensitive. The regularization hyperparameter also requires tuning using cross-validation as in the linear regression case.

The notebook [predicting_price_movements_with_logistic_regression](08_predicting_price_movements_with_logistic_regression.ipynb) demonstrates how to use Logistic Regression for stock price movement prediction. 

## References

- [Risk, Return, and Equilibrium: Empirical Tests](https://www.jstor.org/stable/1831028), Eugene F. Fama and James D. MacBeth, Journal of Political Economy, 81 (1973), pp. 607–636
- [Asset Pricing](http://faculty.chicagobooth.edu/john.cochrane/teaching/asset_pricing.htm), John Cochrane, 2001
