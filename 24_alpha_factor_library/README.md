# Appendix - Alpha Factor Library

Throughout this book, we emphasized how the smart design of features, including appropriate preprocessing and denoising, typically leads to an effective strategy. 
This appendix synthesizes some of the lessons learned on feature engineering and provides additional information on this vital topic.

Chapter 4 categorized factors by the underlying risk they represent and for which an investor would earn a reward above and beyond the market return. 
These categories include value vs growth, quality, and sentiment, as well as volatility, momentum, and liquidity. 
Throughout the book, we used numerous metrics to capture these risk factors. 
This appendix expands on those examples and collects popular indicators so you can use it as a reference or inspiration for your own strategy development. 
It also shows you how to compute them and includes some steps to evaluate these indicators. 

To this end, we focus on the broad range of indicators implemented by TA-Lib (see [Chapter 4](04_alpha_factor_research)) and WorldQuant's [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf) paper (Kakushadze 2016), which presents real-life quantitative trading factors used in production with an average holding period of 0.6-6.4 days.

This chapter covers: 
- How to compute several dozen technical indicators using TA-Lib and NumPy/pandas,
- Creating the formulaic alphas describe in the above paper, and
- Evaluating the predictive quality of the results using various metrics from rank correlation and mutual information to feature importance, SHAP values and Alphalens. 

## Content

1. [The Indicator Zoo](#the-indicator-zoo)
2. [Code example: common alpha factors implemented in TA-Lib](#code-example-common-alpha-factors-implemented-in-ta-lib)
3. [Code example: WorldQuant’s quest for formulaic alphas](#code-example-worldquants-quest-for-formulaic-alphas)
4. [Code example: Bivariate and multivariate factor evaluation](#code-example-bivariate-and-multivariate-factor-evaluation)

## The Indicator Zoo

Chapter 4, [Financial Feature Engineering: How to Research Alpha Factors](../04_alpha_factor_research), summarized the long-standing efforts of academics and practitioners to identify information or variables that helps reliably predict asset returns. 
This research led from the single-factor capital asset pricing model to a “[zoo of new factors](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.407.3913&rep=rep1&type=pdf)" (Cochrane 2011). 

This factor zoo contains hundreds of firm characteristics and security price metrics presented as statistically significant predictors of equity returns in the anomalies literature since 1970 (see a summary in [Green, Hand, and Zhang](https://academic.oup.com/rfs/article-abstract/30/12/4389/3091648), 2017). 
- The notebook [indicator_zoo](00_indicator_zoo.ipynb) lists numerous examples.

## Code example: common alpha factors implemented in TA-Lib

The TA-Lib library is widely used to perform technical analysis of financial market data by trading software developers. It includes over 150 popular indicators from multiple categories that range from Overlap Studies, including moving averages and Bollinger Bands, to Statistic Functions such as linear regression. 

**Function Group**|**# Indicators**
:-----:|:-----:
Overlap Studies|17
Momentum Indicators|30
Volume Indicators|3
Volatility Indicators|3
Price Transform|4
Cycle Indicators|5
Math Operators|11
Math Transform|15
Statistic Functions|9

The notebook [common_alpha_factors](02_common_alpha_factors.ipynb) contains the relevant code samples.

## Code example: WorldQuant’s quest for formulaic alphas

We introduced [WorldQuant](https://www.worldquant.com/home/) in Chapter 1, [Machine Learning for Trading: From Idea to Execution](../01_machine_learning_for_trading), as part of a trend towards crowd-sourcing investment strategies. 
WorldQuant maintains a virtual research center where quants worldwide compete to identify alphas. 
These alphas are trading signals in the form of computational expressions that help predict price movements just like the common factors described in the previous section.
   
These formulaic alphas translate the mechanism to extract the signal from data into code and can be developed and tested individually with the goal to integrate their information into a broader automated strategy ([Tulchinsky 2019](https://onlinelibrary.wiley.com/doi/abs/10.1002/9781119571278.ch1). 
As stated repeatedly throughout the book, mining for signals in large datasets is prone to multiple testing bias and false discoveries. 
Regardless of these important caveats, this approach represents a modern alternative to the more conventional features presented in the previous section.

[Kakushadze (2016) presents [101 examples](https://arxiv.org/pdf/1601.00991.pdf) of such alphas, 80 percent of which were used in a real-world trading system at the time. It defines a range of functions that operate on cross-sectional or time-series data and can be combined, e.g. in nested form.

The notebook [101_formulaic_alphas](03_101_formulaic_alphas.ipynb) contains the relevant code.

## Code example: Bivariate and multivariate factor evaluation

To evaluate the numerous factors, we rely on the various performance measures introduced in this book, including the following:
- Bivariate measures of the signal content of a factor with respect to the one-day forward returns
- Multivariate measures of feature importance for a gradient boosting model trained to predict the one-day forward returns using all factors
- Financial performance of portfolios invested according to factor quantiles using Alphalens

The notebooks [factor_evaluation](04_factor_evaluation.ipynb) and [alphalens_analysis](05_alphalens_analysis.ipynb) contain the relevant code examples.



