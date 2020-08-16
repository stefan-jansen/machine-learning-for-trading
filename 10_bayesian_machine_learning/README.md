# Bayesian ML: From recession forecasts to dynamic pairs trading

In this chapter, we will introduce Bayesian approaches to machine learning (ML) and how their different perspective on uncertainty adds value when developing and evaluating trading strategies.

Bayesian statistics allows us to quantify uncertainty about future events and refine our estimates in a principled way as new information arrives. This dynamic approach adapts well to the evolving nature of financial markets. It is particularly useful when there are fewer relevant data and we require methods that systematically integrate prior knowledge or assumptions.

We will see that Bayesian approaches to machine learning allow for richer insights into the uncertainty around statistical metrics, parameter estimates, and predictions. The applications range from more granular risk management to dynamic updates of predictive models that incorporate changes in the market environment. The Black-Litterman approach to asset allocation (see [Chapter 5, Portfolio Optimization and Performance Evaluation](../05_strategy_evaluation) can be interpreted as a Bayesian model. It computes the expected return of an asset as an average of the market equilibrium and the investor’s views, weighted by each asset’s volatility, cross-asset correlations, and the confidence in each forecast.

## Content

1. [How Bayesian Machine Learning Works](#how-bayesian-machine-learning-works)
        - [References](#references)
    * [How to update assumptions from empirical evidence](#how-to-update-assumptions-from-empirical-evidence)
    * [Exact Inference: Maximum a Posterior Estimation](#exact-inference-maximum-a-posterior-estimation)
        - [How to keep inference simple: Conjugate Priors](#how-to-keep-inference-simple-conjugate-priors)
        - [Code example: How to dynamically estimate the probabilities of asset price moves](#code-example-how-to-dynamically-estimate-the-probabilities-of-asset-price-moves)
    * [Deterministic and stochastic approximate inference](#deterministic-and-stochastic-approximate-inference)
2. [Probabilistic Programming with PyMC3](#probabilistic-programming-with-pymc3)
    * [Bayesian ML with Theano](#bayesian-ml-with-theano)
    * [The PyMC3 workflow](#the-pymc3-workflow)
    * [Code example: Predicting a recession with PyMC3](#code-example-predicting-a-recession-with-pymc3)
    * [The Data: Leading Recession Indicators](#the-data-leading-recession-indicators)
        - [Model Definition: Bayesian Logistic Regression](#model-definition-bayesian-logistic-regression)
3. [Bayesian ML for Trading](#bayesian-ml-for-trading)
    * [Code Example: Bayesian Sharpe ratio for performance comparison](#code-example-bayesian-sharpe-ratio-for-performance-comparison)
    * [Code Example: Bayesian Rolling Regression for Pairs Trading](#code-example-bayesian-rolling-regression-for-pairs-trading)
    * [Code Example: Stochastic Volatility Models](#code-example-stochastic-volatility-models)
4. [Resources](#resources)
    * [PyMC3](#pymc3)
    * [Alternative probabilistic programming libraries](#alternative-probabilistic-programming-libraries)


## How Bayesian Machine Learning Works

Classical statistics is said to follow the frequentist approach because it interprets probability as the relative frequency of an event over the long run, i.e. after observing a large number of trials. In the context of probabilities, an event is a combination of one or more elementary outcomes of an experiment, such as any of six equal results in rolls of two dice or an asset price dropping by 10 percent or more on a given day). 

Bayesian statistics, in contrast, views probability as a measure of the confidence or belief in the occurrence of an event. The Bayesian perspective, thus, leaves more room for subjective views and differences in opinions than the frequentist interpretation. This difference is most striking for events that do not happen often enough to arrive at an objective measure of long-term frequency.

Put differently, frequentist statistics assumes that data is a random sample from a population and aims to identify the fixed parameters that generated the data. Bayesian statistics, in turn, takes the data as given and considers the parameters to be random variables with a distribution that can be inferred from data. As a result, frequentist approaches require at least as many data points as there are parameters to be estimated. Bayesian approaches, on the other hand, are compatible with smaller datasets, and well suited for online learning from one sample at a time.

The Bayesian view is very useful for many real-world events that are rare or unique, at least in important respects. Examples include the outcome of the next election or the question of whether the markets will crash within three months. In each case, there is both relevant historical data as well as unique circumstances that unfold as the event approaches.

We first introduce the Bayes theorem that crystallizes the concept of updating beliefs by combining prior assumptions with new empirical evidence and compare the resulting parameter estimates with their frequentist counterparts. We then demonstrate two approaches to Bayesian statistical inference, namely conjugate priors and approximate inference that produce insights into the posterior distribution of latent, i.e. unobserved parameters, such as the expected value:

1. Conjugate priors facilitate the updating process by providing a closed-form solution that allows us to precisely compute the solution. However, such exact, analytical methods are not always available.
2. Approximate inference simulates the distribution that results from combining assumptions and data and uses samples from this distribution to compute statistical insights.

#### References

- [Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)
- [Andrew Gelman's Blog](https://andrewgelman.com/)
- [Thomas Wiecki's Blog](https://twiecki.github.io/)

### How to update assumptions from empirical evidence

The theorem that Reverend Thomas Bayes came up with over 250 years ago uses fundamental probability theory to prescribe how probabilities or beliefs should change as relevant new information arrives as captured by John Maynard Keynes’ quote “When the facts change, I change my mind. What do you do, sir?”.

- [Bayes' rule: Guide](https://arbital.com/p/bayes_rule/?l=1zq)
- [Bayesian Updating with Continuous Priors](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading13a.pdf), MIT Open Courseware, 18.05 Introduction to Probability and Statistics

### Exact Inference: Maximum a Posterior Estimation

Practical applications of Bayes’ rule to exactly compute posterior probabilities are quite limited because the computation of the evidence term in the denominator is quite challenging.

#### How to keep inference simple: Conjugate Priors
A prior distribution is conjugate with respect to the likelihood when the resulting posterior is of the same type of distribution as the prior except for different parameters. The conjugacy of prior and likelihood implies a closed-form solution for the posterior that facilitates the update process and avoids the need to use numerical methods to approximate the posterior.

#### Code example: How to dynamically estimate the probabilities of asset price moves

The notebook [updating_conjugate_priors](01_updating_conjugate_priors.ipynb) demonstrates how to use a conjugate prior to update price movement estimates from S&P 500 samples.

### Deterministic and stochastic approximate inference

For most models of practical relevance, it will not be possible to derive the exact posterior distribution analytically and compute expected values for the latent parameters.

Although for some applications the posterior distribution over unobserved parameters
will be of interest, most often it is primarily required to evaluate expectations, e.g. to make predictions. In such situations, we can rely on approximate inference:
- **Stochastic techniques** based on Markov Chain Monte Carlo (MCMC) sampling have popularized the use of Bayesian methods across many domains. They generally have the property to converge to the exact result. In practice, sampling methods can be computationally demanding and are often limited to small-scale problems. 
    - [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf), Michael Betancourt, 2018
    - [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://arxiv.org/abs/1111.4246), Matthew D. Hoffman, Andrew Gelman, 2011
    - [ML, MAP, and Bayesian — The Holy Trinity of Parameter Estimation and Data Prediction](https://engineering.purdue.edu/kak/Trinity.pdf)

- **Deterministic methods** called variational inference or variational Bayes are based on analytical approximations to the posterior distribution and can scale well to large applications. They make simplifying assumptions, e.g., that the posterior factorizes in a particular way or it has a specific parametric form such as a Gaussian. Hence, they do not generate exact results and can be used as complements to sampling methods.
    - [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf), David Blei et al, 2018

## Probabilistic Programming with PyMC3

Probabilistic programming provides a language to describe and fit probability distributions so that we can design, encode and automatically estimate and evaluate complex models. It aims to abstract away some of the computational and analytical complexity to allow us to focus on the conceptually more straightforward and intuitive aspects of Bayesian reasoning and inference.
The field has become quite dynamic since new languages emerged since Uber open-sourced Pyro (based on PyTorch) and Google more recently added a probability module to TensorFlow. 

### Bayesian ML with Theano

- [PyMC3](https://docs.pymc.io/) was released in January 2017 to add Hamiltonian MC methods to the Metropolis-Hastings sampler used in PyMC2 (released 2012). PyMC3 uses [Theano](http://www.deeplearning.net/software/theano/) as its computational backend for dynamic C compilation and automatic differentiation. Theano is a matrix-focused and GPU-enabled optimization library developed at Yoshua Bengio’s Montreal Institute for Learning Algorithms (MILA) that inspired TensorFlow. MILA recently ceased to further develop Theano due to the success of newer deep learning libraries (see chapter 16 for details). 
- [PyMC4](https://github.com/pymc-devs/pymc4), planned for 2019, will use TensorFlow instead, with presumably limited impact on the API.

### The PyMC3 workflow

PyMC3 aims for intuitive and readable, yet powerful syntax that reflects how statisticians describe models. The modeling process generally follows these three steps:
1) Encode a probability model by defining: 
    1) The prior distributions that quantify knowledge and uncertainty about latent variables
    2) The likelihood function that conditions the parameters on observed data
2) Analyze the posterior using one of the options described in the previous section:
    1) Obtain a point estimate using MAP inference
    2) Sample from the posterior using MCMC methods
    3)Approximate the posterior using variational Bayes
3) Check your model using various diagnostic tools
4) Generate predictions

- [Documentation](https://docs.pymc.io/)
- [Probabilistic Programming in Python using PyMC](https://arxiv.org/abs/1507.08050), Salvatier et al 2015
- [Theano: A Python framework for fast computation of mathematical expressions](https://pdfs.semanticscholar.org/6b57/0069f14c7588e066f7138e1f21af59d62e61.pdf), Al-Rfou et al, 2016
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- [Bad Traces, or, Don't Use Metropolis](https://colindcarroll.com/2018/01/01/bad-traces-or-dont-use-metropolis/)
- PyMC 4 on [GitHub](https://github.com/pymc-devs/pymc4) with design guide and a usage examples.

### Code example: Predicting a recession with PyMC3

The notebook [pymc3_workflow](02_pymc3_workflow.ipynb) illustrates various aspects of the PyMC3 workflow using a simple logistic regression to model the prediction of a recession.

### The Data: Leading Recession Indicators

We will use a small and simple dataset so we can focus on the workflow. We use the Federal Reserve’s Economic Data (FRED) service (see Chapter 2) to download the US recession dates as defined by the National Bureau of Economic Research. We also source four variables that are commonly used to predict the onset of a recession (Kelley 2019) and available via FRED, namely:
- The long-term spread of the treasury yield curve, defined as the difference between the ten-year and the three-month Treasury yield.
- The University of Michigan’s consumer sentiment indicator
- The National Financial Conditions Index (NFCI), and
- The NFCI nonfinancial leverage subindex.

#### Model Definition: Bayesian Logistic Regression

As discussed in [Linear Models](../07_linear_models), logistic regression estimates a linear relationship between a set of features and a binary outcome, mediated by a sigmoid function to ensure the model produces probabilities. The frequentist approach resulted in point estimates for the parameters that measure the influence of each feature on the probability that a data point belongs to the positive class, with confidence intervals based on assumptions about the parameter distribution. 

Bayesian logistic regression, in contrast, estimates the posterior distribution over the parameters itself. The posterior allows for more robust estimates of what is called a Bayesian credible interval for each parameter with the benefit of more transparency about the model’s uncertainty.

The notebook [pymc3_workflow](02_pymc3_workflow.ipynb) demonstrates the PYMC3 workflow, including:
- MAP Inference
- Markov Chain Monte Carlo Estimate 
    - Metropolis-Hastings
    - NUTS Sampler
- Variational Inference
- Model Diagnostics
    - Energy and Forest Plots
    - Posterior Predictive Checks (PPD), and 
    - Credible Intervals (CI)
- Prediction
- MCMC Sampler Animation
    
## Bayesian ML for Trading

Now that we are familiar with the Bayesian approach to ML and probabilistic programming with PyMC3, let’s explore a few relevant trading-related applications, namely 
- modeling the Sharpe ratio as a probabilistic model for more insightful performance comparison
- computing pairs trading hedge ratios using Bayesian linear regression
- analyzing linear time series models from a Bayesian perspective

### Code Example: Bayesian Sharpe ratio for performance comparison

The notebook [bayesian_sharpe_ratio](03_bayesian_sharpe_ratio.ipynb) illustrates how to define the Sharpe ratio (SR) as a probabilistic model using PyMC3, and how to compare its posterior distributions for different return series. 

The Bayesian estimation for two series offers very rich insights because it provides the complete distributions of the credible values for the effect size, the group SR means and their difference, as well as standard deviations and their difference. The Python implementation is due to Thomas Wiecki and was inspired by the R package BEST (Meredith and Kruschke 2018), see 'Resources' below.

Relevant use cases of a Bayesian SR include the analysis of differences between alternative strategies, or between a strategy’s in-sample return relative to its out-of-sample return (see the notebook bayesian_sharpe_ratio for details). The Bayesian Sharpe ratio is also part of pyfolio’s Bayesian tearsheet.

### Code Example: Bayesian Rolling Regression for Pairs Trading

The last [chapter](../09_time_series_models) introduced pairs trading as a popular trading strategy that relies on the **cointegration** of two or more assets. Given such assets, we need to estimate the hedging ratio to decide on the relative magnitude of long and short positions. A basic approach uses linear regression. 

The notebook [rolling_regression](04_rolling_regression.ipynb) llustrates how Bayesian linear regression tracks changes in the relationship between two assets over time. It follows Thomas Wiecki’s example (see 'Resources' below).

### Code Example: Stochastic Volatility Models

As discussed in the chapter [Time Series Models](../09_time_series_models), asset prices have time-varying volatility. In some periods, returns are highly variable, while in others very stable. 

Stochastic volatility models model this with a latent volatility variable, modeled as a stochastic process. The  No-U-Turn Sampler was introduced using such a model, and the notebook [stochastic_volatility](05_stochastic_volatility.ipynb) illustrates this use case.

## Resources

### PyMC3

Thomas Wiecki, one of the main PyMC3 authors who also leads Data Science at Quantopian has created several examples that the following sections follow and build on. The PyMC3 documentation has many additional tutorials.

- PyMC3 [Tutorials](https://docs.pymc.io/nb_tutorials/index.html)
- [Tackling the Poor Assumptions of Naive Bayes Text Classifiers](http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf), Rennie, et al, MIT SAIL, 2003
- [On Discriminative vs Generative Classifiers: A comparison of logistic regression and naive Bayes](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf), Jordan, Ng, 2002
- [Bayesian estimation supersedes the t test](http://www.indiana.edu/~kruschke/BEST/BEST.pdf), John K. Kruschke, Journal of Experimental Psychology, 2012
- [Automatic Differentiation Variational Inference](https://arxiv.org/pdf/1603.00788.pdf)

### Alternative probabilistic programming libraries
- [Probabilistic Programming](http://www.probabilistic-programming.org/wiki/Home) Community Repository with links to papers and software
- [Stan](https://mc-stan.org/)
- [Edward](http://edwardlib.org/)
- [TensorFlow Probability](https://github.com/tensorflow/probability)
- [Pyro](http://pyro.ai/)

