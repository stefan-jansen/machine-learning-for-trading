# Chapter 9: Bayesian Machine Learning

## How Bayesian Machine Learning Works

- [Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)
- [Andrew Gelman's Blog](https://andrewgelman.com/)
- [Thomas Wiecki's Blog](https://twiecki.github.io/)


### How to update assumptions from empirical evidence

- [Bayes' rule: Guide](https://arbital.com/p/bayes_rule/?l=1zq)
- [Bayesian Updating with Continuous Priors](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading13a.pdf), MIT Open Courseware, 18.05 Introduction to Probability and Statistics

#### Code Examples

The notebook `updating_conjugate_priors` contains the code for this section

### Exact Inference: Maximum a Posterior Estimation

### Approximate Inference: Stochastic vs Deterministic Approaches

#### Markov Chain Monte Carlo Methods

- [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf), Michael Betancourt, 2018
- [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](https://arxiv.org/abs/1111.4246), Matthew D. Hoffman, Andrew Gelman, 2011
- [ML, MAP, and Bayesian — The Holy Trinity of Parameter Estimation and Data Prediction](https://engineering.purdue.edu/kak/Trinity.pdf)
#### Variational Bayes

- [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf), David Blei et al, 2018

## Probabilistic Programming with PyMC3

- [Probabilistic Programming](http://www.probabilistic-programming.org/wiki/Home) Community Repository with links to papers and software
- [Stan](https://mc-stan.org/)
- [Edward](http://edwardlib.org/)
- [TensorFlow Probability](https://github.com/tensorflow/probability)
- [Pyro](http://pyro.ai/)

### Bayesian ML with Theano

### The PyMC3 Workflow

- [Documentation](https://docs.pymc.io/)
- [Probabilistic Programming in Python using PyMC](https://arxiv.org/abs/1507.08050), Salvatier et al 2015
- [Theano: A Python framework for fast computation of mathematical expressions](https://pdfs.semanticscholar.org/6b57/0069f14c7588e066f7138e1f21af59d62e61.pdf), Al-Rfou et al, 2016
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- [Bad Traces, or, Don't Use Metropolis](https://colindcarroll.com/2018/01/01/bad-traces-or-dont-use-metropolis/)

#### Code Examples

The notebook `bayesian_logistic_regression` contains the code used in this section.

### Practical Applications

- [Tackling the Poor Assumptions of Naive Bayes Text Classifiers](http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf), Rennie, et al, MIT SAIL, 2003
- [On Discriminative vs Generative Classifiers: A comparison of logistic regression and naive Bayes](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf), Jordan, Ng, 2002
- [Bayesian estimation supersedes the t test](http://www.indiana.edu/~kruschke/BEST/BEST.pdf), John K. Kruschke, Journal of Experimental Psychology, 2012
- [Automatic Differentiation Variational Inference](https://arxiv.org/pdf/1603.00788.pdf)

#### Code Examples

The notebooks
- `bayesian_sharpe_ratio`
- `bayesian_time_series`
- `linear_regression`, and
- `stochastic_volatility`

contain the code used in this section.
