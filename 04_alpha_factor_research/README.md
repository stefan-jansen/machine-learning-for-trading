# Financial Feature Engineering: How to research Alpha Factors

Algorithmic trading strategies are driven by signals that indicate when to buy or sell assets to generate superior returns relative to a benchmark such as an index. The portion of an asset's return that is not explained by exposure to this benchmark is called alpha, and hence the signals that aim to produce such uncorrelated returns are also called alpha factors.

If you are already familiar with ML, you may know that feature engineering is a key ingredient for successful predictions. This is no different in trading. Investment, however, is particularly rich in decades of research into how markets work and which features may work better than others to explain or predict price movements as a result. This chapter provides an overview as a starting point for your own search for alpha factors.

This chapter also presents key tools that facilitate the computing and testing alpha factors. We will highlight how the NumPy, pandas and TA-Lib libraries facilitate the manipulation of data and present popular smoothing techniques like the wavelets and the Kalman filter that help reduce noise in data.

We also preview how you can use the trading simulator Zipline to evaluate the predictive performance of (traditional) alpha factors. We discuss key alpha factor metrics like the information coefficient and factor turnover. An in-depth introduction to backtesting trading strategies that use machine learning follows in [Chapter 6](../08_ml4t_workflow), which covers the **ML4T workflow** that we will use throughout the book to evaluate trading strategies. 

Please see the [Appendix - Alpha Factor Library](../24_alpha_factor_library) for additional material on this topic, including numerous code examples that compute a broad range of alpha factors.

## Content

1. [Alpha Factors in practice: from data to signals](#alpha-factors-in-practice-from-data-to-signals)
2. [Building on Decades of Factor Research](#building-on-decades-of-factor-research)
    * [References](#references)
3. [Engineering alpha factors that predict returns](#engineering-alpha-factors-that-predict-returns)
    * [Code Example: How to engineer factors using pandas and NumPy](#code-example-how-to-engineer-factors-using-pandas-and-numpy)
    * [Code Example: How to use TA-Lib to create technical alpha factors](#code-example-how-to-use-ta-lib-to-create-technical-alpha-factors)
    * [Code Example: How to denoise your Alpha Factors with the Kalman Filter](#code-example-how-to-denoise-your-alpha-factors-with-the-kalman-filter)
    * [Code Example: How to preprocess your noisy signals using Wavelets](#code-example-how-to-preprocess-your-noisy-signals-using-wavelets)
    * [Resources](#resources)
4. [From signals to trades: backtesting with `Zipline`](#from-signals-to-trades-backtesting-with-zipline)
    * [Code Example: How to use Zipline to backtest a single-factor strategy](#code-example-how-to-use-zipline-to-backtest-a-single-factor-strategy)
    * [Code Example: Combining factors from diverse data sources on the Quantopian platform](#code-example-combining-factors-from-diverse-data-sources-on-the-quantopian-platform)
    * [Code Example: Separating signal and noise – how to use alphalens](#code-example-separating-signal-and-noise--how-to-use-alphalens)
5. [Alternative Algorithmic Trading Libraries and Platforms](#alternative-algorithmic-trading-libraries-and-platforms)

## Alpha Factors in practice: from data to signals

Alpha factors are transformations of market, fundamental, and alternative data that contain predictive signals. They are designed to capture risks that drive asset returns. One set of factors describes fundamental, economy-wide variables such as growth, inflation, volatility, productivity, and demographic risk. Another set consists of tradeable investment styles such as the market portfolio, value-growth investing, and momentum investing.

There are also factors that explain price movements based on the economics or institutional setting of financial markets, or investor behavior, including known biases of this behavior. The economic theory behind factors can be rational, where the factors have high returns over the long run to compensate for their low returns during bad times, or behavioral, where factor risk premiums result from the possibly biased, or not entirely rational behavior of agents that is not arbitraged away.

## Building on Decades of Factor Research

In an idealized world, categories of risk factors should be independent of each other (orthogonal), yield positive risk premia, and form a complete set that spans all dimensions of risk and explains the systematic risks for assets in a given class. In practice, these requirements will hold only approximately.

### References

- [Dissecting Anomalies](http://schwert.ssb.rochester.edu/f532/ff_JF08.pdf) by Eugene Fama and Ken French (2008)
- [Explaining Stock Returns: A Literature Review](https://www.ifa.com/pdfs/explainingstockreturns.pdf) by James L. Davis (2001)
- [Market Efficiency, Long-Term Returns, and Behavioral Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=15108) by Eugene Fama (1997)
- [The Efficient Market Hypothesis and It's Critics](https://pubs.aeaweb.org/doi/pdf/10.1257/089533003321164958) by Burton Malkiel (2003)
- [The New Palgrave Dictionary of Economics](https://www.palgrave.com/us/book/9780333786765) (2008) by Steven Durlauf and Lawrence Blume, 2nd ed.
- [Anomalies and Market Efficiency](https://www.nber.org/papers/w9277.pdf) by G. William Schwert25 (Ch. 15 in Handbook of the- "Economics of Finance", by Constantinides, Harris, and Stulz, 2003)
- [Investor Psychology and Asset Pricing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=265132), by David Hirshleifer (2001)

## Engineering alpha factors that predict returns

Based on a conceptual understanding of key factor categories, their rationale and popular metrics, a key task is to identify new factors that may better capture the risks embodied by the return drivers laid out previously, or to find new ones. In either case, it will be important to compare the performance of innovative factors to that of known factors to identify incremental signal gains.

### Code Example: How to engineer factors using pandas and NumPy

The notebook [feature_engineering.ipynb](00_data/feature_engineering.ipynb) in the [data](00_data) directory illustrates how to engineer basic factors.

### Code Example: How to use TA-Lib to create technical alpha factors

The notebook [how_to_use_talib](02_how_to_use_talib.ipynb) illustrates the usage of TA-Lib, which includes a broad range of common technical indicators. These indicators have in common that they only use market data, i.e., price and volume information.

The notebook [common_alpha_factors](../24_alpha_factor_library/02_common_alpha_factors.ipynb) in th **appendix** contains dozens of additional examples.  

### Code Example: How to denoise your Alpha Factors with the Kalman Filter

The notebook [kalman_filter_and_wavelets](03_kalman_filter_and_wavelets.ipynb) demonstrates the use of the Kalman filter using the `PyKalman` package for smoothing; we will also use it in [Chapter 9](../09_time_series_models) when we develop a pairs trading strategy.

### Code Example: How to preprocess your noisy signals using Wavelets

The notebook [kalman_filter_and_wavelets](03_kalman_filter_and_wavelets.ipynb) also demonstrates how to work with wavelets using the `PyWavelets` package.

### Resources

- [Fama French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) Data Library
- [numpy](https://numpy.org/) website
    - [Quickstart Tutorial](https://numpy.org/devdocs/user/quickstart.html)
- [pandas](https://pandas.pydata.org/) website
    - [User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
    - [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
    - [Python Pandas Tutorial: A Complete Introduction for Beginners](https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/)
- [alphatools](https://github.com/marketneutral/alphatools) - Quantitative finance research tools in Python
- [mlfinlab](https://github.com/hudson-and-thames/mlfinlab) - Package based on the work of Dr Marcos Lopez de Prado regarding his research with respect to Advances in Financial Machine Learning
- [PyKalman](https://pykalman.github.io/) documentation
- [Tutorial: The Kalman Filter](http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf)
- [Understanding and Applying Kalman Filtering](http://biorobotics.ri.cmu.edu/papers/sbp_papers/integrated3/kleeman_kalman_basics.pdf)
- [How a Kalman filter works, in pictures](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) - Wavelet Transforms in Python
- [An Introduction to Wavelets](https://www.eecis.udel.edu/~amer/CISC651/IEEEwavelet.pdf) 
- [The Wavelet Tutorial](http://web.iitd.ac.in/~sumeet/WaveletTutorial.pdf)
- [Wavelets for Kids](http://www.gtwavelet.bme.gatech.edu/wp/kidsA.pdf)
- [The Barra Equity Risk Model Handbook](https://www.alacra.com/alacra/help/barra_handbook_GEM.pdf)
- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [Modern Investment Management: An Equilibrium Approach](https://www.amazon.com/Modern-Investment-Management-Equilibrium-Approach/dp/0471124109) by Bob Litterman, 2003
- [Quantitative Equity Portfolio Management: Modern Techniques and Applications](https://www.crcpress.com/Quantitative-Equity-Portfolio-Management-Modern-Techniques-and-Applications/Qian-Hua-Sorensen/p/book/9781584885580) by Edward Qian, Ronald Hua, and Eric Sorensen
- [Spearman Rank Correlation](https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php)

## From signals to trades: backtesting with `Zipline`

The open source [zipline](http://www.zipline.io/index.html) library is an event-driven backtesting system maintained and used in production by the crowd-sourced quantitative investment fund [Quantopian](https://www.quantopian.com/) to facilitate algorithm-development and live-trading. It automates the algorithm's reaction to trade events and provides it with current and historical point-in-time data that avoids look-ahead bias.

- [Chapter 8](../08_ml4t_workflow) contains a more comprehensive introduction to Zipline.
- Please follow the [instructions](../installation) in the `installation` folder.

### Code Example: How to use Zipline to backtest a single-factor strategy

The notebook [single_factor_zipline](04_single_factor_zipline.ipynb) develops and test a simple mean-reversion factor that measures how much recent performance has deviated from the historical average. Short-term reversal is a common strategy that takes advantage of the weakly predictive pattern that stock price increases are likely to mean-revert back down over horizons from less than a minute to one month.

### Code Example: Combining factors from diverse data sources on the Quantopian platform

The Quantopian research environment is tailored to the rapid testing of predictive alpha factors. The process is very similar because it builds on `zipline`, but offers much richer access to data sources. 

The notebook [multiple_factors_quantopian_research](05_multiple_factors_quantopian_research.ipynb) illustrates how to compute alpha factors not only from market data as previously but also from fundamental and alternative data.
    
### Code Example: Separating signal and noise – how to use alphalens

The notebook [performance_eval_alphalens](06_performance_eval_alphalens.ipynb) introduces the [alphalens](http://quantopian.github.io/alphalens/) library for the performance analysis of predictive (alpha) factors, open-sourced by Quantopian. It demonstrates how it integrates with the backtesting library `zipline` and the portfolio performance and risk analysis library `pyfolio` that we will explore in the next chapter.

`alphalens` facilitates the analysis of the predictive power of alpha factors concerning the:
- Correlation of the signals with subsequent returns
- Profitability of an equal or factor-weighted portfolio based on a (subset of) the signals
- Turnover of factors to indicate the potential trading costs
- Factor-performance during specific events
- Breakdowns of the preceding by sector

The analysis can be conducted using `tearsheets` or individual computations and plots. The tearsheets are illustrated in the online repo to save some space.

- See [here](https://github.com/quantopian/alphalens/blob/master/alphalens/examples/alphalens_tutorial_on_quantopian.ipynb) for a detailed `alphalens` tutorial by Quantopian

## Alternative Algorithmic Trading Libraries and Platforms

- [QuantConnect](https://www.quantconnect.com/)
- [Alpha Trading Labs](https://www.alphalabshft.com/)
    - Alpha Trading Labs is no longer active
- [WorldQuant](https://www.worldquantvrc.com/en/cms/wqc/home/)
- Python Algorithmic Trading Library [PyAlgoTrade](http://gbeced.github.io/pyalgotrade/)
- [pybacktest](https://github.com/ematvey/pybacktest)
- [Trading with Python](http://www.tradingwithpython.com/)
- [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=5041)
