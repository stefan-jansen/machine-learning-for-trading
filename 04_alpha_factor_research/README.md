# Chapter 04: Alpha Factor Research & Evaluation

## Engineering Alpha Factor

### Important Factor Categories

- [Dissecting Anomalies](http://schwert.ssb.rochester.edu/f532/ff_JF08.pdf) by Eugene Fama and Ken French (2008)
- [Explaining Stock Returns: A Literature Review](https://www.ifa.com/pdfs/explainingstockreturns.pdf) by James L. Davis (2001)
- [Market Efficiency, Long-Term Returns, and Behavioral Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=15108) by Eugene Fama (1997)
- [The Efficient Market Hypothesis and It's Critics](https://pubs.aeaweb.org/doi/pdf/10.1257/089533003321164958) by Burton Malkiel (2003)
- [The New Palgrave Dictionary of Economics](https://www.palgrave.com/us/book/9780333786765) (2008) by Steven Durlauf and Lawrence Blume, 2nd ed.
- [Anomalies and Market Efficiency](https://www.nber.org/papers/w9277.pdf) by G. William Schwert25 (Ch. 15 in Handbook of the- "Economics of Finance", by Constantinides, Harris, and Stulz, 2003)
- [Investor Psychology and Asset Pricing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=265132), by David Hirshleifer (2001)

### How to transform Data into Factors

- The notebook `feature_engineering.ipynb` in the data directory illustrates how to engineer basic factors.


#### References

- [The Barra Equity Risk Model Handbook](https://www.alacra.com/alacra/help/barra_handbook_GEM.pdf)
- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [Modern Investment Management: An Equilibrium Approach](https://www.amazon.com/Modern-Investment-Management-Equilibrium-Approach/dp/0471124109) by Bob Litterman, 2003
- [Quantitative Equity Portfolio Management: Modern Techniques and Applications](https://www.crcpress.com/Quantitative-Equity-Portfolio-Management-Modern-Techniques-and-Applications/Qian-Hua-Sorensen/p/book/9781584885580) by Edward Qian, Ronald Hua, and Eric Sorensen
- [Spearman Rank Correlation](https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php)

#### Code Examples

## Seeking Signals - How to use `zipline`

This section introduces the algorithmic trading simulator [`zipline`](http://www.zipline.io/index.html) and the [`alphalens`](http://quantopian.github.io/alphalens/) library for the performance analysis of predictive (alpha) factors.

### Code Examples
- `zipline` installation: see [docs](http://www.zipline.io/index.html) for more detail.

## Separating signal and noise – how to use alphalens

### The Information Coefficient

### Code Examples

- `alphalens` installation see [docs](http://quantopian.github.io/alphalens/) for detail

Alphalens depends on:

-  [`matplotlib`]( <https://github.com/matplotlib/matplotlib)
-  [`numpy`](https://github.com/numpy/numpy)
-  [`pandas`](https://github.com/pydata/pandas)
-  [`scipy`](https://github.com/scipy/scipy)
-  [`seaborn`](https://github.com/mwaskom/seaborn)
-  [`statsmodels`](https://github.com/statsmodels/statsmodels)


## Alternative Algorithmic Trading Libraries

- [QuantConnect](https://www.quantconnect.com/)
- [Alpha Trading Labs](https://www.alphalabshft.com/)
- [WorldQuant](https://www.worldquantvrc.com/en/cms/wqc/home/)
- Python Algorithmic Trading Library [PyAlgoTrade](http://gbeced.github.io/pyalgotrade/)
- [pybacktest](https://github.com/ematvey/pybacktest)
- [Trading with Python](http://www.tradingwithpython.com/)
- [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=5041)
