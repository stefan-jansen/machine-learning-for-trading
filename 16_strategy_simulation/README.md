# Chapter 16: Strategy Simulation

The chapter gives the chapter its intellectual spine: a backtest is not proof that a strategy works, but a structured attempt to show that it fails under realistic assumptions. That framing matters because it shifts the reader from performance worship to disciplined skepticism, emphasizing leakage checks, execution realism, cost sensitivity, and regime robustness as the real standards of evidence.
## Learning Objectives
- Formalize a backtest as an explicit trading protocol covering signal timing, execution, rebalancing, sizing, costs, constraints, data availability, and benchmark choice
- Distinguish vectorized and event-driven backtesting in terms of protocol semantics, state dependence, and appropriate use cases rather than treating one style as universally superior
- Build and interpret a transparent non-ML baseline strategy that provides a stable reference point for later model comparisons
- Evaluate a strategy using a core reporting stack that includes gross and net performance, drawdowns, turnover, baseline comparison, cost sensitivity, and regime-sliced diagnostics
- Assess whether a reported Sharpe ratio is credible by separating fixed-strategy estimation error from search-aware inference and applying tools such as confidence intervals, Reality Check logic, and the Deflated Sharpe Ratio
- Explain why prediction quality and trading quality can diverge, and why IC alone is insufficient for selecting deployable strategies
## Sections
### 16.1 Backtesting as Falsification
This section gives the chapter its intellectual spine: a backtest is not proof that a strategy works, but a structured attempt to show that it fails under realistic assumptions. That framing matters because it shifts the reader from performance worship to disciplined skepticism, emphasizing leakage checks, execution realism, cost sensitivity, and regime robustness as the real standards of evidence.
### 16.2 What a Backtest Must Specify: The Trading Protocol
This section turns falsification into an operational checklist. It shows that results only become interpretable once timing, rebalancing, sizing, fills, costs, and constraints are specified explicitly, and it makes clear that many published disagreements are really disagreements about protocol rather than signal quality. For readers building or evaluating strategies, this is the section that defines what "credible" actually means.
### 16.3 Vectorized and Event-Driven Backtesting
This section clarifies one of the most misunderstood choices in practical research: the difference between array-based and sequential simulation. Its real value is not library comparison, but the idea of simulation semantics, showing that cash release, order sequencing, fill conventions, and state dependence can materially change results even when a strategy looks superficially identical.
### 16.4 Building the Baseline: A Non-ML Strategy You Can Trust
This section grounds the chapter in a deliberately simple ETF baseline so later ML results have something honest to beat. That matters editorially because it prevents the book from comparing sophisticated models only against weak or moving targets, and pedagogically because it forces the reader to see the full backtesting stack before additional model complexity enters.
### 16.5 Performance Reporting: The Core Metric Set
This section defines the reporting stack the rest of the book will rely on: return, risk, risk-adjusted, trading, and cost-impact metrics. Its main contribution is not the formulas alone, but the insistence that gross versus net results, Sharpe uncertainty, turnover, and baseline comparisons all belong in the same report if performance claims are to be taken seriously.
### 16.6 Regime-Based Backtest Diagnostics
This section shows why aggregate performance can be dangerously incomplete. By slicing results across volatility and trend states, it reveals when a strategy's "good" overall statistics are being driven by favorable environments while economically painful losses are concentrated in the regimes that matter most to investors.
### 16.7 The Specter of Overfitting at the Strategy Level
This section extends overfitting from the model stage to the strategy stage, where selection across many variants can inflate apparent Sharpe even when no true edge exists. It matters because it teaches readers that inference must account for the searched family, not just the chosen winner, making DSR, Reality Check logic, and related controls part of serious strategy evaluation.
### 16.8 Summary
This section closes the chapter by restating backtesting as a falsification discipline: a credible result is one that survives explicit protocol specification, regime-sliced diagnostics, and search-aware Sharpe inference. It sets up the next chapters (Ch17-19) on portfolio construction, transaction costs, and risk management, all of which build on the baseline established here.
## Running the Notebooks
```bash
# From the repository root
uv run python 16_strategy_simulation/<notebook>.py
# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "16_strategy_simulation"
```
## References
- **Ashwin Alankar et al.** (2023). [Fairy Tails: Lessons from 150 Years of Drawdowns](https://doi.org/10.3905/jpm.2023.1.503). *The Journal of Portfolio Management*.
- **Andrew Ang and Geert Bekaert** (2002). [International Asset Allocation With Regime Shifts](https://doi.org/10.1093/rfs/15.4.1137). *Review of Financial Studies*.
- **David H. Bailey and Marcos Lopez de Prado** (2012). [The Sharpe Ratio Efficient Frontier](https://doi.org/10.2139/ssrn.1821643).
- **David H. Bailey et al.** (2014). [Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance](https://doi.org/10.1090/noti1105). *Notices of the American Mathematical Society*.
- **David H. Bailey and Marcos Lopez de Prado** (2014). [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality](https://doi.org/10.2139/ssrn.2460551).
- **David H. Bailey et al.** (2015). [The Probability of Backtest Overfitting](https://doi.org/10.2139/ssrn.2326253).
- **Campbell R. Harvey et al.** (2016). [...and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059). *Review of Financial Studies*.
- **Jacques Joubert et al.** (2024). [Enhanced Backtesting for Practitioners](https://doi.org/10.3905/jpm.2024.1.637). *The Journal of Portfolio Management*.
- **Jacques Joubert et al.** (2024). [The Three Types of Backtests](https://doi.org/10.2139/ssrn.4897573).
- **Andrew W. Lo** (2002). [The Statistics of Sharpe Ratios](https://doi.org/10.2469/faj.v58.n4.2453).
- **R. David McLean and Jeffrey Pontiff** (2016). [Does Academic Research Destroy Stock Return Predictability?](https://doi.org/10.1111/jofi.12365). *Journal of Finance*.
- **Giuseppe A. Paleologo** (2025). The Elements of Quantitative Investing. *John Wiley & Sons*.
- **Marcos Lopez de Prado** (2018). Advances in Financial Machine Learning. *John Wiley & Sons*.
- **Marcos Lopez de Prado et al.** (2025). [How to Use the Sharpe Ratio](https://doi.org/10.2139/ssrn.5520741).
- **Halbert White** (2000). [A Reality Check for Data Snooping](https://www.jstor.org/stable/2999444). *Econometrica*.
