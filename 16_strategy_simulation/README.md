# Chapter 16: Strategy Simulation

A backtest is not proof that a strategy works. It is a structured attempt to show that the strategy
fails under realistic assumptions, and what survives that attempt is the only thing worth reporting.
Everything in this chapter follows from that reading: leakage checks, execution realism, cost
sensitivity and regime diagnostics are not hygiene applied after the fact, they are the evidence.

The chapter builds the simulator before it builds a strategy, so that every later result can be
traced to a stated rule about timing, fills, costs and accounting. It then measures how much each of
those rules is worth, which is what makes a disagreement between two engines readable rather than
mysterious. It closes on the question that undoes most reported Sharpe ratios: not whether this
strategy looks good, but how many were looked at before this one was chosen.

## Learning Objectives

- Formalize a backtest as an explicit trading protocol covering signal timing, execution,
  rebalancing, sizing, costs, constraints, data availability, and benchmark choice
- Distinguish vectorized and event-driven backtesting in terms of protocol semantics, state
  dependence, and appropriate use cases rather than treating one style as universally superior
- Build and interpret a transparent non-ML baseline strategy that provides a stable reference point
  for later model comparisons
- Evaluate a strategy using a core reporting stack that includes gross and net performance,
  drawdowns, turnover, baseline comparison, cost sensitivity, and regime-sliced diagnostics
- Assess whether a reported Sharpe ratio is credible by separating fixed-strategy estimation error
  from search-aware inference, using confidence intervals, the Deflated Sharpe Ratio, and the
  Rademacher Anti-Serum bound
- Explain why prediction quality and trading quality can diverge, and why the information
  coefficient alone is insufficient for selecting deployable strategies

## Sections

### 16.1 Backtesting as falsification

Sets the chapter's standard of evidence: a result counts when it survives an attempt to break it,
not when it looks impressive. No notebook belongs to this section alone; every notebook that follows
is an instance of it.

### 16.2 Specifying your backtest protocol

Turns falsification into an operational checklist. Results become interpretable only once timing,
rebalancing, sizing, fills, costs and constraints are written down, and many published disagreements
turn out to be disagreements about protocol rather than about signal quality.

- [`01_backtest_first_principles`](01_backtest_first_principles.ipynb) - Builds a portfolio
  simulator from Polars and NumPy with next-open fills, proportional fees and a cash constraint,
  then runs a ten-ETF risk-adjusted momentum rule and a 60/40 benchmark through it on identical
  dates. Scores the run against pass-or-fail criteria written before it, two of which it misses.
- [`08_signal_method_comparison`](08_signal_method_comparison.ipynb) - Converts registered model
  predictions into positions three ways, a fixed zero cutoff and two percentile rules, and measures
  how often each fires and how often it changes state. Neither rate is turnover, and the notebook
  says why.
- [`02_futures_backtesting`](02_futures_backtesting.ipynb) - Also introduces the protocol questions
  specific to futures: contract multipliers, per-contract commission and overnight session
  boundaries.

### 16.3 Vectorized and event-driven backtesting

The difference between array-based and sequential simulation, treated as a question about simulation
semantics rather than about libraries. Cash release, order sequencing, fill conventions and state
dependence all move the result while the strategy looks unchanged.

- [`03_single_asset_vectorbt`](03_single_asset_vectorbt.ipynb) - An RSI mean-reversion rule on daily
  BTC/USDT under VectorBT, with a buy-and-hold comparison on matched capital and costs and a
  threshold sweep whose surface is flat in one direction and not the other.
- [`04_single_asset_ml4t_backtest`](04_single_asset_ml4t_backtest.ipynb) - The same rule under the
  event-driven `ml4t-backtest` engine, matched to the vectorized run bar for bar, plus trade-log
  reconciliation and maximum favorable and adverse excursion per closed trade.
- [`05_stateful_strategies`](05_stateful_strategies.ipynb) - Three strategies whose logic cannot be
  expressed as an array operation: Kelly sizing that reads its own realized P&L, a two-leg pair
  trade that sizes the hedge from the lead leg's actual fill, and a drawdown circuit breaker.
- [`02_futures_backtesting`](02_futures_backtesting.ipynb) - Multiplier-aware futures simulation
  against a unit-multiplier counterfactual that replays the same intended contract targets.
- [`06_framework_parity`](06_framework_parity.ipynb) - One ETF strategy run as array arithmetic and
  through the sequential engine, matched on cost per dollar traded. The gap between them accumulates
  rather than oscillating.
- [`07_engine_divergence_anatomy`](07_engine_divergence_anatomy.ipynb) - Changes one configuration
  field at a time - share rounding, commission headroom, fill ordering, rebalance mode - and
  measures each. One field measures exactly zero because the executor makes it unreachable, which is
  a different fact from it being small.
- [`15_lean_engine_parity`](15_lean_engine_parity.ipynb),
  [`17_backtrader_zipline_engine_parity`](17_backtrader_zipline_engine_parity.ipynb),
  [`18_vectorbt_engine_parity`](18_vectorbt_engine_parity.ipynb) - The LEAN, Backtrader/Zipline and
  VectorBT rows of the cross-framework audit: fills, valuations and terminal value against the
  matching ML4T profile, with unsupported asset models disclosed rather than approximated.
- [`16_case_study_lean_parity`](16_case_study_lean_parity.ipynb) - The whole audit in one place,
  including the engine-only runtime comparison across all five engines and five case studies.

### 16.4 An auditable non-ML baseline

Grounds the chapter in a deliberately simple ETF rule, so that later model results have an honest
reference point rather than a weak or moving one.

- [`01_backtest_first_principles`](01_backtest_first_principles.ipynb) - Builds that baseline and
  reports it against the 60/40 mix on one protocol.

### 16.5 Understanding performance metrics

Defines the reporting stack the rest of the book relies on. The contribution is not the formulas but
the insistence that gross and net results, Sharpe uncertainty, turnover and a baseline comparison
belong in the same report.

- [`09_performance_reporting`](09_performance_reporting.ipynb) - Assembles that report from the
  protocol-matched BTC backtests: cumulative return against the benchmark, exposure, drawdown,
  rolling Sharpe, monthly returns and the daily return distribution.
- [`04_single_asset_ml4t_backtest`](04_single_asset_ml4t_backtest.ipynb) - Also reconciles the trade
  log against the engine's own accounting, which is what makes the report auditable.

### 16.6 Diagnosing the economic value

Aggregate performance can be dangerously incomplete. Two diagnostics say what it hides: how the
result splits across market conditions, and how much of it survives a higher cost assumption.

- [`10_regime_backtest_analysis`](10_regime_backtest_analysis.ipynb) - Slices the baseline by
  volatility and trend state, decomposes the worst drawdown by state in log returns, and finds the
  crisis tail no worse than an ordinary one - because a rule that buys bonds on a flat curve has
  few equity-crisis days left to lose money on.
- [`14_cost_sensitivity`](14_cost_sensitivity.ipynb) - Re-runs the strategy at every cost on a grid,
  reports the break-even fee two ways, and accounts for the gross-to-net gap in dollars. The fees
  paid are not the whole of what the fees cost.

### 16.7 Statistical inference and backtest overfitting

Extends overfitting from the model stage to the strategy stage, where selecting across many variants
inflates the apparent Sharpe even when no edge exists. Inference has to account for the family that
was searched, not only for the candidate that won.

- [`11_sharpe_ratio_inference`](11_sharpe_ratio_inference.ipynb) - The sampling distribution of the
  Sharpe ratio for a single fixed strategy, its standard error under skewness and kurtosis, and how
  much history a target Sharpe needs before it can be distinguished from zero.
- [`12_dsr_validation`](12_dsr_validation.ipynb) - The Deflated Sharpe Ratio and the probability of
  backtest overfitting: the same observed Sharpe deflated by the number of strategies tested, and
  the combinatorially symmetric cross-validation that estimates the overfitting probability.
- [`13_ras_protocol`](13_ras_protocol.ipynb) - The Rademacher Anti-Serum bound, which charges for
  the complexity of the whole candidate class rather than for the count of trials, and so penalizes
  every candidate instead of only the largest.

### 16.8 Summary

Restates backtesting as a falsification discipline and hands the baseline to Chapters 17 through 19.

## Running the Notebooks

```bash
# From the repository root
uv run python 16_strategy_simulation/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_chapter_notebooks.py -v -k "16_strategy_simulation"

# Headless (no display)
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python 16_strategy_simulation/<notebook>.py
```

> The four parity notebooks (15 through 18) read a committed audit artifact,
> `resources/framework_parity_audit.json`, rather than invoking the external engines. Regenerating
> that artifact needs LEAN, Backtrader, Zipline Reloaded and both VectorBT editions installed, which
> is outside the `ml4t` image.

## Dependencies

**Upstream**: Chapter 6 defines the strategy term sheet these backtests are scored against; Chapter 7
(§7.5) introduces the information coefficient, whose divergence from trading quality is one of this
chapter's recurring points.

**Downstream**: Chapter 17 (Portfolio Construction), Chapter 18 (Transaction Costs) and Chapter 19
(Risk Management) all build on the simulator and the reporting stack established here. Chapter 18
replaces the flat cost rate used in §16.6 with a model that depends on order size and market volume.

**Key libraries**: `ml4t-backtest` for the event-driven engine, its configuration surface and its
trade analytics; `ml4t-diagnostic` for portfolio statistics, Sharpe inference, the Deflated Sharpe
Ratio, the Rademacher bound and the reporting figures; `vectorbt` for the vectorized comparison; and
`polars`, `numpy` and `plotly` throughout.

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
