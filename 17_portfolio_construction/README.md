# Chapter 17: Portfolio Construction

The chapter explains why good forecasts are not yet portfolios. It frames allocation as the step that combines expected returns, risk estimates, and admissible-risk constraints into actual weights, leverage, and rebalancing choices. It matters because small modeling decisions at this stage can amplify a weak edge or destroy a strong one through concentration, unstable sizing, or excess turnover.

## Learning Objectives

- Formalize portfolio construction in terms of expected returns, covariance, constraints, leverage, and rebalancing choices
- Identify the allocator-specific evaluation metrics that complement the Chapter 16 backtest report, especially benchmark-relative performance, concentration, diversification, and implementation stability
- Explain why simple baselines such as equal weight, inverse volatility, and related heuristic allocators remain demanding benchmarks
- Apply mean-variance optimization with shrinkage, realistic constraints, and turnover-aware regularization
- Interpret Kelly sizing, especially fractional Kelly, as a log-growth principle for translating signal strength into position size
- Build and evaluate hierarchical allocations that prioritize diversification stability over direct covariance-matrix inversion
- Compare allocators under a common research protocol while limiting allocator-selection bias and other forms of overfitting

## Sections

### 17.1 From Signals to Positions: Defining the Allocation Problem

This section explains why good forecasts are not yet portfolios. It frames allocation as the step that combines expected returns, risk estimates, and admissible-risk constraints into actual weights, leverage, and rebalancing choices. It matters because small modeling decisions at this stage can amplify a weak edge or destroy a strong one through concentration, unstable sizing, or excess turnover.

### 17.2 A Portfolio Construction Workflow

This section turns portfolio construction into a documented research workflow rather than an optimizer call hidden in notebook code. The allocator term sheet, leakage controls, matched estimation windows, and separation of prediction from sizing make the allocation layer auditable and easier to diagnose. Readers should care because this is the section that keeps portfolio design from becoming a new source of hidden overfitting.

### 17.3 Portfolio Evaluation Metrics

This section extends the Chapter 16 backtest report with the metrics that actually distinguish allocators from one another: benchmark-relative performance, concentration, diversification, and implementation stability. It clarifies why Sharpe alone is not enough once the task is allocator comparison and why information ratio, active share, HHI, risk contributions, and leverage stability belong in the report. This matters because allocator quality shows up as much in portfolio shape and stability as in headline returns.

- [`01_portfolio_metrics`](01_portfolio_metrics.ipynb) — This notebook demonstrates comprehensive portfolio performance analysis using ml4t-diagnostic as a modern Plotly-based replacement for pyfolio. It shows how to compute 22+ metrics and visualize drawdowns, rolling performance, and benchmarks.

### 17.4 Baseline Allocators: Simple Heuristics That Are Hard to Beat

This section establishes robust baselines before moving to heavier optimization. Equal weight, inverse volatility, volatility targeting, score weighting, risk parity, and Kelly-style sizing are presented as serious competitors, not straw men. Readers should care because the chapter makes a strong practical claim: if a sophisticated allocator cannot reliably beat these simpler rules, its extra estimation burden is hard to justify.

### 17.5 Mean-Variance Optimization and the Markowitz Curse

This section presents MVO as the canonical but fragile optimization framework. It shows why noisy expected returns and unstable covariance inversion create extreme weights, then explains how shrinkage, factor structure, and explicit constraints act as regularizers rather than cosmetic fixes. This matters because MVO remains central in finance, but the chapter teaches readers how to use it with realism instead of textbook naivete.

- [`02_mean_variance_optimization`](02_mean_variance_optimization.ipynb) — This notebook demonstrates Modern Portfolio Theory (MPT) and mean-variance optimization, showing how to construct efficient frontiers, find optimal portfolios, and understand the practical limitations of classical MVO (the "Markowitz Curse"). Uses etfs data.
- [`03_robust_optimization`](03_robust_optimization.ipynb) — This notebook explores portfolio optimization techniques beyond classical MVO, including risk parity, minimum variance, minimum CVaR, and HRP using the Riskfolio-Lib library. Uses etfs data.
- [`04_kelly_criterion`](04_kelly_criterion.ipynb) — This notebook demonstrates the Kelly criterion for optimal position sizing, from binary outcomes through continuous returns to multi-asset portfolios. It shows why full Kelly is impractical and how fractional Kelly provides a safer alternative.

### 17.6 Hierarchical Allocation: HRP as a Stability-First Optimizer

This section introduces HRP as a stability-first alternative to matrix-inversion-based optimization. By clustering related assets and allocating hierarchically, it emphasizes diversification structure and avoids some of MVO's worst estimation pathologies. Readers should care because the section gives a concrete, modern answer to a recurring practical problem: how to build diversified portfolios when covariance estimates are noisy and optimizer instability is costly.

- [`06_hierarchical_risk_parity`](06_hierarchical_risk_parity.ipynb) — This notebook demonstrates Hierarchical Risk Parity, a modern portfolio construction method developed by Marcos López de Prado that addresses the fundamental flaws of Mean-Variance Optimization through clustering, quasi-diagonalization, and recursive bisection. Uses etfs data.

### 17.7 Regime-Adaptive Allocation Without Discrete Switching

This section argues for continuous adaptation of inputs, constraints, and sizing rather than brittle on/off regime switching. It uses conformal sizing and end-to-end portfolio learning examples to show that regime awareness is most convincing when it responds to a specific failure mode, such as rising uncertainty, deteriorating liquidity, or weak subperiod robustness. This matters because many readers will be tempted by regime language, and the chapter gives them a more defensible way to operationalize it.

- [`07_conformal_position_sizing`](07_conformal_position_sizing.ipynb) — This notebook demonstrates uncertainty-based position sizing using real conformal prediction intervals from the ML pipeline. We examine two contrasting cases: ETFs (where conformal weighting improves Sharpe by +5.5%) and CME Futures (where it degrades Sharpe by -24.8%).
- [`11_dl_portfolio_allocation`](11_dl_portfolio_allocation.ipynb) — This notebook implements end-to-end portfolio allocation using a neural network that directly maximizes the portfolio Sharpe ratio — bypassing the predict-then-optimize pipeline. The approach follows Zhang, Zohren & Roberts (2020).
- [`12_vlstm_portfolio`](12_vlstm_portfolio.ipynb) — This notebook implements the Saly-Kaufmann VLSTM, which couples a TFT-style variable-selection network to an LSTM encoder under the same volatility-targeted long-short layer and pooled-Sharpe loss. The variable selection mechanism gates per-feature contributions per asset, per timestep.
- [`13_deepm_regime_robust`](13_deepm_regime_robust.ipynb) — This notebook implements the DeePM framework (Wood, Roberts & Zohren 2026), which trains an end-to-end portfolio policy that is robust across market regimes. The key innovation is a SoftMin objective that penalizes poor performance in any rolling window, not just on average.

### 17.8 Comparing Allocators Under a Common Protocol

This section pulls the chapter together by insisting on a controlled comparison: same forecasts, same backtest rules, same constraints unless differences are intrinsic to the method. The ETF illustration and cross-case-study evidence reinforce the main practical lesson that allocator gains are context-dependent, often modest, and easy to overstate once search risk and execution are considered. Readers should care because this section prevents the chapter from turning into a leaderboard and instead teaches disciplined comparative research.

- [`09_allocator_comparison`](09_allocator_comparison.ipynb) — _Runtime ~6 min_. This capstone notebook compares allocation methods on equal footing, using identical data and backtest protocols to isolate the impact of the allocation choice. It demonstrates why simple heuristics often compete with sophisticated optimization.

The cross-case-study allocator comparison lives in Chapter 20: see [`20_strategy_synthesis/05_portfolio_allocation`](../20_strategy_synthesis/05_portfolio_allocation.ipynb).

## Running the Notebooks

```bash
# From the repository root
uv run python 17_portfolio_construction/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "17_portfolio_construction"
```

## References

- **Ashwin Alankar et al.** (2023). [Fairy Tails: Lessons from 150 Years of Drawdowns](https://doi.org/10.3905/jpm.2023.1.503). *The Journal of Portfolio Management*.
- **Andrew Ang and Geert Bekaert** (2002). [International Asset Allocation With Regime Shifts](https://doi.org/10.1093/rfs/15.4.1137). *Review of Financial Studies*.
- **Alexandre Antonov et al.** (2024). [Overcoming Markowitz's Instability with the Help of the Hierarchical Risk Parity (HRP): Theoretical Evidence](https://doi.org/10.2139/ssrn.4748151).
- **Clifford Asness et al.** (2017). [Contrarian Factor Timing Is Deceptively Difficult](https://doi.org/10.3905/jpm.2017.43.5.072). *Journal of Portfolio Management*.
- **Victor DeMiguel et al.** (2009). [Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?](https://doi.org/10.1093/rfs/hhm075). *The Review of Financial Studies*.
- **Ross French** (2024). [Sizing Matters: Optimal Scaling of Long and Short Exposures in Equity Portfolios](https://doi.org/10.3905/jpm.2024.1.596). *The Journal of Portfolio Management*.
- **Richard C.. Grinold and Ronald N.. Kahn** (2000). Active portfolio management: A quantitative approach for providing superior returns and controlling risk. *McGraw-Hill*.
- **Brian Hurst** (2010). Understanding Risk Parity.
- **Theis Ingerslev Jensen et al.** (2024). [Machine Learning and the Implementable Efficient Frontier](https://doi.org/10.2139/ssrn.4187217).
- **Jacques Joubert et al.** (2024). [Enhanced Backtesting for Practitioners](https://doi.org/10.3905/jpm.2024.1.637). *The Journal of Portfolio Management*.
- **Olaf Korn et al.** (2022). [Drawdown Measures: Are They All the Same?](https://doi.org/10.3905/jpm.2022.1.346). *The Journal of Portfolio Management*.
- **Tom Liu and Stefan Zohren** (2023). [Multi-Factor Inception: What to Do with All of These Features?](https://doi.org/10.48550/arXiv.2307.13832).
- **Sébastien Maillard et al.** (2008). [On the Properties of Equally-Weighted Risk Contributions Portfolios](https://doi.org/10.2139/ssrn.1271972). *SSRN Electronic Journal*.
- **Harry Markowitz** (1952). Portfolio selection. *The journal of finance*.
- **Gautier Marti et al.** (2021). [A Review of Two Decades of Correlations, Hierarchies, Networks and Clustering in Financial Markets](https://doi.org/10.1007/978-3-030-65459-7_10). *Springer International Publishing*.
- **Giuseppe A. Paleologo** (2025). The Elements of Quantitative Investing. *John Wiley & Sons*.
- **Marcos Lopez de Prado** (2016). [A Robust Estimator of the Efficient Frontier](https://doi.org/10.2139/ssrn.3469961).
- **Marcos Lopez de Prado** (2016). [Building Diversified Portfolios that Outperform Out-of-Sample](https://doi.org/10.2139/ssrn.2708678).
- **Thomas Raffinot** (2016). [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/abstract=2840729).
- **Yizhan Shu and John M. Mulvey** (2025). [Dynamic Factor Allocation Leveraging Regime-Switching Signals](https://doi.org/10.3905/jpm.2024.1.649). *The Journal of Portfolio Management*.
- **Vincent Tan and Stefan Zohren** (2025). [Estimation of Large Financial Covariances: A Cross-Validation Approach](https://doi.org/10.3905/jpm.2024.1.669). *The Journal of Portfolio Management*.
- **Yijie Wang et al.** (2025). [Machine Learning Meets Markowitz](https://doi.org/10.2139/ssrn.5947774).
- **Kieran Wood et al.** (2026). [DeePM: Regime-Robust Deep Learning for Systematic Macro Portfolio Management](https://doi.org/10.48550/arXiv.2601.05975).
- **Zihao Zhang et al.** (2020). [Deep Learning for Portfolio Optimization](https://doi.org/10.3905/jfds.2020.1.042). *The Journal of Financial Data Science*.
