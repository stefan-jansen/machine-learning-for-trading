# Chapter 19: Risk Management

This opening section reframes risk management as part of system design rather than post hoc reporting. It explains why a strategy is not deployable until its limits, escalation rules, and governance artifacts are defined in advance, auditable, and point-in-time safe.

## Learning Objectives

- Measure tail risk with VaR and CVaR, including regime-conditional estimates and liquidity-aware interpretation
- Evaluate path risk using drawdown depth, drawdown duration, recovery time, and related path-dependent metrics
- Decompose portfolio risk into market, factor, sector, geographic, and macro exposures to distinguish intended from unintended bets
- Design and interpret historical, hypothetical, and reverse stress tests that challenge return, cost, volatility, and correlation assumptions together
- Build adaptive risk controls, including volatility targeting, exposure caps, and position-level exits, using only information available at decision time
- Specify kill switches, drift monitoring, and governance artifacts that turn a backtested strategy into a deployable trading system

## Sections

### 19.1 Risk in the ML4T Workflow: From Backtest Winner to Tradable System

This opening section reframes risk management as part of system design rather than post hoc reporting. It explains why a strategy is not deployable until its limits, escalation rules, and governance artifacts are defined in advance, auditable, and point-in-time safe.

- [`01_var_cvar`](01_var_cvar.ipynb) — This notebook demonstrates VaR and CVaR computation methods—historical, parametric, and Cornish-Fisher—along with backtesting to validate estimates against realized losses. These tail risk metrics form the foundation for risk budgeting and regulatory reporting.
- [`06_stress_testing`](06_stress_testing.ipynb) — This notebook demonstrates stress testing methodologies including historical crisis replay, hypothetical scenarios, and Monte Carlo simulation. We treat historical crises as regime exemplars that reveal strategy behavior under extreme conditions.

### 19.2 A Practical Risk Taxonomy for Quant Strategies

This section gives readers a usable map of what can actually go wrong in production. By linking market, factor, leverage, concentration, liquidity, model, and operational risk to observable proxies and controls, it turns abstract risk categories into practical monitoring and portfolio rules.

- [`04_factor_exposure`](04_factor_exposure.ipynb) — This notebook decomposes portfolio risk into factor components using regression-based methods. We estimate exposures to market, size, value, profitability, and investment factors, and track how these exposures change over time.
- [`06_stress_testing`](06_stress_testing.ipynb) — This notebook demonstrates stress testing methodologies including historical crisis replay, hypothetical scenarios, and Monte Carlo simulation. We treat historical crises as regime exemplars that reveal strategy behavior under extreme conditions.

### 19.3 Measuring the Tail: VaR and CVaR

Here the chapter moves from ordinary variability to true downside risk. It explains why VaR alone is incomplete, why CVaR is more informative once losses breach a threshold, and why regime-conditional and liquidity-aware tail estimates are more realistic than a single unconditional number.

- [`01_var_cvar`](01_var_cvar.ipynb) — This notebook demonstrates VaR and CVaR computation methods—historical, parametric, and Cornish-Fisher—along with backtesting to validate estimates against realized losses. These tail risk metrics form the foundation for risk budgeting and regulatory reporting.

### 19.4 Drawdowns, Path Risk, and Time-to-Recovery

This section shifts the focus from point losses to lived investor experience. By covering drawdown depth, duration, recovery time, and related path-risk measures, it shows why strategies fail not only because they lose money, but because they lose it in ways allocators and operators cannot tolerate.

- [`01_var_cvar`](01_var_cvar.ipynb) — This notebook demonstrates VaR and CVaR computation methods—historical, parametric, and Cornish-Fisher—along with backtesting to validate estimates against realized losses. These tail risk metrics form the foundation for risk budgeting and regulatory reporting.
- [`02_exit_strategies`](02_exit_strategies.ipynb) — This notebook explores exit strategies that protect profits and limit losses. We compare fixed stops, trailing stops, volatility-adjusted exits, and hybrid approaches, analyzing whipsaw costs and the tradeoff between protection and premature exit.
- [`03_position_sizing_mae_mfe`](03_position_sizing_mae_mfe.ipynb) — This notebook demonstrates position sizing methods and MAE/MFE analysis for stop calibration. We implement fixed fractional and volatility-based sizing, then use trade excursion analysis to optimize stop placement.
- [`10_ml4t_backtest_risk_demo`](10_ml4t_backtest_risk_demo.ipynb) — Demonstrates the ml4t.backtest.risk module—the library implementation of risk management concepts discussed in Chapter 19. This provides production-ready stop-loss rules, rule composition, and portfolio-level kill switches.

### 19.5 Decomposing Risk: Factor, Sector, and Macro Exposure

This section asks where portfolio risk really comes from. It shows how factor, sector, geographic, and macro decomposition reveal whether performance reflects intended exposures, accidental bets, or risks that were never part of the original thesis.

- [`04_factor_exposure`](04_factor_exposure.ipynb) — This notebook decomposes portfolio risk into factor components using regression-based methods. We estimate exposures to market, size, value, profitability, and investment factors, and track how these exposures change over time.
- [`05_trade_shap_diagnostics`](05_trade_shap_diagnostics.ipynb) — This notebook demonstrates TradeShapAnalyzer from ml4t.diagnostic.evaluation, which connects SHAP explanations to trade outcomes for systematic improvement. Trade-level SHAP forensics help answer the key risk question: why did this trade fail?

### 19.6 Stress Testing and Scenario Analysis

This section broadens risk measurement beyond history-matching. By replaying crises, constructing hypothetical shock matrices, and using reverse stress tests, it shows how to identify vulnerabilities before markets force the question in real time.

- [`06_stress_testing`](06_stress_testing.ipynb) — This notebook demonstrates stress testing methodologies including historical crisis replay, hypothetical scenarios, and Monte Carlo simulation. We treat historical crises as regime exemplars that reveal strategy behavior under extreme conditions.

### 19.7 Adaptive Risk Controls Without Leakage

This is the chapter's operational core for live implementation. It explains how volatility targeting, exposure caps, turnover tightening, and position-level exits can adapt to changing conditions without smuggling future information into the backtest.

- [`02_exit_strategies`](02_exit_strategies.ipynb) — This notebook explores exit strategies that protect profits and limit losses. We compare fixed stops, trailing stops, volatility-adjusted exits, and hybrid approaches, analyzing whipsaw costs and the tradeoff between protection and premature exit.
- [`07_drift_detection`](07_drift_detection.ipynb) — This notebook demonstrates drift detection methods for production ML trading systems. When live data distributions diverge from training data, models silently degrade.
- [`08_ml_exit_signals`](08_ml_exit_signals.ipynb) — This notebook demonstrates a two-model architecture for exit timing. The entry model predicts high-return opportunities while the exit model predicts adverse moves.
- [`09_deep_hedging`](09_deep_hedging.ipynb) — This notebook demonstrates deep hedging (Buehler et al., 2019): a neural network learns hedging positions that minimize CVaR of terminal PnL under transaction costs. Where Section 19.7 builds adaptive risk controls from rules (vol targeting, regime caps, stops), this notebook shows how the same risk objective can be optimized end-to-end by a neural network — bridging the gap between measurement (Section 19.3) and learned control.
- [`11_systematic_risk_sweep`](11_systematic_risk_sweep.ipynb) — Demonstrates how to systematically optimize position-level exit rules (StopLoss, TakeProfit, TrailingStop) through 1D sweeps, 2D grid searches, and MAE/MFE-calibrated stops. Rather than hand-picking 3-5 configurations, we sweep the full parameter space and visualize Sharpe/Calmar trade-offs as heatmaps --- letting the data reveal the optimal risk regime.

### 19.8 Kill Switches and Risk Governance

This section turns metrics and controls into institutional process. It defines what failure conditions look like, how escalation should work, when re-research is required, and why drift detection and written risk governance are necessary for any strategy that is meant to survive contact with the market.

- [`07_drift_detection`](07_drift_detection.ipynb) — This notebook demonstrates drift detection methods for production ML trading systems. When live data distributions diverge from training data, models silently degrade.
- [`10_ml4t_backtest_risk_demo`](10_ml4t_backtest_risk_demo.ipynb) — Demonstrates the ml4t.backtest.risk module—the library implementation of risk management concepts discussed in Chapter 19. This provides production-ready stop-loss rules, rule composition, and portfolio-level kill switches.

The cross-case-study risk-overlay comparison lives in Chapter 20: see [`20_strategy_synthesis/07_regime_risk`](../20_strategy_synthesis/07_regime_risk.ipynb).

## Running the Notebooks

```bash
# From the repository root
uv run python 19_risk_management/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "19_risk_management"
```

## References

- **Ashwin Alankar et al.** (2023). [Fairy Tails: Lessons from 150 Years of Drawdowns](https://doi.org/10.3905/jpm.2023.1.503). *The Journal of Portfolio Management*.
- **Andrew Ang and Allan Timmermann** (2011). [Regime Changes and Financial Markets](https://doi.org/10.2139/ssrn.1919497).
- **Michele Leonardo Bianchi et al.** (2023). [Fat and Heavy Tails in Asset Management](https://doi.org/10.3905/jpm.2023.1.501). *The Journal of Portfolio Management*.
- **Tim Bollerslev** (1986). [Generalized autoregressive conditional heteroskedasticity](https://doi.org/10.1016/0304-4076(86)90063-1). *Journal of Econometrics*.
- **Aaron Brixton et al.** (2022). [A Changing Stock-Bond Correlation](https://www.aqr.com/Insights/Research/Journal-Article/A-Changing-Stock-Bond-Correlation). *AQR Alternative Thinking*.
- **Sid Browne et al.** (2023). [Timing and Sizing Skills of Systematic Strategies across Time and Economic Regimes](https://doi.org/10.3905/jpm.2023.1.505). *The Journal of Portfolio Management*.
- **H. Buehler et al.** (2019). [Deep hedging](https://doi.org/10.1080/14697688.2019.1571683). *Quantitative Finance*.
- **R. Cont** (2001). [Empirical properties of asset returns: stylized facts and statistical issues](https://doi.org/10.1080/713665670). *Quantitative Finance*.
- **Kent Daniel and Tobias J. Moskowitz** (2016). [Momentum crashes](https://doi.org/10.1016/j.jfineco.2015.12.002). *Journal of Financial Economics*.
- **Peter Reinhard Hansen and Asger Lunde** (2006). [Consistent ranking of volatility models](https://doi.org/10.1016/j.jeconom.2005.01.005). *Journal of Econometrics*.
- **Campbell R. Harvey et al.** (2022). [An Investor’s Guide to Crypto](https://doi.org/10.2139/ssrn.4124576).
- **Brian Hurst** (2010). Understanding Risk Parity.
- **Kevin Khang** (2022). [Toward Regime-Aware Risk Forecasts](https://doi.org/10.3905/jpm.2022.48.5.049). *The Journal of Portfolio Management*.
- **R. Douglas Martin et al.** (2024). [Minimum Downside Risk Portfolios](https://doi.org/10.3905/jpm.2024.1.642). *The Journal of Portfolio Management*.
- **Alan Moreira and Tyler Muir** (2017). [Volatility-Managed Portfolios](https://doi.org/10.1111/jofi.12513). *The Journal of Finance*.
- **Giuseppe A. Paleologo** (2025). The Elements of Quantitative Investing. *John Wiley & Sons*.
- **Andrew J. Patton** (2011). [Volatility forecast comparison using imperfect volatility proxies](https://doi.org/10.1016/j.jeconom.2010.03.034). *Journal of Econometrics*.
- **Marcos Lopez de Prado** (2018). Advances in Financial Machine Learning. *John Wiley & Sons*.
- **R. Tyrrell Rockafellar and Stanislav Uryasev** (2000). [Optimization of conditional value-at-risk](https://doi.org/10.21314/JOR.2000.038). *The Journal of Risk*.
- **G. William Schwert** (1989). [Why Does Stock Market Volatility Change Over Time?](https://doi.org/10.1111/j.1540-6261.1989.tb02647.x). *The Journal of Finance*.
- **Yizhan Shu and John M. Mulvey** (2025). [Dynamic Factor Allocation Leveraging Regime-Switching Signals](https://doi.org/10.3905/jpm.2024.1.649). *The Journal of Portfolio Management*.
- **{Board of Governors of the Federal Reserve System** (2011). [Supervisory Guidance on Model Risk Management - SR Letter 11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm).
- **Samir Varma** (2025). [The False Promise of Drawdown Rules: New Evidence and a Better Framework](https://doi.org/10.3905/jpm.2025.1.765). *The Journal of Portfolio Management*.
- **Hervé Zumbach and Gilles Zumbach** (2025). [A Quantitative Approach to Historical Stress Tests](https://doi.org/10.3905/jpm.2025.1.742). *The Journal of Portfolio Management*.
