# Case Study: ETF Cross-Asset Exposures

This case study applies the ML4T workflow to 100 exchange-traded funds spanning equities, fixed income, commodities, currencies, and real estate. ETFs offer a clean laboratory for cross-asset rotation: standardized pricing, deep liquidity, and broad asset-class coverage at a single rebalance cadence.

The configuration is the most cost-favorable in the book — long-only rank-and-rebalance, monthly month-end decisions on a 21-day forward-return label, with a 5--15 bps-per-leg cost model. That cadence makes it the natural setting for the broadest model-family comparison in the book: linear, GBM, tabular DL, sequence DL, latent factors, and causal DML are all trained on the same feature panel. The teaching point is the gap between IC and Sharpe — the family with the highest rank correlation is not the family with the highest portfolio Sharpe — which makes ETFs the canonical setting for the "portfolio construction mediates prediction quality" thread that runs through Ch16--Ch20.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | Multi-asset ETFs |
| Frequency | Daily data, monthly decisions |
| Universe | 100 ETFs across 9 categories |
| History | 2006--2025 |
| Primary Label | fwd_ret_21d |
| CV Folds | 8 (10Y train, 1Y val) |
| Cost Model | Material (5--15 bps per leg) |

## Pipeline

| Stage | Notebook | Chapter | Description |
|-------|----------|---------|-------------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Universe breadth, point-in-time eligibility, horizon-cost feasibility, walk-forward demonstration |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 21-day and 5-day forward returns with walk-forward splits |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | Momentum, volatility, and cross-asset ranking features |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | ARIMA, HMM, and spectral features from walk-forward fits |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7--9 | Feature-label IC diagnostics across all engineered features |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | Ridge, LASSO, ElasticNet baseline for cross-asset momentum |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | LightGBM with Optuna testing non-linear interactions |
| Tabular DL | [`08_tabular_dl`](08_tabular_dl.ipynb) | Ch12 | TabM rank-1 adapter MLP ensemble |
| LSTM | [`09_dl_lstm`](09_dl_lstm.ipynb) | Ch13 | Temporal gating over sequential ETF return windows |
| TSMixer | [`10_dl_tsmixer`](10_dl_tsmixer.ipynb) | Ch13 | Cross-asset lead-lag patterns via time-feature mixing |
| Latent Factors | [`11_latent_factors`](11_latent_factors.ipynb) | Ch14 | Factor extraction across the ETF universe |
| Causal DML | [`12_causal_dml`](12_causal_dml.ipynb) | Ch15 | Does momentum cause future ETF returns or reflect confounders? |
| Model Analysis | [`13_model_analysis`](13_model_analysis.ipynb) | Ch11--15 | Cross-family IC comparison, checkpoint sensitivity, fold stability |
| Backtest | [`14_backtest`](14_backtest.ipynb) | Ch16 | Strategy simulation with falsification against equal-weight |
| Portfolio | [`15_portfolio_management`](15_portfolio_management.ipynb) | Ch17 | Score-weighted, risk-parity, inverse-vol, MVO, HRP, and conformal-weighted allocation |
| Costs | [`16_costs`](16_costs.ipynb) | Ch18 | Transaction cost impact on the momentum edge |
| Risk | [`17_risk_management`](17_risk_management.ipynb) | Ch19 | Position-level stop-loss, trailing-stop, and time-exit overlays calibrated against the in-sample MAE distribution |
| Strategy Analysis | [`18_strategy_analysis`](18_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with IC, Sharpe, and cost analysis |

## Key Results

**Signal quality**: Daily-pooled IC for the highest-Sharpe configuration is +0.052 [+0.009, +0.095] (HAC $t=2.37$, $p=0.018$, excludes zero on the positive side); pct-positive is 56.4%, modestly above coin flip. The rank-correlation prior is statistically resolved at the validation window, with magnitude small but credibly nonzero.

**Strategy-stage performance with CIs**: The cross-stage rank-1 configuration is `deep_learning/lstm_h64` on `fwd_ret_21d` resolved at the risk-overlay stage (risk-parity top-20 + MAE-calibrated trailing overlay `trailing_mae_p25_h20_4p3pct`). Validation Sharpe is +1.21 [+0.61, +1.87], PSR $p=0.00029$ — both Sharpe CI and PSR exclude zero on the positive side. Selection-adjusted DSR (effective-rank) is +0.072 ($p=4.1\mathrm{e}{-5}$) on the 20-variant family cohort, min_trl_periods is 461 (the 2016-day validation window clears the MinTRL bar by ≈4×), and PBO is 0.157 across 8 folds × 70 combinations — modest in-sample overfitting on the combinatorially shuffled folds but well inside the "low" band; the cross-stage label cohort (714 variants spanning every family × allocator × cost × overlay) carries DSR_ER +0.067 on the same leader.

**Holdout closure**: Validation→holdout Sharpe difference is $-0.14$ [$-1.87$, $+1.53$] ($p=0.886$, straddles zero with an extremely wide CI — the 481-day holdout cannot resolve decay magnitude under the disjoint-window pairing convention). Against an equal-weight benchmark, the holdout-period Sharpe difference is $-0.34$ [$-1.69$, $+0.81$] ($p=0.582$); the two-sided test does not reject. Holdout EW Sharpe runs at +1.36 — unusually high, driven by the 2024--2025 broad-equity rally where cross-asset rotation toward bonds and commodities gave back ground to a static equity-weighted universe.

**Friction floor**: Cost sensitivity scans 11 levels from 0 to 50 bps per leg. The highest-Sharpe configuration stays positive across the full grid; median Sharpe across all configurations stays positive through realistic ETF friction (≤5 bps). Both kill gates pass — validation Sharpe lower bound ≥ 0, and holdout strategy CI does not exclude zero negatively.

## Running

```bash
# From repo root
uv run python case_studies/etfs/01_feasibility_analysis.py
uv run python case_studies/etfs/02_labels.py
uv run python case_studies/etfs/03_financial_features.py
uv run python case_studies/etfs/04_model_based_features.py
uv run python case_studies/etfs/05_evaluation.py
uv run python case_studies/etfs/06_linear.py
uv run python case_studies/etfs/07_gbm.py
uv run python case_studies/etfs/08_tabular_dl.py
uv run python case_studies/etfs/09_dl_lstm.py
uv run python case_studies/etfs/10_dl_tsmixer.py
uv run python case_studies/etfs/11_latent_factors.py
uv run python case_studies/etfs/12_causal_dml.py
uv run python case_studies/etfs/13_model_analysis.py
uv run python case_studies/etfs/14_backtest.py
uv run python case_studies/etfs/15_portfolio_management.py
uv run python case_studies/etfs/16_costs.py
uv run python case_studies/etfs/17_risk_management.py
uv run python case_studies/etfs/18_strategy_analysis.py
```

## Run Log

Model training runs, predictions, and backtest results are tracked in a content-addressed registry under `run_log/registry.db`.
