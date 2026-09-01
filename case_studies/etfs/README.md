# Case Study: ETF Cross-Asset Exposures

This case study applies the ML4T workflow to 100 exchange-traded funds spanning equities, fixed income, commodities, currencies, and real estate. ETFs offer a clean laboratory for cross-asset rotation: standardized pricing, deep liquidity, and broad asset-class coverage at a single rebalance cadence.

The configuration is the most cost-favorable in the book — long-only rank-and-rebalance, monthly month-end decisions on a 21-day forward-return label, with a 5-15 bps-per-leg cost model. That cadence makes it the natural setting for the broadest model-family comparison in the book: linear, GBM, tabular DL, sequence DL, latent factors, and causal DML are all trained on the same feature panel. The teaching point is the gap between IC and Sharpe — the family with the highest rank correlation (latent factors) is not the family with the highest baseline backtest Sharpe (GBM), and portfolio construction decides which prediction set survives to the highest cross-stage Sharpe — which makes ETFs the canonical setting for the "portfolio construction mediates prediction quality" thread that runs through Ch16-Ch20.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | Multi-asset ETFs |
| Frequency | Daily data, monthly decisions |
| Universe | 100 ETFs across 9 categories |
| History | 2006-2025 |
| Primary Label | fwd_ret_21d |
| CV Folds | 8 (10Y train, 1Y val) |
| Cost Model | Material (5-15 bps per leg) |

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Universe breadth, point-in-time eligibility, move scale against cost, carrier persistence, walk-forward demonstration | `eligibility.csv` |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 21-day and 5-day forward returns, sealed on the label endpoint; folds are derived from `setup.yaml` and the label timeline, not written here | `labels/fwd_ret_21d.parquet`, `labels/fwd_ret_5d.parquet` (each with a `.digest.json` sidecar) |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | Momentum, volatility, and cross-asset ranking features | `features/financial.parquet` |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | ARIMA, HMM, and spectral features from walk-forward fits | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7-9 | Feature-label IC diagnostics across all engineered features | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | Ridge, LASSO, ElasticNet baseline for cross-asset momentum | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | LightGBM with Optuna testing non-linear interactions | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | [`08_tabular_dl`](08_tabular_dl.ipynb) | Ch12 | TabM rank-1 adapter MLP ensemble | Training runs and prediction sets; checkpoints under `run_log/training/tabular_dl/` |
| LSTM | [`09_dl_lstm`](09_dl_lstm.ipynb) | Ch13 | Temporal gating over sequential ETF return windows | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| TSMixer | [`10_dl_tsmixer`](10_dl_tsmixer.ipynb) | Ch13 | Cross-asset lead-lag patterns via time-feature mixing | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Latent factors (index) | [`11_latent_factors`](11_latent_factors.ipynb) | Ch14 | Index of the five latent-factor notebooks below; prints their best registered IC | Nothing - it reads the registry |
| PCA | [`11a_pca`](11a_pca.ipynb) | Ch14 | Principal components as the persistent-ID baseline of the suite | Training runs and prediction sets |
| IPCA | [`11b_ipca`](11b_ipca.ipynb) | Ch14 | Instrumented PCA with characteristics loading the factors | Training runs and prediction sets |
| Conditional autoencoder | [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) | Ch14 | Nonlinear beta function estimated jointly with the factors | Training runs and prediction sets |
| SDF | [`11d_stochastic_discount_factor`](11d_stochastic_discount_factor.ipynb) | Ch14 | Stochastic discount factor - the highest-IC family here | Training runs and prediction sets |
| Supervised autoencoder | [`11e_supervised_autoencoder`](11e_supervised_autoencoder.ipynb) | Ch14 | Autoencoder trained against the return label rather than reconstruction | Training runs and prediction sets |
| Causal DML | [`12_causal_dml`](12_causal_dml.ipynb) | Ch15 | Does momentum cause future ETF returns or reflect confounders? | A row in the registry's `causal_runs` |
| Model Analysis | [`13_model_analysis`](13_model_analysis.ipynb) | Ch11-15 | Cross-family IC comparison, checkpoint sensitivity, fold stability | Nothing - it reads the registry |
| Backtest | [`14_backtest`](14_backtest.ipynb) | Ch16 | Strategy simulation with falsification against equal-weight | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`15_portfolio_management`](15_portfolio_management.ipynb) | Ch17 | Score-weighted, risk-parity, inverse-vol, MVO, HRP, and conformal-weighted allocation | One backtest run per allocation method, same artifact layout |
| Costs | [`16_costs`](16_costs.ipynb) | Ch18 | Transaction cost impact on the momentum edge | One backtest run per cost level, same artifact layout |
| Risk | [`17_risk_management`](17_risk_management.ipynb) | Ch19 | Position-level stop-loss, trailing-stop, and time-exit overlays calibrated against the in-sample MAE distribution | One backtest run per overlay variant, same artifact layout |
| Holdout Predictions | [`18_holdout_predictions`](18_holdout_predictions.ipynb) | Ch20 | Refit the selected configuration on history ending before the holdout window | one `training_runs` row and one `prediction_sets` row at `split='holdout'` |
| Holdout Backtest | [`19_holdout_backtest`](19_holdout_backtest.ipynb) | Ch20 | Trade the holdout predictions with the sizing, overlay and costs already settled | one `backtest_runs` row at `stage='holdout'` |
| Strategy Analysis | [`20_strategy_analysis`](20_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with IC, Sharpe, and cost analysis | `results/strategy_assessment.json`, `20_strategy_synthesis/output/etfs/etfs_tearsheet.html`; nothing in the registry |

## Key Results

**Signal quality**: The highest-IC configuration at the primary horizon is `latent_factors/cae`, whose daily-pooled IC is +0.081 [+0.033, +0.129] (HAC $t=3.30$, $p=0.001$, lag 20, 1,995 validation days, pct-positive 62.4%). It is not the configuration the pipeline selects. The cross-stage rank-1 is `gbm/default_mae`, and its IC is +0.024 [$-0.012$, +0.059] (HAC $t=1.31$, $p=0.190$) — indistinguishable from zero on the same window. Best baseline backtest Sharpe by family runs gbm +0.83, sequence DL +0.82, tabular DL +0.70, latent factors +0.68, linear +0.54, so the family ranking by rank correlation and the family ranking by traded Sharpe disagree at the top. This is the case study's teaching point stated at its sharpest: selection is by validation backtest Sharpe, IC selects nothing, and here the two point at different models.

**Strategy-stage performance with CIs**: The cross-stage rank-1 configuration is `gbm/default_mae` on `fwd_ret_21d` resolved at the risk-overlay stage, checkpoint iteration 400 (validation backtest `fb287093aa02`, prediction set `ac905a96646f`). Validation Sharpe is +1.150 [+0.495, +1.900], PSR $p=0.001$ — both the Sharpe CI and PSR exclude zero on the positive side. Sortino is +1.74, annualized return +16.5% [+6.2%, +29.6%], max drawdown $-22.5\%$ [$-33.5\%$, $-12.0\%$], on a 13-period bootstrap block over 1,000 resamples. Selection-adjusted DSR (effective-rank) is +0.063 ($p=4.0\mathrm{e}{-4}$) on the 20-variant overlay cohort, and min_trl_periods is 573 against a 1,995-day validation window. PBO is 0.471 across 8 folds × 70 combinations — mid-band, not low. The search behind that leader is 771 baseline backtests with a median Sharpe of +0.484 and a p90 of +0.617, all positive.

**Holdout closure**: The holdout was executed on 2026-08-31 and it disconfirms the selection. The selected configuration was refit over history ending 2023-11-29 and predicted over 2024-01-01 to 2025-12-31 (training `882301dc3655`, distinct from the validation training hash, so the refit is registered rather than asserted; prediction set `d0fe6611feae`; backtest `d2e65da031af`). Holdout Sharpe is +0.232 over 502 sessions, against +1.150 on validation — CAGR 2.2%, max drawdown $-20.6\%$, win rate 55.0%, 111 trades against 436. The decay itself is not resolvable: the validation-to-holdout Sharpe difference is $-0.915$ [$-2.56$, +0.78] ($p=0.277$), straddling zero on a window this short. **The comparison against the benchmark is resolvable and it goes the wrong way.** The equal-weight universe ran at +0.625 on validation and +1.358 on the holdout. Those two full-window figures do not subtract to the paired statistic and are not meant to: the strategy-minus-benchmark comparison is computed on the two series' common support, with a 21-period bootstrap block over 2,000 resamples, and it comes to $-1.221$ [$-2.346$, $-0.275$] ($p=0.018$, information ratio $-1.19$, probability the strategy wins 0.4%). That excludes zero on the negative side. The same paired comparison on validation runs the other way but not cleanly, at $+0.559$ [$-0.021$, $+1.061$] ($p=0.044$) — its lower bound already sat just under zero before the holdout was opened. The 2024-2025 window rewarded simply holding the universe, and the selected configuration gave that back.

**Friction floor**: The cost sweep prices the configuration this case study reports, `gbm/default_mae` under the `trailing_mae_p10_h40_9p8pct` overlay, across two cost regimes. Sharpe runs from +1.223 with no friction to +0.488 at 50 bps per leg, and from +1.212 to +0.872 at 10c of half-spread. Every level of both grids stays positive and neither reaches breakeven, so friction is not what decides this case study. **The kill gates do not both pass.** The validation gate passes — the Sharpe CI lower bound is +0.495, above zero. The holdout gate fails: the strategy-versus-benchmark CI excludes zero negatively, which is the condition the gate exists to catch. A configuration whose IC cannot be distinguished from zero, selected on validation backtest Sharpe out of 771 candidates, did not survive the window nothing was selected on. That is the result this case study reports, and Chapter 20 is where it is read.

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
uv run python case_studies/etfs/11a_pca.py
uv run python case_studies/etfs/11b_ipca.py
uv run python case_studies/etfs/11c_conditional_autoencoder.py
uv run python case_studies/etfs/11d_stochastic_discount_factor.py
uv run python case_studies/etfs/11e_supervised_autoencoder.py
uv run python case_studies/etfs/11_latent_factors.py   # summarizes 11a-11e
uv run python case_studies/etfs/12_causal_dml.py
uv run python case_studies/etfs/13_model_analysis.py
uv run python case_studies/etfs/14_backtest.py
uv run python case_studies/etfs/15_portfolio_management.py
uv run python case_studies/etfs/16_risk_management.py
uv run python case_studies/etfs/17_costs.py
uv run python case_studies/etfs/18_holdout_predictions.py
uv run python case_studies/etfs/19_holdout_backtest.py
uv run python case_studies/etfs/20_strategy_analysis.py
```

## Run Log

Model training runs, predictions, and backtest results are tracked in a content-addressed registry under `run_log/registry.db`.
