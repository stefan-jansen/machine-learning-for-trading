# Case Study: ETF Cross-Asset Exposures

This case study applies the ML4T workflow to 100 exchange-traded funds spanning equities, fixed income, commodities, currencies, and real estate. ETFs offer a clean laboratory for cross-asset rotation: standardized pricing, deep liquidity, and broad asset-class coverage at a single rebalance cadence.

The configuration is the most cost-favorable in the book — long-only rank-and-rebalance, monthly month-end decisions on a 21-day forward-return label, with a 5-15 bps-per-leg cost model. That cadence makes it the natural setting for the broadest model-family comparison in the book: linear, GBM, tabular DL, sequence DL, latent factors, and causal DML are all trained on the same feature panel. The teaching point is the gap between IC and Sharpe: the family that ranks best by rank correlation need not be the family that ranks best by traded Sharpe, and portfolio construction decides which prediction set survives to the highest cross-stage Sharpe. That makes ETFs the setting for the "portfolio construction mediates prediction quality" thread running through Ch16-Ch20.

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
| NLinear | [`10a_dl_nlinear`](10a_dl_nlinear.ipynb) | Ch13 | The smallest sequence model that still counts as one: no gate, no recurrence, no mixing layer, so it is what the two above have to beat | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Latent factors (index) | [`11_latent_factors`](11_latent_factors.ipynb) | Ch14 | Index of the five latent-factor notebooks below; prints their best registered IC | Nothing - it reads the registry |
| PCA | [`11a_pca`](11a_pca.ipynb) | Ch14 | Principal components as the persistent-ID baseline of the suite | Training runs and prediction sets |
| IPCA | [`11b_ipca`](11b_ipca.ipynb) | Ch14 | Instrumented PCA with characteristics loading the factors | Training runs and prediction sets |
| Conditional autoencoder | [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) | Ch14 | Nonlinear beta function estimated jointly with the factors | Training runs and prediction sets |
| SDF | [`11d_stochastic_discount_factor`](11d_stochastic_discount_factor.ipynb) | Ch14 | Stochastic discount factor: factors estimated to price the cross-section | Training runs and prediction sets |
| Supervised autoencoder | [`11e_supervised_autoencoder`](11e_supervised_autoencoder.ipynb) | Ch14 | Autoencoder trained against the return label rather than reconstruction | Training runs and prediction sets |
| Causal DML | [`12_causal_dml`](12_causal_dml.ipynb) | Ch15 | Does momentum cause future ETF returns or reflect confounders? | A row in the registry's `causal_runs` |
| Model Analysis | [`13_model_analysis`](13_model_analysis.ipynb) | Ch11-15 | Cross-family IC comparison, checkpoint sensitivity, fold stability | Nothing - it reads the registry |
| Backtest | [`14_backtest`](14_backtest.ipynb) | Ch16 | Strategy simulation with falsification against equal-weight | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`15_portfolio_management`](15_portfolio_management.ipynb) | Ch17 | Score-weighted, risk-parity, inverse-vol, MVO, HRP, and conformal-weighted allocation | One backtest run per allocation method, same artifact layout |
| Risk | [`16_risk_management`](16_risk_management.ipynb) | Ch19 | Position-level stop-loss, trailing-stop, and time-exit overlays calibrated against the in-sample MAE distribution | One backtest run per overlay variant, same artifact layout |
| Costs | [`17_costs`](17_costs.ipynb) | Ch18 | Transaction cost impact on the momentum edge | One backtest run per cost level, same artifact layout |
| Holdout Predictions | [`18_holdout_predictions`](18_holdout_predictions.ipynb) | Ch20 | Refit the selected configuration on history ending before the holdout window | one `training_runs` row and one `prediction_sets` row at `split='holdout'` |
| Holdout Backtest | [`19_holdout_backtest`](19_holdout_backtest.ipynb) | Ch20 | Trade the holdout predictions with the sizing, overlay and costs already settled | one `backtest_runs` row at `stage='holdout'` |
| Strategy Analysis | [`20_strategy_analysis`](20_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with IC, Sharpe, and cost analysis | `results/strategy_assessment.json`, `20_strategy_synthesis/output/etfs/etfs_tearsheet.html`; nothing in the registry |

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
uv run python case_studies/etfs/10a_dl_nlinear.py
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

## Results

This README describes how the case study is built, not what it found. Results are
not restated here: the registry is rebuilt whenever the case study is re-derived,
and a number copied into prose stays correct only until the next rebuild.

[`20_strategy_analysis`](20_strategy_analysis.ipynb) reads the registry back and reports the selected
configuration with its interval evidence. That notebook, and the registry it reads,
are where a result comes from.

To read the results without training anything, download the published bundle, which
carries the registry and the artifacts behind it:

```bash
uv run python scripts/download_artifacts.py --cs etfs
```

Two bundles are published, and they are separate generations rather than revisions
of one another. `v3.1.0-artifacts` is current and is what the command above fetches.
`v3.0.0-artifacts` holds the results as first published. The 3.1 rebuild re-keyed
every content-addressed hash, so a hash taken from one bundle does not resolve in
the other.

## Run Log

`run_log/registry.db` records every training run, prediction set and backtest,
each addressed by a hash of the specification that produced it. The artifacts sit
beside it under `run_log/training/`, `run_log/predictions/` and `run_log/backtest/`.
