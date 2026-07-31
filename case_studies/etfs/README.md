# Case Study: ETF Cross-Asset Exposures

This case study applies the ML4T workflow to 100 exchange-traded funds spanning equities, fixed income, commodities, currencies, and real estate. ETFs offer a clean laboratory for cross-asset rotation: standardized pricing, deep liquidity, and broad asset-class coverage at a single rebalance cadence.

The configuration is the most cost-favorable in the book — long-only rank-and-rebalance, monthly month-end decisions on a 21-day forward-return label, with a 5-15 bps-per-leg cost model. That cadence makes it the natural setting for the broadest model-family comparison in the book: linear, GBM, tabular DL, sequence DL, latent factors, and causal DML are all trained on the same feature panel. The teaching point is the gap between IC and Sharpe — the family with the highest rank correlation (latent factors) is not the family with the highest signal-stage Sharpe (the LSTM), and portfolio construction decides which signal survives to the highest cross-stage Sharpe — which makes ETFs the canonical setting for the "portfolio construction mediates prediction quality" thread that runs through Ch16-Ch20.

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
| Strategy Analysis | [`18_strategy_analysis`](18_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with IC, Sharpe, and cost analysis | `results/strategy_assessment.json`, `20_strategy_synthesis/output/etfs/etfs_tearsheet.html`; nothing in the registry |

## Key Results

**Signal quality**: Daily-pooled IC for the cross-stage rank-1 configuration (`latent_factors/sdf`) is +0.085 [+0.051, +0.119] (HAC $t=4.89$, $p=1.1\mathrm{e}{-6}$, excludes zero on the positive side); pct-positive is 64.9% across 2,016 validation days. The latent-factor SDF is the highest-IC family at the primary horizon — the LSTM has the higher signal-stage Sharpe (+0.89) but a lower IC (+0.052), and it is the SDF signal that carries through allocation and the risk overlay to the highest cross-stage Sharpe.

**Strategy-stage performance with CIs**: The cross-stage rank-1 configuration is `latent_factors/sdf` on `fwd_ret_21d` resolved at the risk-overlay stage (HRP top-20 + MAE-calibrated trailing overlay `trailing_mae_p25_h20_4p3pct`). Validation Sharpe is +1.36 [+0.69, +2.00], PSR $p=5.7\mathrm{e}{-5}$ — both Sharpe CI and PSR exclude zero on the positive side. Selection-adjusted DSR (effective-rank) is +0.081 ($p=5.7\mathrm{e}{-6}$) on the 20-variant overlay cohort, min_trl_periods is 366 (the 2016-day validation window clears the MinTRL bar by ≈5.5×), and the cross-stage label cohort (805 variants spanning every family × allocator × cost × overlay) carries DSR_ER +0.073 ($p=1.6\mathrm{e}{-6}$) on the same leader. PBO, however, is 0.629 across 8 folds × 70 combinations (median out-of-sample rank 13.5 of 20) — above the low band. The overfitting is localized to overlay selection: PBO is 0.06 at the allocation (HRP) stage and 0.00 at the cost stage, so the signal and portfolio ranks are stable out-of-sample while the choice among trailing-stop variants is not. Read the overlay's Sharpe contribution as illustrative rather than a tuned edge.

**Holdout closure**: Validation→holdout Sharpe difference is $-0.33$ [$-1.95$, $+1.30$] ($p=0.69$, straddles zero with a wide CI — the 481-day holdout cannot resolve decay magnitude under the disjoint-window pairing convention). Against an equal-weight benchmark, the holdout-period Sharpe difference is $-0.41$ [$-1.43$, $+0.44$] ($p=0.404$); the two-sided test does not reject. Holdout strategy Sharpe is +1.00 [$-0.42$, $+2.52$]; holdout EW Sharpe runs at +1.41 — unusually high, driven by the 2024-2025 broad-equity rally where cross-asset rotation toward bonds and commodities gave back ground to a static equity-weighted universe.

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
uv run python case_studies/etfs/16_costs.py
uv run python case_studies/etfs/17_risk_management.py
uv run python case_studies/etfs/18_strategy_analysis.py
```

## Run Log

Model training runs, predictions, and backtest results are tracked in a content-addressed registry under `run_log/registry.db`.
