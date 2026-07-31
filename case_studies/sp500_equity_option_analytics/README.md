# Case Study: S&P 500 Equity + Option Analytics

This case study trades S&P 500 equities using information from listed options. It combines daily
equity bars with implied-volatility level, skew, term structure, and variance-risk-premium features
for 633 stocks from 2017 through 2021. Decisions use Friday-close information, option features are
lagged one day, and trades execute at the following Monday open. The distinctive question is not
whether options contain information, but whether that information survives point-in-time feature
engineering, weekly portfolio construction, costs, risk controls, and a regime change.

## Dataset Profile

| Property | Value |
|---|---|
| Asset class | S&P 500 equities with option-derived predictors |
| Frequency | Daily inputs; weekly Friday-close decisions |
| History | 2017-2021 |
| Universe | 633 stocks with listed-options coverage |
| Inputs | AlgoSeek S&P 500 daily bars and the materialized daily options surface |
| Primary label | `fwd_ret_5d` |
| Evaluation | Two walk-forward folds; 10-session embargo; 2021 holdout |
| Execution | Friday close to Monday open; one-day lag on option features |
| Cost model | 13 bps round trip at the configured midpoint; 0-50 bps stress grid |
| Current evidence | v3.1 validation carrier: NLinear, score weighted, top 10, 5% trailing stop |

The corrected v3.1 validation Sharpe is 2.088 with a 95% interval of `[1.005, 3.117]`. The
matching NLinear holdout was not run because the 2021 holdout had already been observed on an older
IPCA lineage. The current notebooks therefore present validation evidence, label out-of-sample
efficacy unresolved, and make no deployment claim. The book-aligned v3.0 record remains preserved;
the two result versions are not mixed.

## Pipeline

| Stage | Notebook | Chapter | What it teaches | Writes |
|---|---|---:|---|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | 6 | Tests options coverage, weekly cadence, and whether equity trading costs leave room for research. | Nothing - the evidence stays in the notebook |
| Labels | [`02_labels`](02_labels.ipynb) | 7 | Builds five- and ten-day return, risk-adjusted, and direction labels with walk-forward boundaries. | `labels/fwd_ret_5d.parquet`, `labels/fwd_ret_10d.parquet`, `labels/fwd_ret_risk_adj_5d.parquet`, `labels/fwd_dir_5d.parquet`, `labels/fwd_dir_10d.parquet`, `config/cv_config.json` |
| Financial features | [`03_financial_features`](03_financial_features.ipynb) | 8 | Joins lagged IV surfaces to realized volatility, momentum, and liquidity features. | `features/financial.parquet` |
| Temporal features | [`04_model_based_features`](04_model_based_features.ipynb) | 9 | Produces forward-only GJR-GARCH volatility features and documents the pinned single-start feature vintage. | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | 7-9 | Audits coverage, staleness, daily IC, and HAC uncertainty before model selection. | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear models | [`06_linear`](06_linear.ipynb) | 11 | Establishes regularized linear and classification baselines across the label panel. | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| Gradient boosting | [`07_gbm`](07_gbm.ipynb) | 12 | Trains LightGBM configurations and records fold-complete validation predictions. | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular deep learning | [`08_tabular_dl`](08_tabular_dl.ipynb) | 12 | Evaluates TabM ensembles on the combined equity and options feature panel. | Training runs and prediction sets; checkpoints under `run_log/training/tabular_dl/` |
| LSTM | [`09_dl_lstm`](09_dl_lstm.ipynb) | 13 | Tests whether recurrent sequence structure improves on point-in-time features. | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| PatchTST | [`10_dl_patchtst`](10_dl_patchtst.ipynb) | 13 | Tests multi-scale patch attention and checkpoint stability. | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Latent-factor index | [`11_latent_factors`](11_latent_factors.ipynb) | 14 | Compares the latent-factor family and routes readers to each implementation. | Nothing - it reads the registry |
| PCA | [`11a_pca`](11a_pca.ipynb) | 14 | Fits PCA within each training fold on the persistent panel. | Training runs and prediction sets |
| IPCA | [`11b_ipca`](11b_ipca.ipynb) | 14 | Estimates characteristic-conditioned factors on the ragged panel. | Training runs and prediction sets |
| Conditional autoencoder | [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) | 14 | Learns nonlinear conditional factor exposures without crossing fold boundaries. | Training runs and prediction sets |
| Stochastic discount factor | [`11d_stochastic_discount_factor`](11d_stochastic_discount_factor.ipynb) | 14 | Estimates a neural SDF and compares checkpoint stability. | Training runs and prediction sets |
| Supervised autoencoder | [`11e_supervised_autoencoder`](11e_supervised_autoencoder.ipynb) | 14 | Learns return-supervised latent factors and reports uncertainty by checkpoint. | Training runs and prediction sets |
| Causal DML | [`12_causal_dml`](12_causal_dml.ipynb) | 15 | Estimates the `ivrv_spread` effect with walk-forward DML and panel-robust inference. | A row in the registry's `causal_runs` |
| Model analysis | [`13_model_analysis`](13_model_analysis.ipynb) | 11-15 | Compares full-coverage families using daily IC with HAC intervals and feature provenance. | Nothing - it reads the registry |
| Equal-weight baseline | [`14_backtest`](14_backtest.ipynb) | 16 | Runs equal-weight top-k baselines and applies coverage-aware selection. | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Allocation | [`15_portfolio_management`](15_portfolio_management.ipynb) | 17 | Tests five alternative allocators on the ten advancing model configurations. | One backtest run per allocation method, same artifact layout |
| Costs | [`16_costs`](16_costs.ipynb) | 18 | Replays one selected allocation lineage across the exact 17-point cost surface. | One backtest run per cost level, same artifact layout |
| Risk | [`17_risk_management`](17_risk_management.ipynb) | 19 | Compares 14 predeclared fixed controls with paired return uncertainty. | One backtest run per overlay variant, same artifact layout |
| Strategy assessment | [`18_strategy_analysis`](18_strategy_analysis.ipynb) | 20 | Separates corrected validation evidence from the historical, non-comparable holdout observation. | Nothing - it reads the registry |

## Running

Run the pipeline from the repository root in the locked environment. Set `ML4T_DATA_PATH` to a
directory containing `equities/market/sp500/daily_bars.parquet` and
`equities/market/sp500/options_surface_daily.parquet`. The equity bars require an AlgoSeek license;
the options-surface loader provides the materialized research file when available. Missing data
raises an error with acquisition instructions.

```bash
uv sync --frozen

for notebook in \
  01_feasibility_analysis 02_labels 03_financial_features 04_model_based_features \
  05_evaluation 06_linear 07_gbm 08_tabular_dl 09_dl_lstm 10_dl_patchtst \
  11_latent_factors 11a_pca 11b_ipca 11c_conditional_autoencoder \
  11d_stochastic_discount_factor 11e_supervised_autoencoder 12_causal_dml \
  13_model_analysis 14_backtest 15_portfolio_management 16_costs \
  17_risk_management 18_strategy_analysis
do
  uv run python "case_studies/sp500_equity_option_analytics/${notebook}.py"
done
```

A fresh production run is a several-hour workload on a CUDA-capable machine, with deep learning and
latent-factor training dominating. CPU execution is supported but materially slower. When the
shipped registry and artifacts are present, completed hashes are reused and the notebooks report
each cache hit. Do not skip a model family silently or start downstream from a partial baseline:
the greedy funnel is valid only after all model predictions and all equal-weight baselines exist.

The results source of truth is `run_log/registry.db`, with content-addressed artifacts under
`run_log/training/`, `run_log/predictions/`, and `run_log/backtest/`. Legacy `results/*.json` files
are not used.
