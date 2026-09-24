# Case Study: US Equities Panel

This case study is the broadest cross-sectional equity workflow in the book. It uses daily OHLCV data from NASDAQ Data Link for ~3,200 US stocks spanning 1990 through 2018-Q1 to test whether weak per-stock signals translate into a tradable strategy when scaled across thousands of names. The Fundamental Law of Active Management is the operating frame: the per-stock edge is small, but breadth across the cross-section is supposed to compensate. The role of this case study is to hold that claim against measured signal quality, paired-bootstrap confidence intervals, and an explicit holdout window.

The pipeline is unusually long because the universe is unusually large. Sixteen walk-forward folds (10y train, 1y validation), the most folds of any case study, are paired with multi-horizon labels and a feature panel that mixes momentum, mean-reversion, volatility, liquidity, value proxies, and walk-forward temporal models. The strategy is a daily long-short top-K cross-sectional ranker with dollar-neutral construction and material era-dependent costs (15-30 bps pre-decimalization, 5-15 bps after). The question the strategy-analysis notebook answers is whether the gross signal that survives this much testing also survives selection-adjusted resampling and the 2016-2018 holdout.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | Broad US equities (NYSE/NASDAQ/AMEX) |
| Frequency | Daily |
| Universe | ~3,200 stocks (price > $5, ADV > $1M, point-in-time) |
| History | 1990-2018Q1 |
| Primary Label | fwd_ret_1d |
| CV Folds | 16 (10Y train, 1Y val) |
| Cost Model | Material (5-30 bps per leg, era-dependent + borrow) |

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Universe breadth per decision date, cost regime, move-to-cost scale, walk-forward folds | Nothing - the evidence stays in the notebook |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 1-day, 5-day, and 21-day forward returns | `labels/fwd_ret_1d.parquet`, `labels/fwd_ret_5d.parquet`, `labels/fwd_ret_21d.parquet`, each with a `.digest.json` sidecar |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | 62 cross-sectional factors: momentum, volatility, liquidity, value | `features/financial.parquet` |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | Walk-forward Wasserstein regime distance, FFD, GARCH features | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7--9 | Feature-label IC diagnostics across the full panel | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | Ridge, LASSO, ElasticNet on the full feature matrix | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | LightGBM grid across leaf profiles and loss functions | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | [`08_tabular_dl`](08_tabular_dl.ipynb) | Ch12 | TabM attention-style ensembling on the cross-section | Training runs and prediction sets; checkpoints under `run_log/training/tabular_dl/` |
| NLinear | [`09_dl_nlinear`](09_dl_nlinear.ipynb) | Ch13 | Minimal temporal baseline with last-value normalization | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| LSTM | [`10_dl_lstm`](10_dl_lstm.ipynb) | Ch13 | Sequential memory across daily return windows | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| TSMixer | [`11_dl_tsmixer`](11_dl_tsmixer.ipynb) | Ch13 | Time-mixing and feature-mixing across the 60-day lookback | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Weekly DL | [`12_dl_weekly`](12_dl_weekly.ipynb) | Ch13 | Weekly-cadence LSTM/NLinear comparison | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Latent Factors | [`13_latent_factors`](13_latent_factors.ipynb) | Ch14 | Index notebook for PCA + IPCA on the broad equity panel | Nothing - it reads the registry |
| PCA | [`13a_pca`](13a_pca.ipynb) | Ch14 | Static factor extraction from the return covariance | Training runs and prediction sets |
| IPCA | [`13b_ipca`](13b_ipca.ipynb) | Ch14 | Instrumented PCA with characteristic-conditioned loadings | Training runs and prediction sets |
| Causal DML | [`14_causal_dml`](14_causal_dml.ipynb) | Ch15 | Causal effect of 12-1 momentum on daily returns | A row in the registry's `causal_runs` |
| Model Analysis | [`15_model_analysis`](15_model_analysis.ipynb) | -- | Cross-model IC comparison and fold stability diagnostics | Nothing - it reads the registry |
| Backtest | [`16_backtest`](16_backtest.ipynb) | Ch16 | Daily long-short top-K strategy simulation | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`17_portfolio_management`](17_portfolio_management.ipynb) | Ch17 | Allocation sweep on the highest-IC GBM signal | One backtest run per allocation method, same artifact layout |
| Risk | [`18_risk_management`](18_risk_management.ipynb) | Ch19 | Position-level and portfolio-level risk overlays | One backtest run per overlay variant, same artifact layout |
| Costs | [`19_costs`](19_costs.ipynb) | Ch18 | Cost-grid sweep on the strategies the three earlier stages produced | One backtest run per cost level, same artifact layout |
| Holdout Predictions | [`20_holdout_predictions`](20_holdout_predictions.ipynb) | Ch20 | Refit of the selected configuration on history ending before the holdout window | One training run and one prediction set at `split='holdout'` |
| Holdout Backtest | [`21_holdout_backtest`](21_holdout_backtest.ipynb) | Ch20 | The holdout predictions traded under the selected allocator, overlay and cost level | One backtest run at `stage='holdout'`, same artifact layout |
| Strategy Analysis | [`22_strategy_analysis`](22_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment: signal, lineage, holdout, attribution | `results/strategy_assessment.json`, `20_strategy_synthesis/output/us_equities_panel/us_equities_panel_tearsheet.html` |

## Running

```bash
# From repo root
uv run python case_studies/us_equities_panel/01_feasibility_analysis.py
uv run python case_studies/us_equities_panel/02_labels.py
uv run python case_studies/us_equities_panel/03_financial_features.py
uv run python case_studies/us_equities_panel/04_model_based_features.py
uv run python case_studies/us_equities_panel/05_evaluation.py
uv run python case_studies/us_equities_panel/06_linear.py
uv run python case_studies/us_equities_panel/07_gbm.py
uv run python case_studies/us_equities_panel/08_tabular_dl.py
uv run python case_studies/us_equities_panel/09_dl_nlinear.py
uv run python case_studies/us_equities_panel/10_dl_lstm.py
uv run python case_studies/us_equities_panel/11_dl_tsmixer.py
uv run python case_studies/us_equities_panel/12_dl_weekly.py
uv run python case_studies/us_equities_panel/13_latent_factors.py
uv run python case_studies/us_equities_panel/13a_pca.py
uv run python case_studies/us_equities_panel/13b_ipca.py
uv run python case_studies/us_equities_panel/14_causal_dml.py
uv run python case_studies/us_equities_panel/15_model_analysis.py
uv run python case_studies/us_equities_panel/16_backtest.py
uv run python case_studies/us_equities_panel/17_portfolio_management.py
uv run python case_studies/us_equities_panel/18_risk_management.py
uv run python case_studies/us_equities_panel/19_costs.py
uv run python case_studies/us_equities_panel/20_holdout_predictions.py
uv run python case_studies/us_equities_panel/21_holdout_backtest.py
uv run python case_studies/us_equities_panel/22_strategy_analysis.py
```

The strategy-analysis notebook in `22_strategy_analysis.py` writes a full diagnostic tear sheet (`template="full"`) to the case study's gitignored output directory; readers regenerate it locally.

## Results

This README describes how the case study is built, not what it found. Results are
not restated here: the registry is rebuilt whenever the case study is re-derived,
and a number copied into prose stays correct only until the next rebuild.

[`22_strategy_analysis`](22_strategy_analysis.ipynb) reads the registry back and reports the selected
configuration with its interval evidence. That notebook, and the registry it reads,
are where a result comes from.

To read the results without training anything, download the published bundle, which
carries the registry and the artifacts behind it:

```bash
uv run python scripts/download_artifacts.py --cs us_equities_panel
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
