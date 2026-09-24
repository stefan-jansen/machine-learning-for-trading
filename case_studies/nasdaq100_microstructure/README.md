# Case Study: NASDAQ-100 Microstructure

This case study uses AlgoSeek TAQ-derived 15-minute bars for 115 NASDAQ-100 constituents to test whether microstructure signals -- order flow, quote staleness, relative spreads -- produce tradeable intraday alpha. This is the highest-frequency case in the book, and it is built around what a dominant cost floor does to a raw signal on the full universe, and what two disciplined adjustments, a cost-feasible universe screen and ensemble model selection, do about it. That iteration -- diagnose the cost problem, screen the universe, treat model selection as estimation under uncertainty -- is the lesson.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | US equities (NASDAQ-100) |
| Frequency | 15-minute bars |
| Universe | 115 stocks |
| History | 2020-2021 |
| Primary Label | fwd_ret_15m |
| CV Folds | 2 (6M train, 6M val) |
| Cost Model | per_share_plus_spread ($0.0035/share + measured half-spread; 5 bps friction floor) |

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Measured per-symbol round-trip cost, breadth and move clearance by horizon, return persistence, walk-forward demo | `liquidity_profile.parquet` |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 15-minute, 5-minute, and 60-minute forward returns | One parquet per label in `labels/` (`fwd_ret_15m` plus the `fwd_ret_5m`, `fwd_ret_60m`, `fwd_dir_15m` variants) |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | Order-flow, spread, volatility, and microstructure features | `features/financial.parquet` |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | Walk-forward temporal features for intraday patterns | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7--9 | Feature-label IC diagnostics for 66 financial and temporal features | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | Ridge baseline on the richest feature space in the book | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | LightGBM on 13M+ training samples at 15-minute frequency | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| NLinear | [`08_dl_nlinear`](08_dl_nlinear.ipynb) | Ch13 | Minimal temporal baseline for the intraday microstructure signal | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| LSTM | [`09_dl_lstm`](09_dl_lstm.ipynb) | Ch13 | Recurrent memory for short-lived order-flow and spread dynamics | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| TCN | [`10_dl_tcn`](10_dl_tcn.ipynb) | Ch13 | Dilated causal convolutions for intraday temporal patterns | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| PatchTST | [`11_dl_patchtst`](11_dl_patchtst.ipynb) | Ch13 | Multi-scale patch attention on minute-bar sequences | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Causal DML | [`12_causal_dml`](12_causal_dml.ipynb) | Ch15 | Does signed volume share cause future 15-minute returns? | A row in the registry's `causal_runs` |
| Model Analysis | [`13_model_analysis`](13_model_analysis.ipynb) | -- | Cross-model IC comparison and fold stability diagnostics | Nothing - it reads the registry |
| Backtest | [`14_backtest`](14_backtest.ipynb) | Ch16 | Strategy simulation designed to demonstrate cost-driven failure | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`15_portfolio_management`](15_portfolio_management.ipynb) | Ch17 | Allocation methods under dollar-neutral intraday constraints | One backtest run per allocation method, same artifact layout |
| Risk | [`16_risk_management`](16_risk_management.ipynb) | Ch19 | Intraday risk controls and position-level exit rules | One backtest run per overlay variant, same artifact layout |
| Costs | [`17_costs`](17_costs.ipynb) | Ch18 | Flagship cost analysis: spread, impact, and commission decomposition, over the leading run at any pre-cost stage | One backtest run per cost level, same artifact layout |
| Holdout Predictions | [`18_holdout_predictions`](18_holdout_predictions.ipynb) | Ch20 | Refits the selected configuration on history ending a label buffer before 2021-07-01 and writes its predictions over the holdout window | One training run, one prediction set at `split='holdout'` |
| Holdout Backtest | [`19_holdout_backtest`](19_holdout_backtest.ipynb) | Ch20 | Trades those predictions once, with the sizing and overlay the case study settled on | One backtest run at `stage='holdout'` |
| Strategy Analysis | [`20_strategy_analysis`](20_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with IC, Sharpe, and cost analysis | `results/strategy_assessment.json`, `20_strategy_synthesis/output/nasdaq100_microstructure/nasdaq100_microstructure_tearsheet.html` |

## Running

```bash
# From repo root
uv run python case_studies/nasdaq100_microstructure/01_feasibility_analysis.py
uv run python case_studies/nasdaq100_microstructure/02_labels.py
uv run python case_studies/nasdaq100_microstructure/03_financial_features.py
uv run python case_studies/nasdaq100_microstructure/04_model_based_features.py
uv run python case_studies/nasdaq100_microstructure/05_evaluation.py
uv run python case_studies/nasdaq100_microstructure/06_linear.py
uv run python case_studies/nasdaq100_microstructure/07_gbm.py
uv run python case_studies/nasdaq100_microstructure/08_dl_nlinear.py
uv run python case_studies/nasdaq100_microstructure/09_dl_lstm.py
uv run python case_studies/nasdaq100_microstructure/10_dl_tcn.py
uv run python case_studies/nasdaq100_microstructure/11_dl_patchtst.py
uv run python case_studies/nasdaq100_microstructure/12_causal_dml.py
uv run python case_studies/nasdaq100_microstructure/13_model_analysis.py
uv run python case_studies/nasdaq100_microstructure/14_backtest.py
uv run python case_studies/nasdaq100_microstructure/15_portfolio_management.py
uv run python case_studies/nasdaq100_microstructure/16_risk_management.py
uv run python case_studies/nasdaq100_microstructure/17_costs.py
uv run python case_studies/nasdaq100_microstructure/18_holdout_predictions.py
uv run python case_studies/nasdaq100_microstructure/19_holdout_backtest.py
uv run python case_studies/nasdaq100_microstructure/20_strategy_analysis.py
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
uv run python scripts/download_artifacts.py --cs nasdaq100_microstructure
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
