# Crypto Perpetuals Funding

This case study uses Binance perpetual futures to examine an asset-class-specific return source:
the transfer between long and short positions at each 8-hour funding settlement. Nineteen
perpetuals create the book's smallest cross-section and highest non-intraday decision frequency.
The pipeline therefore emphasizes completed-bar timing, official funding cash flows, transaction
costs, and uncertainty from only two validation folds.

## At a Glance

| Property | Value |
|---|---|
| Asset class | Crypto perpetual futures |
| Frequency | 8-hourly, aligned to funding settlements |
| Universe | 19 perpetual pairs |
| History | 2020-2025 |
| Primary label | `fwd_ret_8h` |
| Validation design | 2 folds, 2-year train and 1-year validation |
| Cost model | 2 bps maker and 4 bps taker |

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|---|---|---|---|---|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Checks universe breadth at the funding timestamp, move scale against the fee, premium persistence, and the walk-forward folds. | Nothing; the contract list is fixed in `setup.yaml` |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | Builds forward returns and class labels without admitting holdout-ending observations; folds are derived from `setup.yaml` and the label timeline, not written here | One parquet per label in `labels/` (`fwd_ret_8h` plus the `fwd_ret_24h`, `fwd_dir_8h`, `fwd_dir_8h_3c` variants), each with a `.digest.json` sidecar |
| Financial features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | Produces 39 premium, funding, momentum, volatility, and liquidity features. | `features/financial.parquet` |
| Model-based features | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | Adds five fold-specific volatility and regime features fit on prior data. | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7-9 | Evaluates the exact 44-feature training frame on the canonical label clock. | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear models | [`06_linear`](06_linear.ipynb) | Ch11 | Fits complete Ridge, Lasso, and ElasticNet validation surfaces. | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| Gradient boosting | [`07_gbm`](07_gbm.ipynb) | Ch12 | Trains the CUDA LightGBM grid and preserves physical boosters and predictions. | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular deep learning | [`08_tabular_dl`](08_tabular_dl.ipynb) | Ch12 | Trains TabM checkpoints on the same fingerprinted frame. | Training runs and prediction sets; checkpoints under `run_log/training/tabular_dl/` |
| LSTM | [`09_dl_lstm`](09_dl_lstm.ipynb) | Ch13 | Evaluates causal 60-bar recurrent sequences on CUDA. | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| TCN | [`10_dl_tcn`](10_dl_tcn.ipynb) | Ch13 | Evaluates dilated causal convolutions on the same sequence contract. | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Causal DML | [`11_causal_dml`](11_causal_dml.ipynb) | Ch15 | Tests whether the basis premium has a causal interpretation after adjustment. | A row in the registry's `causal_runs` |
| Model analysis | [`12_model_analysis`](12_model_analysis.ipynb) | Ch12-15 | Compares four current family leaders on one physical validation panel. | Nothing - it reads the registry |
| Backtest | [`13_backtest`](13_backtest.ipynb) | Ch16 | Runs every prediction set equally weighted on completed-bar prices and official funding. | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`14_portfolio_management`](14_portfolio_management.ipynb) | Ch17 | Sizes the survivors six ways, with point-in-time allocation on the small cross-section. | One backtest run per variant, same artifact layout |
| Risk | [`15_risk_management`](15_risk_management.ipynb) | Ch19 | Fourteen ways of leaving a position early, fixed and pre-validation-calibrated. | One backtest run per variant, same artifact layout |
| Costs | [`16_costs`](16_costs.ipynb) | Ch18 | How much friction the survivor absorbs: price-only versus funding-inclusive breakevens, on the configuration risk management selected. | One backtest run per variant, same artifact layout |
| Holdout Predictions | [`17_holdout_predictions`](17_holdout_predictions.ipynb) | Ch20 | Refits the selected configuration on history ending before the holdout window, because a result selected on validation cannot also be the evidence that the selection was sound | one `training_runs` row and one `prediction_sets` row at `split='holdout'` |
| Holdout Backtest | [`18_holdout_backtest`](18_holdout_backtest.ipynb) | Ch20 | Trades the holdout predictions with the sizing, overlay and costs already settled | one `backtest_runs` row at `stage='holdout'` |
| Synthesis | [`19_strategy_analysis`](19_strategy_analysis.ipynb) | Ch20 | Makes the one choice the case study exists to make, then reports how many candidates it was the best of, how wide its interval is, and what survives correcting for the selection | Nothing - it reads the registry |

## Running

Run notebooks from the repository root. Notebooks 07-10 require CUDA; the rest are CPU.

```bash
uv run python case_studies/crypto_perps_funding/01_feasibility_analysis.py
uv run python case_studies/crypto_perps_funding/02_labels.py
uv run python case_studies/crypto_perps_funding/03_financial_features.py
uv run python case_studies/crypto_perps_funding/04_model_based_features.py
uv run python case_studies/crypto_perps_funding/05_evaluation.py
uv run python case_studies/crypto_perps_funding/06_linear.py
uv run python case_studies/crypto_perps_funding/07_gbm.py
uv run python case_studies/crypto_perps_funding/08_tabular_dl.py
uv run python case_studies/crypto_perps_funding/09_dl_lstm.py
uv run python case_studies/crypto_perps_funding/10_dl_tcn.py
uv run python case_studies/crypto_perps_funding/11_causal_dml.py
uv run python case_studies/crypto_perps_funding/12_model_analysis.py
uv run python case_studies/crypto_perps_funding/13_backtest.py
uv run python case_studies/crypto_perps_funding/14_portfolio_management.py
uv run python case_studies/crypto_perps_funding/15_risk_management.py
uv run python case_studies/crypto_perps_funding/16_costs.py
uv run python case_studies/crypto_perps_funding/17_holdout_predictions.py
uv run python case_studies/crypto_perps_funding/18_holdout_backtest.py
uv run python case_studies/crypto_perps_funding/19_strategy_analysis.py
```

The order matters downstream of 12: the funnel is valid only once every model family has
registered its predictions and every equal-weight baseline exists, and 17 and 18 open the holdout
once, on the configuration 19 reports.

## Results

This README describes how the case study is built, not what it found. Results are
not restated here: the registry is rebuilt whenever the case study is re-derived,
and a number copied into prose stays correct only until the next rebuild.

[`19_strategy_analysis`](19_strategy_analysis.ipynb) reads the registry back and reports the selected
configuration with its interval evidence. That notebook, and the registry it reads,
are where a result comes from.

To read the results without training anything, download the published bundle, which
carries the registry and the artifacts behind it:

```bash
uv run python scripts/download_artifacts.py --cs crypto_perps_funding
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
