# Case Study: FX Spot Pairs

This case study uses daily OHLCV data from OANDA for 20 G10 currency pairs to test whether momentum and carry signals produce tradeable alpha in the most liquid market on earth. FX presents a structurally challenging prediction problem: tight spreads enable low-cost execution, but the cross-section is small and correlated. The daily return-correlation matrix has a participation ratio of 5.27 effective bets. USD and JPY each appear in 7 of the 20 pairs.

The pipeline is a study in hypothesis revision, from short-horizon momentum to multi-horizon carry and a full strategy assessment. It compares signal quality, selection-adjusted Sharpe, and holdout performance against the equal-weight benchmark. The teaching point is what disciplined assessment looks like when a signal has to be judged on interval evidence rather than on a point estimate.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | FX spot pairs (G10 currencies) |
| Frequency | Daily |
| Universe | 20 major and cross pairs |
| History | 2005-2025 |
| Primary Label | fwd_ret_1d |
| CV Folds | 8 (5Y train, 1Y val) |
| Cost Model | Material (1--8 bps spread) |

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Universe breadth at the daily decision, effective independent bets, assumed round-trip cost per pair, move-to-cost scale by horizon, return persistence, and the declared walk-forward folds | none |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 1-day, 5-day, and 21-day forward returns | `labels/fwd_ret_1d.parquet`, `labels/fwd_ret_5d.parquet`, `labels/fwd_ret_21d.parquet` |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | Momentum, carry, volatility, and mean-reversion features | `features/financial.parquet` |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | Walk-forward ARIMA, HMM, and spectral features | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7--9 | Feature-label IC diagnostics across 20 pairs | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | Ridge, LASSO, ElasticNet on momentum and carry signals | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | LightGBM testing non-linear USD factor and momentum interactions | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | [`08_tabular_dl`](08_tabular_dl.ipynb) | Ch12 | TabM attention-style ensembling on the FX feature matrix | Training runs and prediction sets; checkpoints under `run_log/training/tabular_dl/` |
| TCN | [`09_dl_tcn`](09_dl_tcn.ipynb) | Ch13 | Dilated causal convolutions for daily FX dynamics | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| NLinear | [`10_dl_nlinear`](10_dl_nlinear.ipynb) | Ch13 | Tests whether FX dynamics are approximately linear | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| LSTM | [`10a_dl_lstm`](10a_dl_lstm.ipynb) | Ch13 | The recurrent member of the three architectures the `deep_learning` menu declares; what it carries forward is not fixed the way NLinear's subtraction is | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Causal DML | [`11_causal_dml`](11_causal_dml.ipynb) | Ch15 | Does FX momentum cause future returns or reflect overshooting? | A row in the registry's `causal_runs` |
| Model Analysis | [`12_model_analysis`](12_model_analysis.ipynb) | -- | Cross-model IC comparison and fold stability diagnostics | Nothing - it reads the registry |
| Backtest | [`13_backtest`](13_backtest.ipynb) | Ch16 | Long-short daily FX strategy simulation | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`14_portfolio_management`](14_portfolio_management.ipynb) | Ch17 | Allocation methods for the small FX cross-section | One backtest run per allocation method, same artifact layout |
| Risk | [`15_risk_management`](15_risk_management.ipynb) | Ch19 | Position-level controls compared with the unoverlaid carrier | One backtest run per overlay variant, same artifact layout |
| Costs | [`16_costs`](16_costs.ipynb) | Ch18 | Spread impact on the selected carrier, controls included | One backtest run per cost level, same artifact layout |
| Holdout Predictions | [`17_holdout_predictions`](17_holdout_predictions.ipynb) | Ch16-20 | Refits the validation-selected configuration on the holdout interval | One holdout prediction set at the selected checkpoint |
| Holdout Backtest | [`18_holdout_backtest`](18_holdout_backtest.ipynb) | Ch16-20 | Replays the selected strategy against those predictions | One holdout backtest run |
| Strategy Analysis | [`19_strategy_analysis`](19_strategy_analysis.ipynb) | Ch20 | Reads the validation and holdout results back and reports them with interval evidence | Registry rows rather than files: `cohort_metrics` for the candidate cohorts and `backtest_paired_metrics` for the bootstrapped comparisons |

## Running

Run every command from the repository root with the project environment. Set `ML4T_DATA_PATH` to
a data root containing the consolidated OANDA files under `fx/`; see
[`data/fx/README.md`](../../data/fx/README.md). The project requires Python 3.14 and installs with
`uv sync`. The GBM notebook uses the configured GPU when available and falls back to CPU. The
TabM, TCN, and NLinear notebooks use their deterministic CPU defaults.

The sequence below reads the registry and its cached artifacts. Notebooks 13-16 default
to `RUN_SWEEP=False`, so this path reports the registered result without rerunning the sweep.

```bash
# From repo root
uv run python case_studies/fx_pairs/01_feasibility_analysis.py
uv run python case_studies/fx_pairs/02_labels.py
uv run python case_studies/fx_pairs/03_financial_features.py
uv run python case_studies/fx_pairs/04_model_based_features.py
uv run python case_studies/fx_pairs/05_evaluation.py
uv run python case_studies/fx_pairs/06_linear.py
uv run python case_studies/fx_pairs/07_gbm.py
uv run python case_studies/fx_pairs/08_tabular_dl.py
uv run python case_studies/fx_pairs/09_dl_tcn.py
uv run python case_studies/fx_pairs/10_dl_nlinear.py
uv run python case_studies/fx_pairs/10a_dl_lstm.py
uv run python case_studies/fx_pairs/11_causal_dml.py
uv run python case_studies/fx_pairs/12_model_analysis.py
uv run python case_studies/fx_pairs/13_backtest.py
uv run python case_studies/fx_pairs/14_portfolio_management.py
uv run python case_studies/fx_pairs/15_risk_management.py
uv run python case_studies/fx_pairs/16_costs.py
uv run python case_studies/fx_pairs/17_holdout_predictions.py
uv run python case_studies/fx_pairs/18_holdout_backtest.py
uv run python case_studies/fx_pairs/19_strategy_analysis.py
```

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
uv run python scripts/download_artifacts.py --cs fx_pairs
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
