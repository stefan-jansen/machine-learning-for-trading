# Crypto Perpetuals Funding

This case study uses Binance perpetual futures to examine an asset-class-specific return source:
the transfer between long and short positions at each 8-hour funding settlement. Nineteen
perpetuals create the book's smallest cross-section and highest non-intraday decision frequency.
The pipeline therefore emphasizes completed-bar timing, official funding cash flows, transaction
costs, and uncertainty from only two validation folds.

## Dataset Profile

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
| Backtest | [`13_backtest`](13_backtest.ipynb) | Ch16 | Replays a frozen carrier with completed-bar prices and official funding. | Nothing - it replays a frozen carrier with `register=False` |
| Portfolio | [`14_portfolio_management`](14_portfolio_management.ipynb) | Ch17 | Compares corrected point-in-time allocation methods on that carrier. | Nothing - it replays a frozen carrier with `register=False` |
| Costs | [`15_costs`](15_costs.ipynb) | Ch18 | Measures cost sensitivity and price-only versus funding-inclusive breakevens. | Nothing - it replays a frozen carrier with `register=False` |
| Risk | [`16_risk_management`](16_risk_management.ipynb) | Ch19 | Evaluates fixed and pre-validation-calibrated position-risk rules. | Nothing - it replays a frozen carrier with `register=False` |
| Synthesis | [`17_strategy_analysis`](17_strategy_analysis.ipynb) | Ch20 | Keeps current model evidence separate from frozen carrier diagnostics. | Nothing - it reads the registry |

## Running

Run notebooks from the repository root. Notebooks 07-10 require CUDA; the other notebooks use CPU.
The complete release pipeline is not yet supported because the current model registry has no
backtests or cohorts, while notebooks 13-16 preserve a frozen carrier for diagnostic replay. The
publication lineage must be chosen before the downstream producer sequence can be documented as a
reader-reproducible run.

The signed current-model sequence is:

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
```

Notebooks 13-17 are signed for their declared frozen-versus-current boundaries. They are not a
current end-to-end strategy and should not be combined into one until the release registry is fixed.
