# Case Studies

Nine case studies thread through Chapters 6--20, applying the same ML4T workflow to different asset classes, frequencies, and trading constraints. Each case study defines a universe, builds labels and features, trains models from linear baselines through deep learning, and evaluates strategies through backtesting, portfolio construction, cost analysis, and risk management.

## Overview

| # | Case Study | Asset Class | Frequency | Universe | Primary Label |
|---|------------|-------------|-----------|----------|---------------|
| 1 | [ETFs](etfs/) | Multi-asset ETFs | Daily | 100 ETFs | fwd_ret_21d |
| 2 | [Crypto Perps Funding](crypto_perps_funding/) | Crypto perpetual futures | 8-hourly | 19 pairs | fwd_ret_8h |
| 3 | [NASDAQ-100 Microstructure](nasdaq100_microstructure/) | US equities (intraday) | 15-min | 114 stocks | fwd_ret_15m |
| 4 | [S&P 500 Equity + Options](sp500_equity_option_analytics/) | S&P 500 equities | Daily | 634 stocks | fwd_ret_5d |
| 5 | [US Firm Characteristics](us_firm_characteristics/) | US equities (fundamental) | Monthly | ~2,500 stocks | fwd_ret_1m |
| 6 | [FX Pairs](fx_pairs/) | G10 currency pairs | Daily | 20 pairs | fwd_ret_1d |
| 7 | [CME Futures](cme_futures/) | Multi-sector futures | Daily | 30 products | fwd_ret_5d |
| 8 | [S&P 500 Options](sp500_options/) | S&P 500 equity options | Daily | S&P 500 straddles | fwd_ret_dh_10d |
| 9 | [US Equities Panel](us_equities_panel/) | Broad US equities | Daily | ~3,200 stocks | fwd_ret_1d |

## Pipeline Stages

Each case study follows the same chapter progression. Notebooks are numbered sequentially, with each number mapping to a chapter:

| Stage | Chapter | Typical Notebook | What It Produces |
|-------|---------|------------------|------------------|
| Feasibility | Ch6 | `01_feasibility_analysis` | Universe and cost feasibility evidence for the canonical `config/setup.yaml` |
| Labels | Ch7 | `02_labels` | Forward returns, walk-forward CV splits |
| Features | Ch8 | `03_financial_features` | Momentum, volatility, carry, and domain-specific features |
| Temporal | Ch9 | `04_model_based_features` | ARIMA, HMM, spectral features from walk-forward fits |
| Evaluation | Ch7--9 | `05_evaluation` | Feature-label IC diagnostics |
| Linear | Ch11 | `06_linear` | Ridge, LASSO, ElasticNet baseline predictions |
| GBM | Ch12 | `07_gbm` | LightGBM predictions with Optuna |
| Tabular DL | Ch12 | `08_tabular_dl` | TabM / neural tabular predictions |
| Deep Learning | Ch13 | `09-10_dl_*` | LSTM, TCN, TSMixer, PatchTST, N-BEATS predictions |
| Latent Factors | Ch14 | `*_latent_factors`, `*_pca`, `*_ipca`, `*_sdf` | PCA, IPCA, CAE, SAE, SDF factor models |
| Causal | Ch15 | `*_causal_dml` | Double ML treatment effect estimates |
| Backtest | Ch16 | `*_backtest` | Strategy simulation results |
| Analysis | Ch16 | `*_backtest_analysis` | Performance attribution and reporting |
| Portfolio | Ch17 | `*_portfolio_management` | Allocation methods and portfolio construction |
| Costs | Ch18 | `*_costs` | Transaction cost impact analysis |
| Risk | Ch19 | `*_risk_management` | Drawdown controls, position limits, risk budgets |
| Synthesis | Ch20 | `*_synthesis` | End-to-end strategy assessment |

Not every case study has every model type. The exact notebook set depends on the dataset characteristics. See each case study's README for the complete pipeline table.

## Running a Complete Pipeline

```bash
# From repo root -- always

# Example: ETF pipeline
uv run python case_studies/etfs/01_feasibility_analysis.py
uv run python case_studies/etfs/02_labels.py
uv run python case_studies/etfs/03_financial_features.py
uv run python case_studies/etfs/04_model_based_features.py
uv run python case_studies/etfs/05_evaluation.py
uv run python case_studies/etfs/06_linear.py
uv run python case_studies/etfs/07_gbm.py
# ... continue through synthesis

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "etfs"

# Headless (no display)
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python case_studies/etfs/01_feasibility_analysis.py
```

## Directory Layout

Each case study follows this structure:

```
case_studies/{id}/
+-- README.md                  # Dataset profile, pipeline table
+-- config/
|   +-- setup.yaml             # SSOT: universe, costs, CV, labels
+-- 01_feasibility_analysis.py / .ipynb       # Numbered notebook sequence
+-- 02_labels.py / .ipynb
+-- ...
+-- labels/                    # Generated learning targets (gitignored)
+-- features/                  # Generated financial and temporal features
+-- evaluation/                # Generated feature diagnostics
+-- run_log/                   # Downloaded or generated result state
|   +-- registry.db            # Result and provenance source of truth
|   +-- training/{hash}/       # Specs, coefficients, boosters, checkpoints
|   +-- predictions/{hash}/    # Stored prediction arrays
|   +-- backtest/{hash}/       # Returns, trades, weights, and configs
```

## Reproducibility

- `config/setup.yaml` defines the trading setup, cost model, and evaluation protocol
- `run_log/` implements the Chapter 6.7 run log: every model run is content-addressed by its config hash
- `run_log/registry.db` is the only result source of truth; legacy result JSON files are not used
- `scripts/download_artifacts.py` installs the accepted run log and stored artifacts without retraining
- `scripts/create_experiment.py` creates a writable copy for new configurations and backtests

For the schema and querying API, see **[RUN_LOG.md](RUN_LOG.md)**. For reproduction and
experimentation steps, see **[Running Notebooks](../docs/running-notebooks.md#case-study-notebooks)**.
