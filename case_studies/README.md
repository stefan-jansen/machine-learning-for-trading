# Case Studies

Nine case studies thread through Chapters 6-20, applying the same ML4T workflow to different asset classes, frequencies, and trading constraints. Each case study defines a universe, builds labels and features, trains models from linear baselines through deep learning, and evaluates strategies through backtesting, portfolio construction, cost analysis, and risk management.

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

Every case study runs the same sequence of phases, and each phase maps to a book chapter. The stage
*numbers* differ from one case study to the next, because each market gets a different set of
model-family stages, but the phase order is identical everywhere. The phase-to-chapter table lives in
**[Running Notebooks](../docs/running-notebooks.md#the-workflow)**.

Not every case study has every model type. The exact notebook set depends on the dataset
characteristics; each case study's own README lists its stages, in order, with the chapter each one
belongs to.

## Running a Complete Pipeline

Run the stages in order from the repo root. See
**[Running a Case Study End to End](../docs/running-notebooks.md#running-a-case-study-end-to-end)**
for the pattern, and the case study's own README for its exact stage list. Reduced-parameter test
runs and headless execution are covered in the same document.

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
+-- benchmark/                 # Tracked equal-weight market return series and metadata
+-- run_log/                   # Downloaded or generated result state
|   +-- registry.db            # Result and provenance source of truth
|   +-- training/{hash}/       # Specs, coefficients, boosters, checkpoints
|   +-- predictions/{hash}/    # Stored prediction arrays
|   +-- backtest/{hash}/       # Returns, trades, weights, and configs
```

The strategy-analysis notebooks read `benchmark/` as a release input. Each parquet contains an
equal-weight reference return series for one label. Most use the cross-sectional mean of
close-to-close returns; the US firm-characteristics benchmarks use the cross-sectional mean of the
label. The matching JSON states the method, coverage, validation and holdout windows, and
annualized summary statistics. Files remain label-specific because coverage can differ by label,
even when two labels use the same underlying reference series.

## Reproducibility

- `config/setup.yaml` defines the trading setup, cost model, and evaluation protocol
- `run_log/` implements the Chapter 6.7 run log: every model run is content-addressed by its config hash
- `run_log/registry.db` is the only result source of truth; legacy result JSON files are not used
- `scripts/download_artifacts.py` installs the accepted run log and stored artifacts without retraining
- `scripts/create_experiment.py` creates a writable copy for new configurations and backtests

For the schema and querying API, see **[RUN_LOG.md](RUN_LOG.md)**. For reproduction and
experimentation steps, see **[Running Notebooks](../docs/running-notebooks.md#case-study-notebooks)**.
