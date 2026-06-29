# Case Studies

Nine case studies thread through Chapters 6–20, applying the same ML4T workflow to different asset classes,
frequencies, and trading constraints. Each one defines a universe, builds labels and features, trains models from
linear baselines through deep learning, and evaluates strategies through backtesting, portfolio construction, cost
analysis, and risk management.

> **What's in this release.** This directory currently ships the shared `utils/` package the chapter notebooks build
> on, plus each case study's canonical `config/setup.yaml` (universe, costs, walk-forward CV, labels). The per-case-study
> notebook pipelines, generated data, and cross-study results are released in book order over the coming weeks — watch
> or star the repo to follow along. Cross-case-study results are synthesized in Chapter 20.

## Overview

| # | Case Study | Asset Class | Frequency | Universe | Primary Label | What It Explores |
|---|------------|-------------|-----------|----------|---------------|------------------|
| 1 | ETFs | Multi-asset ETFs | Daily | 100 ETFs | fwd_ret_21d | Cross-asset momentum and mean-reversion |
| 2 | Crypto Perps Funding | Crypto perpetual futures | 8-hourly | 19 pairs | fwd_ret_8h | Funding-rate arbitrage on perpetuals |
| 3 | NASDAQ-100 Microstructure | US equities (intraday) | 15-min | 114 stocks | fwd_ret_15m | Intraday microstructure signals from order flow and the LOB |
| 4 | S&P 500 Equity + Options | S&P 500 equities | Daily | 634 stocks | fwd_ret_5d | Equity selection enhanced with implied-volatility features |
| 5 | US Firm Characteristics | US equities (fundamental) | Monthly | ~2,500 stocks | fwd_ret_1m | Firm-level characteristics panel (size, value, momentum, quality) |
| 6 | FX Pairs | G10 currency pairs | Daily | 20 pairs | fwd_ret_1d | Carry and momentum across major currency pairs |
| 7 | CME Futures | Multi-sector futures | Daily | 30 products | fwd_ret_5d | Term-structure and roll-yield signals |
| 8 | S&P 500 Options | S&P 500 equity options | Daily | S&P 500 straddles | fwd_ret_dh_10d | Options-only strategies (straddles, delta-hedged positions) |
| 9 | US Equities Panel | Broad US equities | Daily | ~3,200 stocks | fwd_ret_1d | Broad cross-section with classic factor exposures |

## Pipeline Stages

Each case study follows the same chapter progression. Notebooks are numbered sequentially, with each number mapping to
a chapter:

| Stage | Chapter | Typical Notebook | What It Produces |
|-------|---------|------------------|------------------|
| Feasibility | Ch6 | `01_feasibility_analysis` | Universe and cost feasibility evidence for the canonical `config/setup.yaml` |
| Labels | Ch7 | `02_labels` | Forward returns, walk-forward CV splits |
| Features | Ch8 | `03_financial_features` | Momentum, volatility, carry, and domain-specific features |
| Temporal | Ch9 | `04_model_based_features` | ARIMA, HMM, spectral features from walk-forward fits |
| Evaluation | Ch7–9 | `05_evaluation` | Feature-label IC diagnostics |
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

Not every case study has every model type — the exact notebook set depends on the dataset. Each case study's own README
(shipped with its pipeline) documents its complete table.

## Directory Layout

Once a case study's pipeline is released, it follows this structure:

```
case_studies/{id}/
├── README.md                 # Dataset profile, pipeline table
├── config/
│   └── setup.yaml            # SSOT: universe, costs, CV, labels   ← shipped now
├── 01_feasibility_analysis.py / .ipynb       # Numbered notebook sequence
├── 02_labels.py / .ipynb
├── ...
├── data/                     # Labels and features (gitignored, reproducible)
│   ├── labels/
│   └── features/
└── run_log/                  # Model registry + runs (gitignored)
    └── registry.db           # Content-addressed per-run artifacts
```

## Reproducibility

- `config/setup.yaml` defines the trading setup, cost model, and evaluation protocol — the single source of truth for
  each case study.
- The run log (Chapter 6.7) records every model run, content-addressed by its config hash, in `run_log/registry.db`.
- Generated artifacts (`data/`, `run_log/`, `strategy/`) are gitignored but reproducible by running the notebook
  sequence from the repository root.
