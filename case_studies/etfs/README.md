# Case Study: ETF Cross-Asset Exposures

This case study applies the ML4T workflow to 100 exchange-traded funds spanning equities, fixed income, commodities, currencies, and real estate. ETFs offer a clean laboratory for cross-asset rotation: standardized pricing, deep liquidity, and broad asset-class coverage at a single rebalance cadence.

The configuration is the most cost-favorable in the book — long-only rank-and-rebalance, monthly month-end decisions on a 21-day forward-return label, with a 5--15 bps-per-leg cost model. That cadence makes it the natural setting for the broadest model-family comparison in the book, and the recurring teaching point is the gap between rank correlation (IC) and portfolio Sharpe.

> **Progressive release.** This directory currently ships the first five notebooks — the *research-design front half* of the pipeline: feasibility, labels, feature engineering, and feature evaluation. The modeling, backtest, portfolio-construction, cost, and risk stages (notebooks 06--18) are released progressively as their chapters go public.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | Multi-asset ETFs |
| Frequency | Daily data, monthly decisions |
| Universe | 100 ETFs across 9 categories |
| History | 2006--2025 |
| Primary Label | fwd_ret_21d |
| CV Folds | 8 (10Y train, 1Y val) |
| Cost Model | Material (5--15 bps per leg) |

## Pipeline (released so far)

| Stage | Notebook | Chapter | Description |
|-------|----------|---------|-------------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Universe breadth, point-in-time eligibility, horizon-cost feasibility, walk-forward demonstration |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 21-day and 5-day forward returns with walk-forward splits |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | Momentum, volatility, and cross-asset ranking features |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | ARIMA, HMM, and spectral features from walk-forward fits |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7--9 | Feature-label IC diagnostics across all engineered features |

The notebooks run in order: `01` establishes that the universe is tradeable, `02` writes the label parquet files and the walk-forward CV configuration, `03` and `04` write the feature parquet files, and `05` reads all three to evaluate feature-label information coefficients.

## Running

```bash
# From repo root — first download the free ETF data (Yahoo Finance, no API key)
uv run python data/etfs/market/download.py

# Then run the research-design front half in order
uv run python case_studies/etfs/01_feasibility_analysis.py
uv run python case_studies/etfs/02_labels.py
uv run python case_studies/etfs/03_financial_features.py
uv run python case_studies/etfs/04_model_based_features.py
uv run python case_studies/etfs/05_evaluation.py
```

Each notebook is paired with a Jupytext `.py` source of record; open the `.ipynb` for the rendered, executed version. Generated labels, features, and reports are written under `case_studies/etfs/` and are git-ignored — they are reproduced by running the notebooks.
