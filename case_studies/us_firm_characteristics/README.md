# Case Study: US Firm Characteristics

This case study uses anonymized monthly firm characteristics from NASDAQ Data Link to test the canonical factor investing question: can ML improve on traditional long-short decile sorts when point-in-time accounting lags, survivorship bias, and era-dependent transaction costs are treated as binding constraints. With 57 firm-level characteristics spanning valuation, profitability, momentum, and risk across roughly 2,500 stocks (1996–2016), this is the most feature-rich fundamental dataset in the book and the natural home for latent factor models.

The case study runs at monthly cadence with a 6-month accounting lag enforced for point-in-time compliance, equal-weight long-short decile portfolios, dollar-neutral construction, and an era-dependent cost grid (pre-decimalization spreads 15–30 bps; post-2001 spreads 5–15 bps). Ten CV folds with 10-year training windows and 1-year validation provide the deepest cross-validation in the book; the calendar 2016 holdout supplies 12 monthly out-of-sample observations.

The teaching arc threads four questions that have to be answered jointly: how far the choice between a regression and a classification label moves IC on the same features, which families produce a credible standalone signal and which do not, what a 12-period holdout can and cannot resolve about decay, and whether a universe-mean cost grid understates the friction a long-short book faces when its legs cluster in small-cap, wide-spread, high-idiosyncratic-volatility names.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | US equities (fundamental characteristics, long-short) |
| Frequency | Monthly |
| Universe | ~2,500 stocks (price > $5, ADV > $1M) |
| History | 1996-2016 |
| Primary Label | fwd_ret_1m |
| CV Folds | 10 (10Y train, 1Y val) |
| Cost Model | Material (5–20 bps per leg, era-dependent) |

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Cross-sectional breadth, move scale against the assumed round trip, rank persistence, and the walk-forward folds, against `config/setup.yaml` | Nothing |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 1-month forward returns with winsorized regression and median-split classification variants | `labels/prices.parquet`, `labels/fwd_ret_1m.parquet`, `labels/fwd_ret_1m_win.parquet`, `labels/fwd_class_1m.parquet`, `config/cv_config.json` |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | 57 firm characteristics across value, quality, momentum, risk, and investment families | `features/financial.parquet`, `features/feature_doc.json` |
| Evaluation | [`04_evaluation`](04_evaluation.ipynb) | Ch7–9 | HAC-adjusted feature IC with FDR control across the characteristic panel | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`05_linear`](05_linear.ipynb) | Ch11 | Ridge, LASSO, ElasticNet, and logistic baselines on the characteristic matrix | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| GBM | [`06_gbm`](06_gbm.ipynb) | Ch12 | LightGBM testing non-linear value-quality-momentum interactions | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | [`07_tabular_dl`](07_tabular_dl.ipynb) | Ch12 | TabM rank-1 adapter MLP ensemble on the flat characteristic matrix | Training runs and prediction sets; checkpoints under `run_log/training/tabular_dl/` |
| Latent factors (index) | [`08_latent_factors`](08_latent_factors.ipynb) | Ch14 | Index of the four latent-factor notebooks below | Nothing - it reads the registry |
| IPCA | [`08a_ipca`](08a_ipca.ipynb) | Ch14 | Instrumented PCA with characteristic-conditioned loadings | Training runs and prediction sets |
| Conditional autoencoder | [`08b_conditional_autoencoder`](08b_conditional_autoencoder.ipynb) | Ch14 | Nonlinear conditional factor exposures on the characteristic panel | Training runs and prediction sets |
| SDF | [`08c_stochastic_discount_factor`](08c_stochastic_discount_factor.ipynb) | Ch14 | Neural stochastic discount factor on the same panel | Training runs and prediction sets |
| Supervised autoencoder | [`08d_supervised_autoencoder`](08d_supervised_autoencoder.ipynb) | Ch14 | Return-supervised latent factors | Training runs and prediction sets |
| Causal DML | [`09_causal_dml`](09_causal_dml.ipynb) | Ch15 | Does 12-month momentum cause future returns under FF5 confounder controls? | A row in the registry's `causal_runs` |
| Model Analysis | [`10_model_analysis`](10_model_analysis.ipynb) | n/a | Cross-family IC comparison, conformal coverage, fold-stability diagnostics | Nothing - it reads the registry |
| Backtest | [`11_backtest`](11_backtest.ipynb) | Ch16 | Long-short decile strategy simulation across the prediction-signal sweep | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, and `spec.json` under `run_log/backtest/{hash}/` (the vectorized path produces no trade or fill ledger) |
| Portfolio | [`12_portfolio_management`](12_portfolio_management.ipynb) | Ch17 | Allocator and concentration sweep on the deep cross-section | One backtest run per allocation method, same artifact layout |
| Risk | [`13_risk_management`](13_risk_management.ipynb) | Ch19 | Position-level and portfolio-level risk overlays on the monthly cadence | Nothing - the position-control loop is gated off on the vectorized path and the portfolio-control list is asserted empty, so no overlay variant is registered |
| Costs | [`14_costs`](14_costs.ipynb) | Ch18 | Era-dependent cost grid spanning pre- and post-decimalization | One backtest run per cost level, same artifact layout |
| Holdout Predictions | [`15_holdout_predictions`](15_holdout_predictions.ipynb) | Ch20 | Refits the selected configuration on the history before the holdout window | One training run and one prediction set, both keyed to the derived holdout fold |
| Holdout Backtest | [`16_holdout_backtest`](16_holdout_backtest.ipynb) | Ch20 | Trades the holdout predictions with the sizing and cost assumption the case study settled on | One backtest run at `stage='holdout'` |
| Strategy Analysis | [`17_strategy_analysis`](17_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with uncertainty-aware metrics | Nothing - the tear sheet needs a `trades.parquet` the vectorized backtester never emits, so the stage takes its no-trades branch |

## Running

```bash
# From repo root
uv run python case_studies/us_firm_characteristics/01_feasibility_analysis.py
uv run python case_studies/us_firm_characteristics/02_labels.py
uv run python case_studies/us_firm_characteristics/03_financial_features.py
uv run python case_studies/us_firm_characteristics/04_evaluation.py
uv run python case_studies/us_firm_characteristics/05_linear.py
uv run python case_studies/us_firm_characteristics/06_gbm.py
uv run python case_studies/us_firm_characteristics/07_tabular_dl.py
uv run python case_studies/us_firm_characteristics/08a_ipca.py
uv run python case_studies/us_firm_characteristics/08b_conditional_autoencoder.py
uv run python case_studies/us_firm_characteristics/08c_stochastic_discount_factor.py
uv run python case_studies/us_firm_characteristics/08d_supervised_autoencoder.py
uv run python case_studies/us_firm_characteristics/08_latent_factors.py   # summarizes 08a-08d
uv run python case_studies/us_firm_characteristics/09_causal_dml.py
uv run python case_studies/us_firm_characteristics/10_model_analysis.py
uv run python case_studies/us_firm_characteristics/11_backtest.py
uv run python case_studies/us_firm_characteristics/12_portfolio_management.py
uv run python case_studies/us_firm_characteristics/13_risk_management.py
uv run python case_studies/us_firm_characteristics/14_costs.py
uv run python case_studies/us_firm_characteristics/15_holdout_predictions.py
uv run python case_studies/us_firm_characteristics/16_holdout_backtest.py
uv run python case_studies/us_firm_characteristics/17_strategy_analysis.py
```

## Results

This README describes how the case study is built, not what it found. Results are
not restated here: the registry is rebuilt whenever the case study is re-derived,
and a number copied into prose stays correct only until the next rebuild.

[`17_strategy_analysis`](17_strategy_analysis.ipynb) reads the registry back and reports the selected
configuration with its interval evidence. That notebook, and the registry it reads,
are where a result comes from.

To read the results without training anything, download the published bundle, which
carries the registry and the artifacts behind it:

```bash
uv run python scripts/download_artifacts.py --cs us_firm_characteristics
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
