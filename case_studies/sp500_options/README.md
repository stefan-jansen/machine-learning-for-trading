# Case Study: S&P 500 Options (Straddles)

This case study trades options directly rather than using options as side information for an equity model. It is built around the central finding of O'Donovan and Yu (2024, *A Transaction Cost Perspective on Option Anomalies*): of 24 widely-cited single-name option-return predictors, 17 generate significant gross long-short returns, but none survive realistic transaction costs in the standard one-month delta-hedged framing. Option spreads are large enough that a strategy entering and exiting at quoted bid-ask prices pays the spread twice; for ATM single-name straddles the round-trip cost is typically a sizable fraction of the premium itself.

The pipeline adopts the **hold-to-maturity (HTM) mitigation** as its primary modeling target, following the first of three cost-mitigation strategies in O'Donovan and Yu. Each position is a short ATM straddle on an S&P 500 constituent, written on the last available session of each ISO week and held to expiry. This schedule uses Thursday when Friday is a market holiday. Daily delta hedging captures the variance risk premium while the option leg accrues to intrinsic value at expiration. There is no exit-side option trade, so the round-trip option spread becomes a one-sided entry cost. The `ret_to_expiry` label measures the strategy's per-position return and is the registry's only strategy label. Four legacy forward-return variants remain outside the strategy pipeline because their interpretation does not match the HTM engine. Equity-style bps-of-notional accounting understates option spread cost by one to two orders of magnitude.

The teaching point is methodological: equity-style bps-of-notional cost models are structurally mismatched with option premium returns. The case study supplies a worked example of switching to a premium-denominated cost framework as the cost mitigation itself, and quantifying what survives.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | S&P 500 equity options (ATM straddles) |
| Frequency | Weekly last-available-session entry, daily delta hedge during hold |
| Universe | S&P 500 constituents with listed options |
| History | 2017–2021 |
| Primary Label | `ret_to_expiry` (HTM short straddle, ~30-day DTE) |
| CV Folds | 2 (single-window, expanding) |
| Cost Model | HTM daily-MTM with full per-leg accounting (entry-side option spread + daily underlying hedge spread; no exit-leg option trade) |

## Backtest engines

All `ret_to_expiry` backtests dispatch through the **HTM cohort engine** (`_htm_backtest.py` to `_run_htm_daily_mtm`). Weekly last-available-session entry with about 30 days to expiry puts up to **5 concurrent cohorts** per underlying at any time. Each cohort carries a short straddle plus a daily-rebalanced delta hedge. Cohort capital is 1/N_ROLL, and portfolio P&L is the weighted sum of per-cohort daily MTM. The shared `ml4t-backtest` engine assumes one position per symbol with continuous reallocation and does not model overlap, paired option and hedge legs, or daily option-premium MTM.

The cost-mitigation cascade (O'Donovan & Yu 2024) is encoded in the `strategy.signal.universe_filter` spec field: `None` runs on the full S&P 500 ATM straddle surface (rung 2 in O'Donovan & Yu's framing), `'liquid'` restricts to the per-rebalance bottom-quintile half-spread subset (rung 3). The canonical sweep is pinned to `'liquid'` (`setup.yaml::backtest.sweep.universe_filter`), since the full surface does not survive round-trip costs; the `'full'` vs `'liquid'` contrast is retained in the Ch18 HTM cost cascade as a narrative comparison only, not as a rank-1 candidate.

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Setup | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Universe breadth, round-trip cost against premium, premium persistence, fold structure | Nothing |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | HTM short-straddle return + delta-hedged and raw forward variants | Five parquets in `labels/` — `ret_to_expiry` plus the 5- and 10-session forward returns and their delta-hedged counterparts — each with a `.digest.json` sidecar. Cross-validation folds come from `config/setup.yaml` |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | VRP, IV surface, skew, term structure, and Greeks features | `features/financial.parquet` |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | Walk-forward GJR-GARCH volatility + particle-filtered stochastic volatility | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7–9 | IC diagnostics on the engineered feature set | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | Ridge / Lasso / Elastic Net on each label | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | LightGBM regression and classification on each label | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `feature_importance.parquet` under `run_log/training/{hash}/` (this case study has its own artifact writer and does not emit `fold_metrics.parquet`) |
| Tabular DL | [`08_tabular_dl`](08_tabular_dl.ipynb) | Ch12 | TabM rank-1 adapter MLP on the options feature matrix | Training runs and prediction sets; checkpoints under `run_log/training/tabular_dl/` |
| Deep Learning | [`09_deep_learning`](09_deep_learning.ipynb) | Ch13 | Index notebook for sequence models | Nothing - it reads the registry |
| LSTM | [`09a_lstm`](09a_lstm.ipynb) | Ch13 | Sequential gating over daily options features | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| PatchTST | [`09b_patchtst`](09b_patchtst.ipynb) | Ch13 | Multi-scale patch attention on options dynamics | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Causal DML | [`10_causal_dml`](10_causal_dml.ipynb) | Ch15 | Treatment effect of VRP on delta-hedged returns | A row in the registry's `causal_runs` |
| Model Analysis | [`11_model_analysis`](11_model_analysis.ipynb) | — | Cross-model IC comparison and fold stability diagnostics | Nothing - it reads the registry |
| Backtest | [`12_backtest`](12_backtest.ipynb) | Ch16 | HTM dispatch with multi-cohort daily-MTM aggregation | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, and `spec.json` under `run_log/backtest/{hash}/` (the vectorized path produces no trade or fill ledger) |
| Portfolio | [`13_portfolio_management`](13_portfolio_management.ipynb) | Ch17 | Long-short straddle allocation with margin constraints | One backtest run per allocation method, same artifact layout |
| Costs | [`14_costs`](14_costs.ipynb) | Ch18 | HTM cost-sensitivity grid in % of premium across families and universes | `evaluation/htm_cost_sensitivity.parquet`, plus one registered backtest run per cost cell with `daily_returns.parquet` and `spec.json` under `run_log/backtest/{hash}/` (the grid is aggregated inline rather than through `run_backtest()`, so there are no weights) |
| Risk | [`15_risk_management`](15_risk_management.ipynb) | Ch19 | Position-level exit rules (portfolio-level overlays framed as §19.8 governance) | Nothing - the comparison stays in the notebook |
| Strategy Analysis | [`16_strategy_analysis`](16_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with paired-bootstrap holdout closure | `results/strategy_assessment.json`. The tear sheet is gated on a `trades.parquet` the vectorized HTM backtester does not emit, so it is skipped |
| Appendix | [`90_ic_diagnostic`](90_ic_diagnostic.ipynb) | — | Signal-attribution deep dive outside the main pipeline | Nothing - it reads the registry |

## Key Results

A null-edge case on the HTM primary label `ret_to_expiry`. The validation carrier is an equal-weight top-5 selection on the cost-feasible liquid universe, anchored on `linear / ridge_a10000000.0`. Two alternative allocations tie it exactly, so the simpler equal-weight baseline remains the carrier.

**Signal direction.** The carrier's daily IC is -0.0040 [-0.0343, +0.0262] over 484 validation dates (HAC p=0.794). LSTM has the highest model IC at +0.0138 [-0.0089, +0.0366], also unresolved. The prediction and strategy evidence agree that validation does not establish an edge.

**Baseline performance.** Validation Sharpe is +0.0017 [-1.0793, +1.5630] over 497 daily periods, with CAGR -0.7888 and maximum drawdown -0.9961. The distinct full-universe baseline leader has Sharpe +0.0027. Its all-family baseline cohort has 342 variants and DSR_ER -0.0094 (p=0.611). Linear PBO has only two combinations, below the ten-combination reporting threshold.

**Holdout closure.** The fixed carrier has holdout Sharpe +1.0947 [-0.5816, +3.1555]. Its holdout-minus-validation difference is +1.0931 [-1.2007, +3.2558] (p=0.327). Against the equal-weight holdout universe, the difference is -1.4527 [-3.5364, +0.6402] (p=0.166). Both comparisons remain uncertain, and the holdout never enters selection.

**Friction floor.** The HTM cost grid contains 32 rows across four model families, two universes, and cost fractions 0.203, 0.5, 0.75, and 1.0. Every row is negative. At the full quoted half-spread, net Sharpes range from -0.698 to -1.266. Premium-denominated option spreads remain the binding constraint.

The printed book records the frozen production environment used for its results. This README reports the corrected living-code registry, including the holiday-aware weekly schedule and current model cohort. Hardware and library differences can cause small numerical variation, while the no-edge conclusion should remain stable.

## Running

Run from the repository root with the project environment. The pipeline requires the materialized AlgoSeek S&P 500 options straddles and matching daily underlying bars under `ML4T_DATA_PATH`. Missing licensed data fails at the loader boundary.

Notebooks 09a and 09b require explicit CUDA. On an RTX 3090, the accepted full runs took about 12 minutes for LSTM and 67 minutes for PatchTST. Other notebooks use the stored registry and artifacts when available. Do not replace a skipped long model with a CPU run; retain the accepted artifact or document the skip.

```bash
# From repo root
uv run python case_studies/sp500_options/01_feasibility_analysis.py
uv run python case_studies/sp500_options/02_labels.py
uv run python case_studies/sp500_options/03_financial_features.py
uv run python case_studies/sp500_options/04_model_based_features.py
uv run python case_studies/sp500_options/05_evaluation.py
uv run python case_studies/sp500_options/06_linear.py
uv run python case_studies/sp500_options/07_gbm.py
uv run python case_studies/sp500_options/08_tabular_dl.py
uv run python case_studies/sp500_options/09_deep_learning.py
uv run python case_studies/sp500_options/09a_lstm.py
uv run python case_studies/sp500_options/09b_patchtst.py
uv run python case_studies/sp500_options/10_causal_dml.py
uv run python case_studies/sp500_options/11_model_analysis.py
uv run python case_studies/sp500_options/12_backtest.py
uv run python case_studies/sp500_options/13_portfolio_management.py
uv run python case_studies/sp500_options/14_costs.py
uv run python case_studies/sp500_options/15_risk_management.py
uv run python case_studies/sp500_options/16_strategy_analysis.py
uv run python case_studies/sp500_options/90_ic_diagnostic.py
```

## Run Log

Model training runs, predictions, and backtest results are tracked in a content-addressed registry under `run_log/registry.db`.
