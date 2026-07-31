# Case Study: FX Spot Pairs

This case study uses daily OHLCV data from OANDA for 20 G10 currency pairs to test whether momentum and carry signals produce tradeable alpha in the most liquid market on earth. FX presents a structurally challenging prediction problem: tight spreads enable low-cost execution, but the cross-section is small and correlated. The daily return-correlation matrix has a participation ratio of 5.27 effective bets. USD and JPY each appear in 7 of the 20 pairs.

The pipeline is a study in hypothesis revision, from short-horizon momentum to multi-horizon carry and a full strategy assessment. It compares signal quality, selection-adjusted Sharpe, and holdout performance against the equal-weight benchmark. The teaching point is what disciplined assessment looks like when the underlying signal is statistically thin.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | FX spot pairs (G10 currencies) |
| Frequency | Daily |
| Universe | 20 major and cross pairs |
| History | 2005--2025 |
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
| Causal DML | [`11_causal_dml`](11_causal_dml.ipynb) | Ch15 | Does FX momentum cause future returns or reflect overshooting? | A row in the registry's `causal_runs` |
| Model Analysis | [`12_model_analysis`](12_model_analysis.ipynb) | -- | Cross-model IC comparison and fold stability diagnostics | Nothing - it reads the registry |
| Backtest | [`13_backtest`](13_backtest.ipynb) | Ch16 | Long-short daily FX strategy simulation | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`14_portfolio_management`](14_portfolio_management.ipynb) | Ch17 | Allocation methods for the small FX cross-section | One backtest run per allocation method, same artifact layout |
| Costs | [`15_costs`](15_costs.ipynb) | Ch18 | Spread impact on the selected 21-day carrier | One backtest run per cost level, same artifact layout |
| Risk | [`16_risk_management`](16_risk_management.ipynb) | Ch19 | Position-level controls compared with the unoverlaid carrier | One backtest run per overlay variant, same artifact layout |
| Strategy Analysis | [`17_strategy_analysis`](17_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with IC, Sharpe, and cost analysis | `20_strategy_synthesis/output/fx_pairs/fx_pairs_tearsheet.html` and `strategy_assessment.json` (a `tearsheet_predictions.parquet` is staged alongside them and deleted again) |

## Key Results

**Signal quality.** On the 1-day primary label, the four trained families all post HAC 95% CIs that straddle zero: deep_learning (nlinear) IC +0.011 [-0.005, +0.026], tabular_dl (tabm_s) +0.007 [-0.007, +0.020], linear (ridge α=10⁴) +0.005 [-0.012, +0.022], gbm (leaves_63_mae) +0.003 [-0.013, +0.018]. The 5d and 21d panels are just as zero-straddling; the largest point estimate at any horizon is a shrunk linear model (LASSO/ElasticNet) at 21d (+0.045, t≈1.5), and the credibility-at-any-horizon flag stays false.

**Strategy performance with CIs.** The current-design rank-1 lineage by validation Sharpe is the
linear Lasso configuration `linear/lasso_f0.35` on the 21-day variant label `fwd_ret_21d`, not the
`setup.yaml` primary `fwd_ret_1d`. The canonical validation backtest is `d43230f42502`: a
long-short equal-weight top-5 signal with MVO Ledoit-Wolf top-5 allocation and no risk overlay,
rebalanced daily at the NY 5pm close. MVO raises the exact equal-weight sibling's Sharpe from
-0.085 to +0.047. Validation Sharpe is +0.047 [-0.660, +0.760] over 2,064 daily periods, and the
rank-1 prediction set's daily-pooled IC is +0.045 [-0.013, +0.103] (HAC t=1.51). Both intervals
cross zero, so the registry demonstrates no resolved validation edge.

**Holdout closure.** The sealed 2024-2025 prediction set `fc3675f8457a` resolves to backtest
`e50554f931bb` on the same Lasso/MVO carrier. Holdout Sharpe is -0.157 [-1.374, +1.034] over 498
daily periods, with daily IC +0.040 [-0.090, +0.170]. The holdout-minus-validation Sharpe
difference is -0.204 [-1.403, +1.113], and the holdout-minus-equal-weight difference is -0.975
[-2.245, +0.224]. These wide, zero-straddling intervals leave the result economically
indistinguishable from zero. The accepted update changes the point estimate from +0.194 to -0.157
without changing the conclusion: this price-and-volume feature panel demonstrates no
cross-sectional FX edge in the 20-pair universe.

**Friction floor.** The 11-point cost sweep covers 0/1/2/3/5/7/10/15/20/30/50 bps per traded leg
on the selected Lasso/MVO carrier. Gross Sharpe is +0.464 at zero cost, +0.168 at 5 bps, +0.047 at
7 bps, and -0.128 at 10 bps, implying a breakeven near 7.8 bps per leg. At realistic FX spreads,
the daily-rebalanced carrier remains cost-sensitive and the friction floor is binding.

**Risk controls.** All 14 declared position controls trail the unoverlaid Lasso/MVO carrier. The
best overlay reaches Sharpe +0.027 versus +0.047 without an overlay, so the accepted strategy uses
no risk overlay. Portfolio-level controls remain library examples rather than members of this
case-study sweep.

## Verify the Accepted Analysis

Run every command from the repository root with the project environment. Set `ML4T_DATA_PATH` to
a data root containing the consolidated OANDA files under `fx/`; see
[`data/fx/README.md`](../../data/fx/README.md). The project requires Python 3.14 and installs with
`uv sync`. The GBM notebook uses the configured GPU when available and falls back to CPU. The
TabM, TCN, and NLinear notebooks use their deterministic CPU defaults.

The sequence below reads the accepted registry and its cached artifacts. Notebooks 13-16 default
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
uv run python case_studies/fx_pairs/11_causal_dml.py
uv run python case_studies/fx_pairs/12_model_analysis.py
uv run python case_studies/fx_pairs/13_backtest.py
uv run python case_studies/fx_pairs/14_portfolio_management.py
uv run python case_studies/fx_pairs/15_costs.py
uv run python case_studies/fx_pairs/16_risk_management.py
uv run python case_studies/fx_pairs/17_strategy_analysis.py
```

## Registry Status

The accepted registry contains exactly the configured current-design surface: 292 equal-weight
baselines, 360 allocation tests, 33 cost points, 42 risk controls, and one sealed holdout. Its 728
backtest hashes are portable and reconcile one-to-one with physical artifacts. The v3.0 book-print
registry and its referenced artifacts remain preserved under `run_log/registry_versions/v3.0-book/`.
Environment metadata columns remain follow-up schema debt and do not affect the accepted result.

## Run Log

Model training runs, predictions, and backtest results use `run_log/registry.db` as their only
source of truth. Migration and cleanup work belongs on a checksummed candidate copy.
