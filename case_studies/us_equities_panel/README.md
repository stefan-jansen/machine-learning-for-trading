# Case Study: US Equities Panel

This case study is the broadest cross-sectional equity workflow in the book. It uses daily OHLCV data from NASDAQ Data Link for ~3,200 US stocks spanning 1990 through 2018-Q1 to test whether weak per-stock signals translate into a tradable strategy when scaled across thousands of names. The Fundamental Law of Active Management is the operating frame: the per-stock edge is small, but breadth across the cross-section is supposed to compensate. The role of this case study is to hold that claim against measured signal quality, paired-bootstrap confidence intervals, and an explicit holdout window.

The pipeline is unusually long because the universe is unusually large. Sixteen walk-forward folds (10y train, 1y validation), the most folds of any case study, are paired with multi-horizon labels and a feature panel that mixes momentum, mean-reversion, volatility, liquidity, value proxies, and walk-forward temporal models. The strategy is a daily long-short top-K cross-sectional ranker with dollar-neutral construction and material era-dependent costs (15-30 bps pre-decimalization, 5-15 bps after). The question the strategy-analysis notebook answers is whether the gross signal that survives this much testing also survives selection-adjusted resampling and the 2016-2018 holdout.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | Broad US equities (NYSE/NASDAQ/AMEX) |
| Frequency | Daily |
| Universe | ~3,200 stocks (price > $5, ADV > $1M, point-in-time) |
| History | 1990 -- 2018-Q1 |
| Primary Label | fwd_ret_1d |
| CV Folds | 16 (10Y train, 1Y val) |
| Cost Model | Material (5-30 bps per leg, era-dependent + borrow) |

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Universe breadth per decision date, cost regime, move-to-cost scale, walk-forward folds | Nothing - the evidence stays in the notebook |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 1-day, 5-day, and 21-day forward returns | `labels/fwd_ret_1d.parquet`, `labels/fwd_ret_5d.parquet`, `labels/fwd_ret_21d.parquet`, each with a `.digest.json` sidecar |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | 62 cross-sectional factors: momentum, volatility, liquidity, value | `features/financial.parquet` |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | Walk-forward Wasserstein regime distance, FFD, GARCH features | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7--9 | Feature-label IC diagnostics across the full panel | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | Ridge, LASSO, ElasticNet on the full feature matrix | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | LightGBM grid across leaf profiles and loss functions | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | [`08_tabular_dl`](08_tabular_dl.ipynb) | Ch12 | TabM attention-style ensembling on the cross-section | Training runs and prediction sets; checkpoints under `run_log/training/tabular_dl/` |
| NLinear | [`09_dl_nlinear`](09_dl_nlinear.ipynb) | Ch13 | Minimal temporal baseline with last-value normalization | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| LSTM | [`10_dl_lstm`](10_dl_lstm.ipynb) | Ch13 | Sequential memory across daily return windows | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| TSMixer | [`11_dl_tsmixer`](11_dl_tsmixer.ipynb) | Ch13 | Time-mixing and feature-mixing across the 60-day lookback | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Weekly DL | [`12_dl_weekly`](12_dl_weekly.ipynb) | Ch13 | Weekly-cadence LSTM/NLinear comparison | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Latent Factors | [`13_latent_factors`](13_latent_factors.ipynb) | Ch14 | Index notebook for PCA + IPCA on the broad equity panel | Nothing - it reads the registry |
| PCA | [`13a_pca`](13a_pca.ipynb) | Ch14 | Static factor extraction from the return covariance | Training runs and prediction sets |
| IPCA | [`13b_ipca`](13b_ipca.ipynb) | Ch14 | Instrumented PCA with characteristic-conditioned loadings | Training runs and prediction sets |
| Causal DML | [`14_causal_dml`](14_causal_dml.ipynb) | Ch15 | Causal effect of 12-1 momentum on daily returns | A row in the registry's `causal_runs` |
| Model Analysis | [`15_model_analysis`](15_model_analysis.ipynb) | -- | Cross-model IC comparison and fold stability diagnostics | Nothing - it reads the registry |
| Backtest | [`16_backtest`](16_backtest.ipynb) | Ch16 | Daily long-short top-K strategy simulation | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`17_portfolio_management`](17_portfolio_management.ipynb) | Ch17 | Allocation sweep on the highest-IC GBM signal | One backtest run per allocation method, same artifact layout |
| Costs | [`18_costs`](18_costs.ipynb) | Ch18 | Cost-grid sweep on the top allocation combinations | One backtest run per cost level, same artifact layout |
| Risk | [`19_risk_management`](19_risk_management.ipynb) | Ch19 | Position-level and portfolio-level risk overlays | One backtest run per overlay variant, same artifact layout |
| Strategy Analysis | [`20_strategy_analysis`](20_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment: signal, lineage, holdout, attribution | `results/strategy_assessment.json`, `20_strategy_synthesis/output/us_equities_panel/us_equities_panel_tearsheet.html` |

## Key Results

**Signal direction.** GBM `leaves_31_huber` on the 5-day variant horizon achieves the highest cross-stage validation Sharpe and a strong daily-pooled IC on the panel's fwd_ret_1d grid. Pooled IC is 0.0357 with the HAC-adjusted 95% CI at [0.0293, 0.0421] over 4,018 daily cross-sections (t_HAC = 10.96), well clear of zero. Per-family rank-1 IC is monotone in horizon for GBM (1d 0.032 → 5d 0.043 → 21d 0.058) and linear (1d 0.016 → 5d 0.022 → 21d 0.029), with each CI excluding zero. Tree-based and linear families produce signals with low pairwise correlation, so an ensemble across families would not be fighting a single shared signal.

**Strategy-stage performance with CIs.** Validation Sharpe for this lineage's risk_overlay carrier (score_weighted top_k=20 + `time_exit_40`) is 2.028 with a paired-bootstrap 95% CI of [1.464, 2.549] (PSR p ≈ 2e-15, classification `excludes_zero_strong`). The strategy posts a higher Sharpe than the equal-weight US-equities universe over the same window by 1.11 [0.48, 1.76] (p ≈ 0, `excludes_zero_strong`). A FF5+MOM HAC regression credits the validation edge as alpha-driven: annualized alpha ≈ 0.76 with t_HAC ≈ 7.7, residual Sharpe ≈ 2.04, R² ≈ 0.01. Cohort-level selection-bias metrics from `cohort_metrics` (family cohort `risk_overlay/fwd_ret_5d/gbm`, K_variants = 20 position-level overlays, K_eff_MP ≈ 2.0, K_eff_ER ≈ 2.5) record DSR_ER 0.106 with p ≈ 0 (and DSR_MP 0.112, ER and MP within 0.006), and PBO 0.0 across 12,870 CSCV combinations × 16 folds. On the broader label cohort (`label/fwd_ret_5d`, K_variants = 314 across all families and stages) the same leader records DSR_ER 0.065 at p ≈ 7e-13 with K_eff_ER ≈ 12.2 — the leader's edge survives both the small overlay-cohort adjustment and the cross-stage cross-family adjustment.

**Holdout closure.** The 2016-Q1 to 2018-Q1 holdout puts this lineage at Sharpe −0.492. The index-paired diff against validation reads −2.520 [−3.804, −1.117] with p ≈ 5e-4. The CI excludes zero on the negative side, so the deterioration is statistically resolved. The reference equal-weight universe over the same holdout window posts Sharpe 1.71 (well above its validation reading of 0.92, reflecting the cap-weighted bull market of that period); against that elevated reference, strategy minus benchmark over the holdout reads −2.363 [−4.591, −0.213] (p ≈ 0.03, `excludes_zero_strong` on the negative side). The overall reading is that the validation edge does not carry across the holdout regime under the chosen rebalance cadence.

**Friction floor.** The cost-sensitivity sweep on this lineage produces a moderately steep envelope. Within the cross-stage rank-1 prediction lineage, zero-cost gross Sharpe is 2.503 (the gross-return ceiling under the 5-day-label / daily-marked strategy), and Sharpe at the 10 bps post-decimalization midpoint is 2.117. The edge-to-cost ratio comfortably clears the 1.2× kill-condition floor (`evidence_passes`). The universal Ch20 gates resolve: validation Sharpe lower bound (1.46) is above zero, and the holdout strategy-vs-EW CI excludes zero negatively. The steepness of the cost curve and the daily rebalance cadence place the strategy in a regime where execution quality is the binding operational constraint.

## Running

```bash
# From repo root
uv run python case_studies/us_equities_panel/01_feasibility_analysis.py
uv run python case_studies/us_equities_panel/02_labels.py
uv run python case_studies/us_equities_panel/03_financial_features.py
uv run python case_studies/us_equities_panel/04_model_based_features.py
uv run python case_studies/us_equities_panel/05_evaluation.py
uv run python case_studies/us_equities_panel/06_linear.py
uv run python case_studies/us_equities_panel/07_gbm.py
uv run python case_studies/us_equities_panel/08_tabular_dl.py
uv run python case_studies/us_equities_panel/09_dl_nlinear.py
uv run python case_studies/us_equities_panel/10_dl_lstm.py
uv run python case_studies/us_equities_panel/11_dl_tsmixer.py
uv run python case_studies/us_equities_panel/12_dl_weekly.py
uv run python case_studies/us_equities_panel/13_latent_factors.py
uv run python case_studies/us_equities_panel/13a_pca.py
uv run python case_studies/us_equities_panel/13b_ipca.py
uv run python case_studies/us_equities_panel/14_causal_dml.py
uv run python case_studies/us_equities_panel/15_model_analysis.py
uv run python case_studies/us_equities_panel/16_backtest.py
uv run python case_studies/us_equities_panel/17_portfolio_management.py
uv run python case_studies/us_equities_panel/18_costs.py
uv run python case_studies/us_equities_panel/19_risk_management.py
uv run python case_studies/us_equities_panel/20_strategy_analysis.py
```

The strategy-analysis notebook in `20_strategy_analysis.py` writes a full diagnostic tear sheet (`template="full"`) to the case study's gitignored output directory; readers regenerate it locally.

## Run Log

Model training runs, predictions, and backtest results are tracked in a content-addressed registry under `run_log/registry.db`.
