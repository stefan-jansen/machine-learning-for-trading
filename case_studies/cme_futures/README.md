# Case Study: CME Futures

This case study uses daily Databento data on 30 CME futures products spanning seven sectors (equity indices, treasuries, energy, metals, currencies, agriculture, and livestock) to test whether carry and term-structure signals produce tradeable alpha at a weekly cadence. Futures have a return decomposition that splits into spot and roll components, natural sector groupings that constrain diversification, and inherent leverage that magnifies both signal and friction.

The pipeline runs a long-short carry-ranked strategy with weekly Friday-close decisions and Monday-open execution, trades 30 front-month continuous contracts built with ratio back-adjustment, and prices in commission, bid-ask spread, and roll slippage. The teaching point is that a modest per-product IC — the latent-factor SDF carries the credible cross-sectional signal at +0.042 (HAC 95% CI [+0.017, +0.068], t_HAC = 3.25) — translates into outsized portfolio P&L through magnitude on the top of the cross-section: the cross-stage rank-1 lineage (GBM + `score_weighted` top-k=5 allocator + 3.3% trailing-stop) posts a validation Sharpe of **1.264** [+0.345, +2.087] and a holdout Sharpe of **1.142** [−0.186, +2.342].

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | CME futures (30 products, 7 sectors) |
| Frequency | Daily data, weekly decisions |
| Universe | 30 front-month continuous contracts |
| History | 2011-2025 |
| Primary Label | fwd_ret_5d |
| CV Folds | 5 (8Y train, 1Y val) |
| Cost Model | Material (commission + spread + roll slippage) |

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Universe breadth per decision date, round-trip spread per product, move-to-spread scale, carry persistence, and the declared walk-forward folds | none |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 5-day and 21-day forward returns from ratio back-adjusted continuous data | `labels/fwd_ret_5d.parquet`, `labels/fwd_ret_21d.parquet`, each with a `.digest.json` sidecar |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | Term structure, carry, momentum, and roll-return features | `features/financial.parquet` |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | Expanding-window ARIMA and HMM features via statsforecast | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7-9 | Feature-label IC diagnostics across 30 products and 7 sectors | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | Ridge, LASSO, ElasticNet on carry and momentum signals | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | LightGBM testing non-linear carry and momentum interactions | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| Tabular DL | [`08_tabular_dl`](08_tabular_dl.ipynb) | Ch12 | TabM rank-1 adapter MLP ensemble on flat features | Training runs and prediction sets; checkpoints under `run_log/training/tabular_dl/` |
| LSTM | [`09_dl_lstm`](09_dl_lstm.ipynb) | Ch13 | Gated recurrence on the 30-product daily panel | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Latent factors (index) | [`10_latent_factors`](10_latent_factors.ipynb) | Ch14 | Index of the two latent-factor notebooks below | Nothing - it reads the registry |
| PCA | [`10a_pca`](10a_pca.ipynb) | Ch14 | Principal components on the cross-sectional characteristics panel | Training runs and prediction sets |
| SDF | [`10b_stochastic_discount_factor`](10b_stochastic_discount_factor.ipynb) | Ch14 | Stochastic discount factor on the same panel | Training runs and prediction sets |
| Causal DML | [`11_causal_dml`](11_causal_dml.ipynb) | Ch15 | Does the carry signal cause future returns or proxy for risk? | A row in the registry's `causal_runs` |
| Model Analysis | [`12_model_analysis`](12_model_analysis.ipynb) | Ch11-15 | Cross-model IC comparison and fold stability diagnostics | Nothing - it reads the registry |
| Backtest | [`13_backtest`](13_backtest.ipynb) | Ch16 | Long-short carry-ranked strategy simulation | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`14_portfolio_management`](14_portfolio_management.ipynb) | Ch17 | Equal-risk, score-weighted, and sector-constrained allocation | One backtest run per allocation method, same artifact layout |
| Costs | [`15_costs`](15_costs.ipynb) | Ch18 | Commission, spread, and roll slippage impact analysis | One backtest run per cost level, same artifact layout |
| Risk | [`16_risk_management`](16_risk_management.ipynb) | Ch19 | Position-level risk overlays (stop-loss, trailing stops, time exits) | One backtest run per overlay variant, same artifact layout |
| Strategy Analysis | [`17_strategy_analysis`](17_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with uncertainty-aware metrics | `results/strategy_assessment.json`, `20_strategy_synthesis/output/cme_futures/cme_futures_tearsheet.html`; fills the registry's `cohort_metrics` and `backtest_paired_metrics` tables, but only when either is missing or empty |

## Margin Model

Per-product margin is computed once from CME's outright maintenance rates (published on `cmegroup.com`) and 2025-12-31 front-month settlement prices, then expressed as a fraction of notional via `ContractSpec.margin_pct = (initial, maintenance)` in `data/futures/market/futures_specs.yaml`. The engine applies the ratio to each historical bar's notional, so the dollar margin moves with price even though the rate is anchored at one point. For the 8 products not covered by the CSV (the 4 equity-index e-minis ES/NQ/YM/RTY and 4 energies CL/NG/HO/RB), per-category SPAN-style initial-margin approximations are used (initial: 5% equity_index, 8% energy; the table reports maintenance, derived as initial ÷ 1.10 per the CME SPAN convention).

This is a stable-pct approximation. CME publishes maintenance dollars in scan-volume steps that adjust roughly with realized volatility. Historical SPAN snapshots (free up to ~5 years from CME's [historical-margins page](https://www.cmegroup.com/solutions/risk-management/margin-services/historical-margins.html) and paid beyond that via [CME Datamine catalog F001](https://datamine.new.cmegroup.com/catalog?category=F001)) would let us anchor period-specific ratios; the marginal effect on conclusions for 30 liquid CME products is small but non-zero. The table below shows representative pct drift between start- and end-of-window prices for products spanning the secular regimes:

| Product | StartDate | StartPx | EndDate | EndPx | Ratio | Anchored pct | Start-window implied pct | Drift |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ES | 2011-01-03 | 1,103 | 2025-12-31 | 6,942 | 6.30× | 4.54% | 28.61% | +530% |
| NQ | 2011-01-03 | 2,612 | 2025-12-31 | 25,657 | 9.82× | 4.54% | 44.64% | +882% |
| GC | 2011-01-03 | 1,978 | 2025-12-31 | 4,351 | 2.20× | 6.28% | 13.82% | +120% |
| ZN | 2011-01-03 | 100.9 | 2025-12-31 | 112.6 | 1.12× | 1.67% | 1.86% | +12% |
| CL | 2011-01-03 | 189.6 | 2025-12-31 | 57.9 | 0.31× | 7.27% | 2.22% | −69% |
| ZC | 2011-01-03 | 723.2 | 2025-12-30 | 440.5 | 0.61× | 4.43% | 2.70% | −39% |
| NG | 2011-01-03 | 221.3 | 2025-12-31 | 3.97 | 0.018× | 7.27% | 0.13% | −98% |

Anchored pct is closest to truth near 2025-12-31; back in 2011 the stable-pct approximation under-margins high-momentum equity indices (engine accepts orders a live broker may have rejected) and over-margins crashed commodities (engine rejects orders a live broker may have accepted). Net effect on absolute Sharpe in the 2011–2015 portion of the validation window is on the order of single-digit percent; relative comparisons across families, allocators, and cost variants are unaffected. Holdout (2024–2025) is anchored at the same window as the pct calculation, so this drift does not bear on the headline holdout numbers.

Account sizing: `initial_cash` is **$10M** in `config/setup.yaml`. The 30-product universe spans contract notionals from ≈$35k (NG) to ≈$22M (ZT 2-year T-Note), and the engine sizes positions as `target_notional / contract_notional → integer contracts`. At k=5 per side (10 positions total) the per-position dollar budget is 5% × cash; $10M clears the binding constraint (ES at ≈$347k holdout notional in the late window) and lets all 30 products participate. NQ in the very-late holdout (Dec 2025, peak notional ≈$513k vs $500k per-position budget) is the one residual integer-share footnote.

## Key Results

The IC-credible signal and the highest-Sharpe lineage both sit on the 5-day primary label.

**Signal quality.** On the 5-day primary label, the latent-factor SDF is the one family whose HAC 95% CI excludes zero on 1,290 daily IC observations: IC +0.042 [+0.017, +0.068] (t_HAC = 3.25, p = 0.001). GBM (leaves_31_mse) is positive at +0.025 [−0.002, +0.053] (t_HAC = 1.80, p = 0.072), but its interval touches zero. On the 21-day variant the SDF again clears — IC +0.070 [+0.029, +0.112] (t_HAC = 3.31, p = 0.001) — while linear (lasso) +0.040 [−0.032, +0.112], GBM −0.004, tabular DL (tabm) −0.009, and deep_learning (lstm_h64) −0.033 straddle or sit below zero. The predictive signal concentrates in the carry-driven cross-section captured by the latent-factor SDF.

**Strategy-stage performance with CIs.** The cross-stage rank-1 lineage by validation Sharpe is gbm/`leaves_7_huber` on `fwd_ret_5d`: equal-weight long-short top-5 selection at the signal stage, `score_weighted` top-5 allocator at the portfolio stage, and a 3.3% trailing-stop overlay at the risk stage. Validation Sharpe **1.264** [+0.345, +2.087] over 1,290 daily periods (CAGR +35.1%, MaxDD −35.6%, 1,939 trades, Sortino 1.44, PSR p = 0.002). Selection adjustment is read from the fwd_ret_5d label cohort (k = 320 variants across all stages and families): **DSR_ER = 0.038, p = 0.005** — the leader survives deflation at the label scale. Effective-rank corrected trials = 15.6 of nominal 320; expected-max Sharpe under noise = 0.041. The fwd_ret_21d label cohort (k = 330) also survives: DSR_ER = 0.027, p = 0.044.

**Holdout closure.** The same `gbm/leaves_7_huber/fwd_ret_5d` lineage on the 2024-2025 holdout window: Sharpe **1.142** [−0.19, +2.34] over 511 daily periods (CAGR +35.9%, MaxDD −23.1%, 702 trades, Sortino 1.34, PSR p = 0.049). Paired val→holdout Sharpe diff is −0.103 [−1.68, +1.41] (p = 0.90, straddles zero — no significant val→holdout decay). Strategy vs holdout-window equal-weight benchmark: Sharpe diff +0.364 [−1.43, +2.05] (p = 0.69, straddles zero). The two-year commodity holdout is too short to resolve dispersion either way.

**Friction floor.** The cost-grid sweep runs at 0/1/2/3/5/7/10/15/20/30/50 bps total cost per leg (commission and slippage split evenly, applied symmetrically at entry and exit). Both curves below are allocation-stage combos, measured before the risk overlay. `15_costs.ipynb` sweeps the top allocation-stage combo on the primary label — latent_factors `sdf` × `mvo_ledoit_wolf` — the most cost-sensitive candidate, because MVO rebalances to full mean-variance weights each period: Sharpe 1.112 (0 bps) → 0.785 (5) → 0.444 (10) → 0.166 (15) → −0.131 (20) → −1.948 (50), breaking even near 17 bps. The allocation combo that carries the shipped lineage — gbm/`leaves_7_huber` × `score_weighted` (top-5) — turns over far less and stays positive across the whole grid: Sharpe 0.884 (0 bps) → 0.814 (5) → 0.749 (10) → 0.623 (20) → 0.483 (30) → 0.235 (50). The 3.3% trailing-stop overlay applied at the risk stage lifts this lineage's validation Sharpe to 1.264 but is not part of the cost sweep. At CME-typical friction (1–5 bps per leg for liquid contracts) both hold well above zero; the sdf/MVO allocation-stage leader is the one that erodes at institutional 10–20 bps costs, while the concentrated `score_weighted` lineage does not. The binding constraint on the shipped strategy is signal stability across regimes, not cost.

## Running

```bash
# From repo root
uv run python case_studies/cme_futures/01_feasibility_analysis.py
uv run python case_studies/cme_futures/02_labels.py
uv run python case_studies/cme_futures/03_financial_features.py
uv run python case_studies/cme_futures/04_model_based_features.py
uv run python case_studies/cme_futures/05_evaluation.py
uv run python case_studies/cme_futures/06_linear.py
uv run python case_studies/cme_futures/07_gbm.py
uv run python case_studies/cme_futures/08_tabular_dl.py
uv run python case_studies/cme_futures/09_dl_lstm.py
uv run python case_studies/cme_futures/10a_pca.py
uv run python case_studies/cme_futures/10b_stochastic_discount_factor.py
uv run python case_studies/cme_futures/10_latent_factors.py   # summarizes 10a-10b
uv run python case_studies/cme_futures/11_causal_dml.py
uv run python case_studies/cme_futures/12_model_analysis.py
uv run python case_studies/cme_futures/13_backtest.py
uv run python case_studies/cme_futures/14_portfolio_management.py
uv run python case_studies/cme_futures/15_costs.py
uv run python case_studies/cme_futures/16_risk_management.py
uv run python case_studies/cme_futures/17_strategy_analysis.py
```

## Run Log

Model training runs, predictions, and backtest results are tracked in a content-addressed registry under `run_log/registry.db`.
