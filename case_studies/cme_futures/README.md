# Case Study: CME Futures

This case study uses daily Databento data on 30 CME futures products spanning seven sectors (equity indices, treasuries, energy, metals, currencies, agriculture, and livestock) to test whether carry and term-structure signals produce tradeable alpha at a weekly cadence. Futures have a return decomposition that splits into spot and roll components, natural sector groupings that constrain diversification, and inherent leverage that magnifies both signal and friction.

The pipeline runs a long-short carry-ranked strategy with weekly Friday-close decisions and Monday-open execution, trades 30 front-month continuous contracts built with ratio back-adjustment, and prices in commission, bid-ask spread, and roll slippage. Two results sit side by side and belong to different model families. The credible cross-sectional signal is the latent-factor SDF's: IC +0.034 (HAC 95% CI [+0.006, +0.062], t_HAC = 2.36), the only family clearing zero. The strategy the case study ships is a different lineage — GBM with an `hrp` allocator and a 2% trailing stop — and it posts a validation Sharpe of **1.236** [+0.397, +2.126]. The teaching point is that portfolio Sharpe comes from magnitude at the top of the cross-section rather than from average IC, which is also why the family with the best IC is not the family that carries. Its holdout Sharpe is **0.287** [−1.034, +1.638], an interval wide enough to contain both the validation estimate and zero, which is the second teaching point: a two-year window on weekly decisions does not adjudicate a strategy.

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
| Risk | [`15_risk_management`](15_risk_management.ipynb) | Ch19 | Position-level risk overlays (stop-loss, trailing stops, time exits) | One backtest run per overlay variant, same artifact layout |
| Costs | [`16_costs`](16_costs.ipynb) | Ch18 | Commission, spread, and roll slippage impact analysis | One backtest run per cost level, same artifact layout |
| Holdout predictions | [`17_holdout_predictions`](17_holdout_predictions.ipynb) | Ch20 | Refits the resolved carrier through the holdout fold and predicts 2024-2025 | One training run under a new identity whose CV declares the holdout fold, and one prediction set at `split='holdout'` |
| Holdout backtest | [`18_holdout_backtest`](18_holdout_backtest.ipynb) | Ch20 | Replays the carrier's own strategy specification on the holdout prediction set | One backtest run at `stage='holdout'`, same artifact layout |
| Strategy Analysis | [`19_strategy_analysis`](19_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with uncertainty-aware metrics | `results/strategy_assessment.json`, `20_strategy_synthesis/output/cme_futures/cme_futures_tearsheet.html`; rebuilds the registry's `cohort_metrics` and `backtest_paired_metrics` tables on every canonical run, pruning rows a previous selection wrote |

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

All figures below are read from `run_log/registry.db` as rebuilt on 2026-08-30: 100 training
runs, 496 validation prediction sets, 1,151 backtests, and one holdout evaluation. An earlier
edition of this section reported a different lineage, a holdout Sharpe of 1.142 over 511 periods,
and IC counts of 1,290 - none of which the current registry contains. They described a registry
that was replaced, and they are not reproduced here.

**Signal quality.** The latent-factor SDF is the one family whose HAC 95% CI excludes zero, and it
does so on both horizons. On the 5-day primary label: IC +0.034 [+0.006, +0.062] (t_HAC = 2.36) over
1,285 daily observations. On the 21-day variant: IC +0.064 [+0.020, +0.109] (t_HAC = 2.82) over
1,269. Nothing else clears on either. The best GBM on the primary label is `leaves_15_mae` at +0.021
[−0.009, +0.050] (t_HAC = 1.39); linear (`lasso_f0.85`) +0.018, deep_learning (`lstm_h64`) +0.010 and
tabular DL (`tabm_l`) +0.009 all straddle zero, and on the 21-day label GBM (−0.007) and tabular DL
(−0.003) sit below it. The predictive signal concentrates in the carry-driven cross-section the
latent-factor SDF captures.

**The shipped strategy.** The carrier is resolved by `resolve_solvent_carrier`, which ranks
candidates on the 1,270 sessions they all price rather than on each one's own window - a raw ranking
of the Sharpe column names a different family and a different horizon, and rewards whichever
candidate had the most forgiving span. The carrier is gbm/`leaves_31_mse` on `fwd_ret_5d`, from the
risk-overlay stage: equal-weight long-short top-5 at the signal stage, an `hrp` allocator with a
63-bar volatility window at the portfolio stage, and a 2% trailing stop at the risk stage.

Validation Sharpe **1.236** [+0.397, +2.126] over 1,286 daily periods (CAGR +19.5%, MaxDD −25.9%,
2,355 trades, Sortino 2.04, PSR p = 0.003). On common support the same run reads 1.294.

Selection adjustment, from the `fwd_ret_5d` label cohort: K = 550 candidates, effective trials
12.8 after correlation correction, **DSR_ER = 0.045**. The carrier survives deflation at the label
scale. The `fwd_ret_21d` cohort reads 0.037 on K = 554. Those two do not add to the 1,140-candidate
pool below and are not meant to: a cohort drops any backtest whose prediction set has a fold with no
computable IC, which is 36 of the 1,140 here. A prediction that could not be scored on every fold is
not a variant the selection could have chosen. Both were absent from earlier editions of
this file, which quoted deflation numbers `cohort_metrics` did not contain - not because the
computation was wrong but because `19_strategy_analysis` never called for one, while three sibling
case studies did. It calls for it now.

Two distinctions matter when reading 1.236 as a selected maximum. The pool is the 1,140 candidates
across the signal, allocation and risk-overlay stages; the registry's other 12 backtests are the 11
cost-sensitivity cells and the holdout itself, neither of which is selected from. And 1.236 is not
the statistic selection maximised - that is common-support Sharpe, on which this carrier reads 1.294.
Ranked on their own full windows the pool's maximum is 1.274, a different configuration. So 1.236 is
the carrier's full-window Sharpe reported after a selection made on a different number, and it is
optimistic by construction in the ordinary way a pool maximum is.

**Holdout.** The same configuration, refitted through the holdout fold and replayed on 2024-2025:
Sharpe **0.287** [−1.034, +1.638] over 516 daily periods (CAGR +2.8%, MaxDD −13.3%, 970 trades,
Sortino 0.42, PSR p = 0.333). The refit is genuine rather than a validation-fitted model scored on a
later window - the holdout training run carries its own identity, `365d0ce706e2`, and its own CV
declares the holdout fold.

**The honest reading is that this establishes very little, in either direction.** The point estimate
falls from 1.236 to 0.287, which invites a decay story, and the interval will not support one:
comparing the carrier's validation series against its own holdout replay gives a Sharpe difference of
**−0.949 [−2.583, +0.624], p = 0.246**. The two windows are disjoint - that is what a holdout is - so
each side is bootstrapped independently over its own sessions, 1,286 in validation against 516 in the
holdout, rather than paired on shared dates. The difference in the point estimates is therefore all
the comparison has to work with; what it adds is the interval around it, and that interval contains
zero comfortably.

Against a holdout-window equal-weight benchmark the strategy reads −0.470 [−2.071, +1.136],
p = 0.554, also indistinguishable. A 516-session window on a weekly rebalance carries too few
independent decisions to separate a strategy that decayed from one that had two ordinary years. What
the holdout does establish is the one thing it exists for: no choice in this case study was made on
this period.

**Friction floor.** One curve, not two. `16_costs.ipynb` sweeps the shipped carrier including its
risk overlay, where a previous edition swept two pre-overlay allocation-stage combinations that are
not what the case study reports. Total cost per leg, commission and slippage split evenly and applied
symmetrically at entry and exit:

| bps | 0 | 1 | 2 | 3 | 5 | 7 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sharpe | 1.499 | 1.415 | 1.341 | 1.285 | 1.182 | 1.043 | 0.857 | 0.549 | 0.227 | −0.390 | −1.582 |

Break-even sits between 20 and 30 bps per leg. At CME-typical friction for liquid contracts, 1-5 bps,
the strategy retains most of its frictionless Sharpe. The zero-cost figure exceeds the carrier's own
registered 1.236 because the carrier's run carries `setup.yaml`'s declared friction and the 0 bps
cell removes it. Cost is not the binding constraint on this strategy; the holdout interval is.

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
uv run python case_studies/cme_futures/15_risk_management.py
uv run python case_studies/cme_futures/16_costs.py
uv run python case_studies/cme_futures/17_holdout_predictions.py
uv run python case_studies/cme_futures/18_holdout_backtest.py
uv run python case_studies/cme_futures/19_strategy_analysis.py
```

## Run Log

Model training runs, predictions, and backtest results are tracked in a content-addressed registry under `run_log/registry.db`.
