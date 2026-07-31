# Case Study: US Firm Characteristics

This case study uses anonymized monthly firm characteristics from NASDAQ Data Link to test the canonical factor investing question: can ML improve on traditional long-short decile sorts when point-in-time accounting lags, survivorship bias, and era-dependent transaction costs are treated as binding constraints. With 57 firm-level characteristics spanning valuation, profitability, momentum, and risk across roughly 2,500 stocks (1996–2016), this is the most feature-rich fundamental dataset in the book and the natural home for latent factor models.

The case study runs at monthly cadence with a 6-month accounting lag enforced for point-in-time compliance, equal-weight long-short decile portfolios, dollar-neutral construction, and an era-dependent cost grid (pre-decimalization spreads 15–30 bps; post-2001 spreads 5–15 bps). Ten CV folds with 10-year training windows and 1-year validation provide the deepest cross-validation in the book; the calendar 2016 holdout supplies 12 monthly out-of-sample observations.

The teaching arc threads four claims that must be evaluated jointly: regression-vs-classification labels move IC by an order of magnitude on the same features, GBM and the supervised autoencoder both produce credible standalone signals while linear and IPCA do not, holdout decay over 12 periods is consistent with material erosion under wide CIs rather than a precise estimate, and the long-short legs cluster in small-cap, wide-spread, high-idio-vol names so the universe-mean cost grid understates the friction the strategy actually faces.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | US equities (fundamental characteristics, long-short) |
| Frequency | Monthly |
| Universe | ~2,500 stocks (price > $5, ADV > $1M) |
| History | 1996–2016 |
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
| Costs | [`13_costs`](13_costs.ipynb) | Ch18 | Era-dependent cost grid spanning pre- and post-decimalization | One backtest run per cost level, same artifact layout |
| Risk | [`14_risk_management`](14_risk_management.ipynb) | Ch19 | Position-level and portfolio-level risk overlays on the monthly cadence | Nothing - the position-control loop is gated off on the vectorized path and the portfolio-control list is asserted empty, so no overlay variant is registered |
| Strategy Analysis | [`15_strategy_analysis`](15_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with uncertainty-aware metrics | Nothing - the tear sheet needs a `trades.parquet` the vectorized backtester never emits, so the stage takes its no-trades branch |

## Key Results

A genuine validation edge: a long-short equal-weight decile portfolio on the winsorized regression label clears both kill gates, paired with a positive but wide-CI holdout (12 monthly observations) and a deployable range bounded by the small-cap concentration of the selected legs.

**Signal direction**: On the winsorized regression label `fwd_ret_1m_win`, GBM is the signal-stage Sharpe leader across families. The rank-1 lineage's daily-pooled IC is +0.037 [+0.026, +0.048] (HAC t=6.62, p=1.4e-09, n=110 monthly periods, 76% positive), an interval that clears zero. The signal-stage family rank-1s on this label are gbm 2.75, tabular_dl 2.18, linear 1.24, and latent_factors 0.22 by validation Sharpe; GBM's non-linear value-quality-momentum interactions carry the signal, and per the latent-factor comparison (Ch14 §8) the supervised autoencoder is the one other family with a credible standalone signal. The cross-stage rank-1 strategy lineage builds on the `gbm/default_huber` prediction.

**Strategy-stage performance with CIs**: The cross-stage val rank-1 is the allocation-stage backtest `e676e1989e1f`: a long-short equal-weight top-50 decile portfolio (`gbm/default_huber` on `fwd_ret_1m_win`), rebalanced monthly at next-month-open. Validation Sharpe 2.754 [2.332, 3.373] (PSR p=1.1e-12), Sortino 6.04, max drawdown -7.9%, Calmar 7.27, CAGR +57% over 110 monthly periods; all ten CV folds positive (Sharpe range 6.16 to 25.39). No allocator improves on equal weight in the re-swept registry: the signal → allocation paired difference is exactly 0.00 [0.00, 0.00] because risk-parity, HRP, inverse-vol, and MVO reproduce or trail the plain equal-weight top-50 portfolio, so the edge sits in the signal rather than the sizing. Selection-bias adjustment: the carrier belongs to the gbm allocation-stage `fwd_ret_1m_win` family cohort (K_variants=211), whose DSR-deflated leader clears every deflation layer (DSR_ER 0.626, DSR_MP 0.673, PBO 0.063 from CSCV on the 10 fold Sharpes, RAS lower bound 1.44); the broader stage-label (K=411, DSR_ER 0.515) and label (K=611, DSR_ER 0.513) cohorts clear deflation as well. The carrier's own Sharpe (2.754) sits just below the cohort leader (2.82) inside a tight band. Universal kill gate 1 (validation Sharpe CI lower bound 2.332 ≥ 0) PASSES. The risk_overlay stage carries no rows (deferred sweep), so the rank-1 is resolved across the signal and allocation stages.

**Holdout closure**: The holdout backtest `4b7f87bb619b` retrains the same equal-weight top-50 lineage (`gbm/default_huber` on `fwd_ret_1m_win`) on the calendar-2016 window (12 monthly observations). Holdout Sharpe +1.771 [-0.391, +4.597] (PSR p=0.142). Validation Sharpe 2.754 → holdout 1.771 is a point decay of ≈0.98, but the val-vs-holdout paired difference is -0.983 [-2.864, +1.664] and straddles zero, so the decay is not statistically resolved at n=12. The holdout 95% CI is wide by construction (12 periods) and its lower bound sits below zero, so unlike the validation edge the holdout does not on its own exclude zero. Universal kill gate 2 (holdout strategy-vs-EW paired bootstrap does not exclude zero negatively) PASSES: the difference is +0.537 [-2.811, +3.618], straddling zero with a positive point estimate, so the strategy is not resolved as worse than the passive decile universe. Gate 2 is a *not-worse-than-equal-weight* test: a difference whose CI includes zero is statistically indistinguishable from EW and passes; it fails only when the CI lies entirely below zero. Chapter 20 places US Firms in the group that produced a credible standalone signal whose out-of-sample precision is limited by the 12-period window, not the group that never resolved an edge.

**Friction floor**: On the rank-1 lineage the registry cost grid (11 symmetric per-leg levels, 0 to 25 bps commission-plus-slippage each side) holds the validation Sharpe strongly positive throughout: 2.926 at zero cost, 2.754 at the deployment cost (12.5 bps round-trip-additive), and 2.240 at the 25 bps-per-leg extreme. The binding constraint is the extended turnover-by-cost grid in the spine notebook §5: at 196% mean monthly turnover both the long and short legs cluster at small-cap (LME negative), wide-spread, and high-idio-vol names, so universe-mean spreads understate execution costs (the Avramov, Cheng, and Metzker (2020) critique materialized). At micro-cap-realistic spreads (200+ bps one-way) the gross profile erodes materially. This is a capacity-binding constraint and an input to Ch20's cost-survival aggregation, not a kill condition on the validation result.

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
uv run python case_studies/us_firm_characteristics/13_costs.py
uv run python case_studies/us_firm_characteristics/14_risk_management.py
uv run python case_studies/us_firm_characteristics/15_strategy_analysis.py
```

## Run Log

Model training runs, predictions, and backtest results are tracked in a content-addressed registry under `run_log/registry.db`.
