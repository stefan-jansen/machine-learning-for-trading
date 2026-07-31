# Case Study: NASDAQ-100 Microstructure

This case study uses AlgoSeek TAQ-derived 15-minute bars for 114 NASDAQ-100 constituents to test whether microstructure signals -- order flow, quote staleness, relative spreads -- produce tradeable intraday alpha. This is the highest-frequency case in the book, and it is designed to show how a dominant cost floor makes a raw signal loss-making on the full universe -- and how two disciplined adjustments, a cost-feasible universe screen and ensemble model selection, recover it. The naive build fails; the disciplined build claws the holdout back from clearly-negative to marginally-positive. That iteration -- diagnose the cost problem, screen the universe, treat model selection as estimation under uncertainty -- is the lesson.

## At a Glance

| Property | Value |
|----------|-------|
| Asset Class | US equities (NASDAQ-100) |
| Frequency | 15-minute bars |
| Universe | 114 stocks |
| History | 2020--2021 |
| Primary Label | fwd_ret_15m |
| CV Folds | 2 (6M train, 6M val) |
| Cost Model | per_share_plus_spread ($0.0035/share + measured half-spread; 5 bps friction floor) |

## Pipeline

| Stage | Notebook | Chapter | Description | Writes |
|-------|----------|---------|-------------|--------|
| Feasibility | [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) | Ch6 | Measured per-symbol round-trip cost, breadth and move clearance by horizon, return persistence, walk-forward demo | `liquidity_profile.parquet` |
| Labels | [`02_labels`](02_labels.ipynb) | Ch7 | 15-minute, 5-minute, and 60-minute forward returns | One parquet per label in `labels/` (`fwd_ret_15m` plus the `fwd_ret_5m`, `fwd_ret_60m`, `fwd_dir_15m` variants) |
| Features | [`03_financial_features`](03_financial_features.ipynb) | Ch8 | Order-flow, spread, volatility, and microstructure features | `features/financial.parquet` |
| Temporal | [`04_model_based_features`](04_model_based_features.ipynb) | Ch9 | Walk-forward temporal features for intraday patterns | `features/model_based.parquet` |
| Evaluation | [`05_evaluation`](05_evaluation.ipynb) | Ch7--9 | Feature-label IC diagnostics for 66 financial and temporal features | `evaluation/triage_ledger.parquet`, `evaluation/ic_timeseries.parquet` |
| Linear | [`06_linear`](06_linear.ipynb) | Ch11 | Ridge baseline on the richest feature space in the book | Training runs and prediction sets in `run_log/registry.db`; coefficients under `run_log/training/{hash}/`, scores under `run_log/predictions/{hash}/` |
| GBM | [`07_gbm`](07_gbm.ipynb) | Ch12 | LightGBM on 13M+ training samples at 15-minute frequency | Training runs and prediction sets; boosters, `learning_curves.parquet`, and `fold_metrics.parquet` under `run_log/training/{hash}/` |
| NLinear | [`08_dl_nlinear`](08_dl_nlinear.ipynb) | Ch13 | Minimal temporal baseline for the intraday microstructure signal | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| LSTM | [`09_dl_lstm`](09_dl_lstm.ipynb) | Ch13 | Recurrent memory for short-lived order-flow and spread dynamics | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| TCN | [`10_dl_tcn`](10_dl_tcn.ipynb) | Ch13 | Dilated causal convolutions for intraday temporal patterns | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| PatchTST | [`11_dl_patchtst`](11_dl_patchtst.ipynb) | Ch13 | Multi-scale patch attention on minute-bar sequences | Training runs and prediction sets; checkpoints under `run_log/training/deep_learning/` |
| Causal DML | [`12_causal_dml`](12_causal_dml.ipynb) | Ch15 | Does signed volume share cause future 15-minute returns? | A row in the registry's `causal_runs` |
| Model Analysis | [`13_model_analysis`](13_model_analysis.ipynb) | -- | Cross-model IC comparison and fold stability diagnostics | Nothing - it reads the registry |
| Backtest | [`14_backtest`](14_backtest.ipynb) | Ch16 | Strategy simulation designed to demonstrate cost-driven failure | One backtest run per prediction set and entry scheme; `daily_returns.parquet`, `weights.parquet`, `trades.parquet`, `fills.parquet`, `equity.parquet`, `portfolio_state.parquet`, and `spec.json` under `run_log/backtest/{hash}/` |
| Portfolio | [`15_portfolio_management`](15_portfolio_management.ipynb) | Ch17 | Allocation methods under dollar-neutral intraday constraints | One backtest run per allocation method, same artifact layout |
| Costs | [`16_costs`](16_costs.ipynb) | Ch18 | Flagship cost analysis: spread, impact, and commission decomposition | One backtest run per cost level, same artifact layout |
| Risk | [`17_risk_management`](17_risk_management.ipynb) | Ch19 | Intraday risk controls and position-level exit rules | One backtest run per overlay variant, same artifact layout |
| Strategy Analysis | [`18_strategy_analysis`](18_strategy_analysis.ipynb) | Ch20 | End-to-end strategy assessment with IC, Sharpe, and cost analysis | `results/strategy_assessment.json`, `20_strategy_synthesis/output/nasdaq100_microstructure/nasdaq100_microstructure_tearsheet.html` |

## Key Results

The book's study in **cost and selection discipline**: a raw intraday cross-sectional signal that is friction-dominated on the full universe, and the two disciplined adjustments that recover it. Built the naive way — the full NASDAQ-100, no careful spread analysis — the strategy is loss-making across the sweep and the holdout Sharpe lands at **−0.89**. Careful cost analysis shows the edge is real but concentrated in names too expensive to trade at this cadence; screening to a cost-feasible universe and replacing the noisy single-best model with an ensemble walks the holdout from **−0.89 → −0.21 → +0.53**. The recovered number is modest and its confidence interval is wide — the lesson is *how disciplined adjustments change the answer*, not a claim of a deployable edge.

**Signal quality**: The highest-IC configuration on the primary `fwd_ret_15m` label is gbm/leaves_7_mae at +0.0060 (HAC CI [+0.0026, +0.0094], excludes zero); linear/ridge_a1000000.0 follows at +0.0049 (CI [+0.0022, +0.0076], excludes zero). IC strengthens monotonically toward the short end — the highest-IC GBM configuration runs +0.0104 at fwd_ret_5m. At fwd_ret_60m the linear/ridge_a1000000.0 IC is +0.0069 with HAC CI [-0.0021, +0.0159] (t=1.50, p=0.13) — point estimate faintly positive but the CI straddles zero. The most-shrunk regularizers achieve the highest IC at every horizon: small-leaf MAE GBM and ridge with $\alpha \in \{10^6, 10^7\}$.

**The two adjustments, on the holdout.** Each step is a single deliberate change, and each is one holdout consultation of the chosen configuration (all three reproducible from `run_log/registry.db`):

| Step | Change | Holdout Sharpe | CI95 |
|---|---|---|---|
| 1. Naive baseline | full universe, single-best model | **−0.89** | [−3.59, +2.32] |
| 2. Cost-feasible universe | screen to the cheapest-to-trade names, frozen per split | **−0.21** | [−2.37, +3.64] |
| 3. Ensemble selection | average the 12-model set instead of the single-best pick | **+0.53** | [−1.94, +3.07] |

Adjustment 1 is the microstructure lesson: the cost-expensive tail of the 114-name panel consumes the intraday edge, so the full universe collapses out of sample while the cost-feasible subset does not (see `16_costs.py` for the full-vs-screened contrast). Adjustment 2 is the selection lesson: the per-model validation Sharpes have enormous, completely overlapping 95% CIs, so the single-best pick is noise — averaging the model set (an ensemble) is estimation under selection uncertainty. The ensemble is a **robustness device against selection noise, not a return booster**: it converts the single-best pick's out-of-sample loss (−0.21) into an honest modest positive rather than adding alpha. **Every holdout CI above is wide and straddles zero (n≈128 days); the recovery is a point-estimate rescue from clearly-negative to marginally-positive, not a significance result.** The naive-baseline row uses the direction label `fwd_dir_15m` while the screened steps use `fwd_ret_60m`; the ladder is directionally honest — the full universe is deeply negative regardless of label — but it is not one strategy tuned in place.

**Holdout closure (the ensemble carrier)**: The deployed configuration is the cost-feasible ensemble at holdout Sharpe **+0.53** [−1.94, +3.07] (n=128 trading days) — positive on the point estimate, where the naive full-universe baseline (−0.89) and the single-best cost-feasible pick (−0.21) are not. It remains **below a passive equal-weight buy-and-hold of the same screened basket** over the window: the recovery is to *viability*, not to out-performance — the active signal claws back to positive but does not beat simply holding the cost-feasible names. The strategy-vs-equal-weight paired-difference interval is not populated in the registry for this case study (`backtest_paired_metrics` has no producer here), so kill gate 2 reads **no data** rather than a spurious pass — the gate helpers treat the missing bootstrap as no-evidence, not as a green light. The honest headline: the ensemble recovers a positive point estimate, not statistical significance and not a market-beating edge; every interval above straddles zero.

**Friction floor — why the naive build fails**: The bps cost trajectory for the naive-lineage prediction (`1c4327c80284`, linear/ridge_a1000000.0/fwd_ret_60m) walks monotonically from Sharpe -0.78 at zero cost through -10 at the 50 bps tail; the CI upper bound clips positive only at the zero / 1 / 2 / 3 bps cells. The realistic NQ100 large-cap half-spread of 1–3 bps plus the per-share $0.0035 floor (~5 bps friction floor) sits inside the negative band. This is the diagnostic behind Adjustment 1: on the full universe the cost floor swamps the edge, which is why screening to the cheapest-to-trade names is load-bearing rather than cosmetic — the expensive tail is where the cost is, and dropping it is what moves the holdout off the floor. (Position-level overlays alone do not: the 20-row-per-label risk-overlay sweep — trailing_stop, stop_loss, time_exit — is uniformly negative across all three regression labels on the naive lineage.) Portfolio-level kill switches (max-drawdown breaker, daily-loss limit) are NOT swept for model selection — their permanent-halt semantics produced zero-std Sharpe artifacts in earlier passes; they remain available as Ch19 §19.8 governance instruments.

## Running

```bash
# From repo root
uv run python case_studies/nasdaq100_microstructure/01_feasibility_analysis.py
uv run python case_studies/nasdaq100_microstructure/02_labels.py
uv run python case_studies/nasdaq100_microstructure/03_financial_features.py
uv run python case_studies/nasdaq100_microstructure/04_model_based_features.py
uv run python case_studies/nasdaq100_microstructure/05_evaluation.py
uv run python case_studies/nasdaq100_microstructure/06_linear.py
uv run python case_studies/nasdaq100_microstructure/07_gbm.py
uv run python case_studies/nasdaq100_microstructure/08_dl_nlinear.py
uv run python case_studies/nasdaq100_microstructure/09_dl_lstm.py
uv run python case_studies/nasdaq100_microstructure/10_dl_tcn.py
uv run python case_studies/nasdaq100_microstructure/11_dl_patchtst.py
uv run python case_studies/nasdaq100_microstructure/12_causal_dml.py
uv run python case_studies/nasdaq100_microstructure/13_model_analysis.py
uv run python case_studies/nasdaq100_microstructure/14_backtest.py
uv run python case_studies/nasdaq100_microstructure/15_portfolio_management.py
uv run python case_studies/nasdaq100_microstructure/16_costs.py
uv run python case_studies/nasdaq100_microstructure/17_risk_management.py
uv run python case_studies/nasdaq100_microstructure/18_strategy_analysis.py
```

## Run Log

Model training runs, predictions, and backtest results are tracked in a content-addressed registry under `run_log/registry.db`.
