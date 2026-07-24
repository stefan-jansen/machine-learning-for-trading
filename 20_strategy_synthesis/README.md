# Chapter 20: Strategy Synthesis

The chapter passes nine case studies through the same standardized pipeline (data, features, labels, models, predictions, portfolios, costs, risk overlays) and reads what the resulting cross-section says about translating machine-learning predictions into trading strategies. The unit of comparison is each case study's arc from signal to frozen holdout, not a league-table ranking of Sharpe ratios. The chapter's value is diagnostic: it identifies where the pipeline amplifies or dampens signal, which family of constraints binds in which trading game, and where a second iteration is most likely to pay off.

## Learning Objectives

1. Read a feature triage ledger as a screen for downstream strategy work, and recognize where feature-level survival predicts strategy-level survival and where it does not
2. Distinguish signal quality, portfolio translation, cost survival, and temporal stability as separate evaluation stages
3. Compare how model families behave after the full pipeline, and recognize when several configurations cluster within measurement error of each other
4. Diagnose differences between validation and holdout through prediction-quality drift, portfolio-translation drift, and structural break as distinct mechanisms
5. Evaluate strategies under realistic constraints, including instrument-appropriate cost models, capacity limits, and multiple-testing adjustments
6. Identify next-iteration priorities (label redesign, ensembling, feature engineering, allocator research) inside the Ch6 iterative workflow
7. Apply a practitioner workflow that moves from data and diagnostics through signal generation, strategy construction, and frozen holdout validation

## Chapter Sections

| #     | Title                                                       | Core Idea                                                                                       |
|-------|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| 20.1  | The Nine Case Studies, End-to-End                           | Per-CS arc from signal to portfolio to holdout; rank-1 cluster reading where appropriate         |
| 20.2  | Setup and Feature Evaluation Across the Case Studies        | FDR-controlled feature triage; feature survival is a screen, not a strategy-survival predictor   |
| 20.3  | Signal Quality and Prediction Uncertainty                   | IC plus stability bundle (ICIR, positive-fold share, checkpoint sensitivity); label engineering  |
| 20.4  | From Signals to Strategies                                  | Fundamental Law mapping; cadence/breadth/win-rate; family rankings shift across the pipeline     |
| 20.5  | Portfolio Allocation Across the Case Studies                | Allocator cross-section: HRP wins where signal is broad; spread widens when signal is weak       |
| 20.6  | Trading Realism: Costs, Capacity, and Execution             | Per-CS breakeven against assumed cost; SP500 options HTM cascade; bps-of-notional fails options  |
| 20.7  | Risk Overlays and Stability Across Regimes                  | Three decay mechanisms (prediction, translation, regime); overlay effectiveness is strategy-specific |
| 20.8  | Causal Credibility and Confounding Bias                     | DML estimates as a fragility metric; publication-standard threshold; refutation companion         |
| 20.9  | Limitations and a Practitioner Workflow                     | Constraint inventory; ensemble opportunity; sequenced workflow; per-CS next steps                |

## Notebooks

| Notebook                                                                    | What It Does                                                                                              |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| [`00_holdout_predictions`](00_holdout_predictions.ipynb)                    | Generates rank-1-on-holdout predictions per case study via `holdout.py`; populates the holdout split rows |
| [`01_aggregate_synthesis`](01_aggregate_synthesis.ipynb)                    | Aggregates per-CS registries into chapter-wide parquets in `output/`                                      |
| [`02_feature_evaluation`](02_feature_evaluation.ipynb)                      | Builds the cross-CS triage funnel (Table 20.3) and the feature-survival vs strategy-survival figure       |
| [`03_signal_quality`](03_signal_quality.ipynb)                              | Per-CS family-mean IC table (Table 20.4); IC vs Sharpe scatter (Figure 20.2)                              |
| [`04_signal_to_strategy`](04_signal_to_strategy.ipynb)                      | Family cascade table (Table 20.5); Fundamental Law diagnostic (Figure 20.3); top-K sweep (Figure 20.4)    |
| [`05_portfolio_allocation`](05_portfolio_allocation.ipynb)                  | Allocator winners (Table 20.6); best-worst spread (Table 20.7); allocator-signal interplay                |
| [`06_cost_survival`](06_cost_survival.ipynb)                                | Breakeven scorecard (Table 20.8); cost waterfall (Figure 20.5); HTM cascade reading                       |
| [`07_regime_risk`](07_regime_risk.ipynb)                                    | Validation-vs-holdout decay (Figure 20.6); risk overlay impact (Figure 20.7)                              |
| [`08_recommendations`](08_recommendations.ipynb)                            | Pipeline evidence snapshot (Table 20.11); per-CS next-step ledger; ensembling note                        |

## Running the Notebooks

```bash
# From the repository root, in dependency order
uv run python 20_strategy_synthesis/01_aggregate_synthesis.py
uv run python 20_strategy_synthesis/02_feature_evaluation.py
# ... through 08_recommendations.py

# Headless (no display)
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python 20_strategy_synthesis/03_signal_quality.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "20_strategy_synthesis"
```

## Dependencies

Upstream: every case study under `case_studies/` must have a populated `run_log/registry.db` with training, prediction, backtest, and (for §20.8) `causal_runs` rows on the primary label. `00_holdout_predictions` writes the holdout-split rows that downstream notebooks read.

Downstream: none. Ch20 is the synthesis end of the pipeline.

## References

- **Avramov, Cheng, and Metzker** (2020). [Machine Learning vs. Economic Restrictions](https://doi.org/10.2139/ssrn.3450322).
- **Bailey and López de Prado** (2014). [The Deflated Sharpe Ratio](https://doi.org/10.2139/ssrn.2460551).
- **Chernozhukov et al.** (2018). [Double/Debiased Machine Learning for Treatment and Structural Parameters](https://doi.org/10.1111/ectj.12097).
- **Frazzini, Israel, and Moskowitz** (2018). [Trading Costs](https://doi.org/10.2139/ssrn.3229719).
- **Grinold and Kahn** (2000). *Active Portfolio Management*. Second edition.
- **Gu, Kelly, and Xiu** (2020). [Empirical Asset Pricing via Machine Learning](https://doi.org/10.1093/rfs/hhaa009).
- **Harvey, Liu, and Zhu** (2016). [...and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059).
- **López de Prado** (2016). [Building Diversified Portfolios that Outperform Out-of-Sample](https://doi.org/10.3905/jpm.2016.42.4.059).
- **McLean and Pontiff** (2016). [Does Academic Research Destroy Stock Return Predictability?](https://doi.org/10.1111/jofi.12365).
- **Novy-Marx and Velikov** (2016). [A Taxonomy of Anomalies and Their Trading Costs](https://doi.org/10.1093/rfs/hhv063).
- **O'Donovan and Yu** (2025). *Transaction Costs and Cost Mitigation in Option Investment Strategies*.
