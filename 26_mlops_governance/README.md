# Chapter 26: MLOps and Governance

A model has cleared every backtest, the deployment verification of Chapter 25 has taken the system live, and real capital is now flowing through it. Keeping it working — and failing safely when it cannot — is the next problem. Every deployed model decays: regimes shift, competitors discover similar signals, and the relationships the model learned gradually erode. The difference between a profitable trading operation and a capital-destroying one often comes down to how quickly that decay is detected and how safely the system responds.

This chapter builds the post-deployment infrastructure that separates production-grade systems from research prototypes. The central diagnostic is the failure taxonomy in Section 26.1: *technical* failures are verification problems (same inputs producing different outputs); *statistical* failures are performance problems (same outputs no longer predicting returns). Conflating them wastes time and capital. From there, the chapter builds three layers: detection (rolling metrics, drift diagnostics, online detectors), response (shadow mode, A/B, staged rollouts), and automated safety (multi-level circuit breakers). The supporting MLOps stack — feature stores, registries, CI/CD — is treated as enabling infrastructure to be right-sized to team maturity, not adopted wholesale.

All seven notebooks operate on real artifacts from the `us_equities_panel` case study: the sealed holdout boundary, stored prediction streams, real feature panels, and the SQLite registry that tracks every training run. The circuit-breaker notebook drives its market-risk rules with the real SPY 2020 H1 path through the COVID crash; only the infrastructure-latency stream remains synthetic, scoped to make the latency breaker's rolling-average trip inspectable. Broker connectivity stays out of scope.

## Learning Objectives

1. Distinguish technical pipeline divergence from statistical performance decay, and choose the corresponding diagnostic and response workflow.
2. Build a live-monitoring framework that combines data-integrity gates, rolling performance metrics, backtest-to-live comparison, and execution-quality checks.
3. Apply drift diagnostics to production artifacts, including PSI, K-S, SHAP-based feature monitoring, and online detectors such as ADWIN-style methods and DDM.
4. Design a safe model-update workflow using shadow mode, incumbent-vs-candidate evaluation, explicit promotion criteria, staged rollout gates, and tested rollback procedures.
5. Implement multi-level circuit breakers across trade, strategy, portfolio, and system layers, with clear recovery and override discipline.
6. Evaluate and right-size the supporting MLOps stack, including feature stores, data versioning and lineage, model registries, experiment tracking, and CI/CD controls.

## Chapter Sections

| §    | Title                                 | Core Idea                                                                                                |
|------|---------------------------------------|----------------------------------------------------------------------------------------------------------|
| 26.1 | Two Sources of Live Trading Failure   | The technical-vs-statistical failure taxonomy that organizes the rest of the chapter.                    |
| 26.2 | Performance Monitoring                | Rolling metrics, alert thresholds, dashboards, and backtest-to-live realization ratios.                  |
| 26.3 | Drift Detection                       | PSI / K-S for data drift, SHAP for feature drift, ADWIN-style and DDM for concept drift.                 |
| 26.4 | Safe Model Updates                    | Shadow mode → capital-capped A/B → staged rollout, gated by explicit promotion criteria and rollback.    |
| 26.5 | Circuit Breakers and Safety           | Multi-level breakers (loss, position, anomaly, infrastructure) and the CLOSED/OPEN/HALF_OPEN lifecycle.  |
| 26.6 | MLOps Infrastructure Overview        | Feature stores, data versioning, model registries, and CI/CD — right-sized to operational maturity.       |
| 26.7 | Summary                               | The three-layer governance model from detection to response to automated safety.                         |

## Notebooks

### Detection (§26.2–§26.3)

| Notebook                                                            | What It Teaches                                                                                                                                                                                                                                                                       |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`01_drift_monitoring`](01_drift_monitoring.ipynb)                  | Builds the four-panel operational drift dashboard (Figure 26.2) on the `us_equities_panel` holdout stream: per-feature PSI bars, baseline-vs-current prediction-score histogram, rolling 63-day IC, and rolling 63-day hit rate. Disjoint launch/current windows; alert states wired to a configurable K-S p-value threshold.                                                                                                                                                                                                                                                          |
| [`02_online_drift_detection`](02_online_drift_detection.ipynb)      | Runs ADWIN-style (two-window mean-shift) and DDM detectors on real chronological validation-error streams for OLS and ridge linear models. Reports signed lead/lag against an equal-weight market-stress proxy (ADWIN +21–22 days, DDM-OLS +34 days) so "lead" vs "lag" remains interpretable, and connects alert clusters to the 2015 stress windows rather than synthetic change points.                                                                                                                                                                                          |

### Response (§26.4)

| Notebook                                                       | What It Teaches                                                                                                                                                                                                                                                                                  |
|---------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`03_safe_model_rollout`](03_safe_model_rollout.ipynb)         | Walks the full incumbent-candidate rollout protocol on real prediction artifacts: selection on 2010–2014, shadow mode on 2015 with five explicit promotion criteria, capital-capped A/B test, and a staged allocation path stepping the candidate share from 10% to 25%. Produces Figure 26.3's four-panel rollout dashboard.                                                                                                                                                                                                       |

### Automated Safety (§26.5)

| Notebook                                                  | What It Teaches                                                                                                                                                                                                                                                                                  |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`04_circuit_breakers`](04_circuit_breakers.ipynb)        | Implements a four-breaker (drawdown, daily-loss, consecutive-loss, latency) state machine with shared CLOSED → OPEN → HALF_OPEN transitions and a manager event log that records every transition, not just trips. Driven by the real SPY 2020 H1 (Jan–May 2020) path through the COVID crash; the latency stream stays synthetic and scoped to the infrastructure breaker.                                                                                                                                                       |

### MLOps Infrastructure (§26.6)

| Notebook                                                              | What It Teaches                                                                                                                                                                                                                                                                                                                                |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`05_feast_feature_store`](05_feast_feature_store.ipynb)              | Implements feature-store control logic by hand against the real `us_equities_panel` feature tables: feature view definitions, a point-in-time offline join (298k rows from 150k training events), an as-of online snapshot, lineage tracking, and a quantified training-serving skew demonstration (RSI mean abs delta 1.99, max 4.47).        |
| [`05b_feast_live`](05b_feast_live.ipynb)                              | Runs Feast end-to-end on the same artifacts: defines entity and feature views, applies them to a temporary registry, performs offline and online retrieval, and verifies parity with the manual Polars implementation. 6 of 7 features match exactly; `garch_cond_vol` exposes a vintage-tracking diagnostic, not a Feast bug. ~11 GB peak RAM, ~2 min runtime; requires `feast>=0.40`. |
| [`06_mlflow_experiments`](06_mlflow_experiments.ipynb)                | Walks the `us_equities_panel` registry schema (`training_runs`, `prediction_sets`, `backtest_runs`, `cohort_metrics`), builds a searchable experiment catalog, logs 50 training runs into MLflow, and verifies ranking parity across four (family, label) spot-checks — showing governance value comes from disciplined run accounting, not specific tooling.                            |

## Running the Notebooks

```bash
# Production run (from the repo root)
uv run python 26_mlops_governance/01_drift_monitoring.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "26_mlops_governance"

# Headless (no display)
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python 26_mlops_governance/01_drift_monitoring.py
```

## Dependencies

**Upstream chapters**:

* Chapter 11 (ML Pipeline) — registry schema (`training_runs`, `prediction_sets`, `backtest_runs`) consumed throughout.
* Chapter 16 (Strategy Simulation) — backtest metrics referenced by NB06's top-Sharpe panel.
* Chapter 19 (Risk Management) — drawdown and limit framework reused by §26.5.
* Chapter 25 (Live Trading Systems) — deployment verification preceding the monitoring workflow.

**Forward references**: none. Chapter 26 closes the implementation arc; Chapter 27 reflects on the systematic edge that emerges when the full pipeline operates as a coherent, continuously validated system.

**Key external libraries**: Feast (≥ 0.40) for §26.6 live integration, MLflow (3.x) for experiment tracking, scikit-learn / SciPy for K-S and PSI helpers, and the `ml4t-data` / `ml4t-diagnostic` libraries for loaders, IC/Sharpe metrics, and the case-study registry interface.

## References

- Bailey, David H., and Marcos López de Prado. 2014. ["The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality."](https://doi.org/10.2139/ssrn.2460551) SSRN.
- Bifet, Albert, and Ricard Gavaldà. 2007. ["Learning from Time-Changing Data with Adaptive Windowing."](https://doi.org/10.1137/1.9781611972771.42) *Proceedings of the 2007 SIAM International Conference on Data Mining*.
- Board of Governors of the Federal Reserve System. 2011. [*Supervisory Guidance on Model Risk Management — SR Letter 11-7*.](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
- Capponi, Agostino, Chengpiao Huang, J. Antonio Sidaoui, Kaizheng Wang, and Jiacheng Zou. 2025. ["The Nonstationarity-Complexity Tradeoff in Return Prediction."](https://doi.org/10.2139/ssrn.5980654) SSRN.
- Gama, João, Pedro Medas, Gladys Castillo, and Pedro Rodrigues. 2004. ["Learning with Drift Detection."](https://doi.org/10.1007/978-3-540-28645-5_29) *Advances in Artificial Intelligence – SBIA 2004*.
- Harvey, Campbell R., Yan Liu, and Heqing Zhu. 2016. ["…and the Cross-Section of Expected Returns."](https://doi.org/10.1093/rfs/hhv059) *Review of Financial Studies* 29 (1): 5–68.
- Hinder, Fabian, Valerie Vaquet, and Barbara Hammer. 2023. ["One or Two Things We know about Concept Drift — A Survey on Monitoring Evolving Environments."](https://doi.org/10.48550/arXiv.2310.15826) arXiv:2310.15826.
- Korn, Olaf, Philipp M. Möller, and Christian Schwehm. 2022. ["Drawdown Measures: Are They All the Same?"](https://doi.org/10.3905/jpm.2022.1.346) *The Journal of Portfolio Management* 48 (5): 104–120.
- Lopez de Prado, Marcos, Alexander Lipton, and Vincent Zoonekynd. 2025. ["How to Use the Sharpe Ratio."](https://doi.org/10.2139/ssrn.5520741) SSRN.
- Lu, Jie, Anjin Liu, Fan Dong, Feng Gu, Joao Gama, and Guangquan Zhang. 2018. ["Learning under Concept Drift: A Review."](https://doi.org/10.1109/TKDE.2018.2876857) *IEEE Transactions on Knowledge and Data Engineering*.
- Lundberg, Scott M., and Su-In Lee. 2017. ["A Unified Approach to Interpreting Model Predictions."](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf) *NeurIPS 30*.
- McLean, R. David, and Jeffrey Pontiff. 2016. ["Does Academic Research Destroy Stock Return Predictability?"](https://doi.org/10.1111/jofi.12365) *Journal of Finance* 71 (1): 5–32.
- Paleyes, Andrei, Raoul-Gabriel Urma, and Neil D. Lawrence. 2023. ["Challenges in Deploying Machine Learning: A Survey of Case Studies."](https://doi.org/10.1145/3533378) *ACM Computing Surveys* 55 (6): 1–29.
- Sculley, D., Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, and Dan Dennison. 2015. ["Hidden Technical Debt in Machine Learning Systems."](https://proceedings.neurips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) *NeurIPS 28*.
- Studer, Stefan, Thanh Binh Bui, Christian Drescher, Alexander Hanuschkin, Ludwig Winkler, Steven Peters, and Klaus-Robert Müller. 2021. ["Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology."](https://doi.org/10.3390/make3020020) *Machine Learning and Knowledge Extraction* 3 (2): 392–413.
- Varma, Samir. 2025. ["The False Promise of Drawdown Rules: New Evidence and a Better Framework."](https://doi.org/10.3905/jpm.2025.1.765) *The Journal of Portfolio Management* 52 (1): 145–161.
