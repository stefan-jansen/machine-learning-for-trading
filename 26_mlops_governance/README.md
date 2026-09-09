# Chapter 26: MLOps and Governance

A model has cleared every backtest, the deployment verification of Chapter 25 has taken the system live, and real capital is now flowing through it. Keeping it working, and failing safely when it cannot, is the next problem. Every deployed model decays: regimes shift, competitors discover similar signals, and the relationships the model learned gradually erode. The difference between a profitable trading operation and a capital-destroying one often comes down to how quickly that decay is detected and how safely the system responds.

This chapter builds the post-deployment infrastructure that separates production-grade systems from research prototypes. The central diagnostic is the failure taxonomy in Section 26.1: *technical* failures are verification problems (same inputs producing different outputs); *statistical* failures are performance problems (same outputs no longer predicting returns). Conflating them wastes time and capital. From there, the chapter builds three layers: detection (rolling metrics, drift diagnostics, online detectors), response (shadow mode, A/B, staged rollouts), and automated safety (multi-level circuit breakers). The supporting MLOps stack (feature stores, registries, CI/CD) is treated as enabling infrastructure to be right-sized to team maturity rather than adopted wholesale.

All seven notebooks operate on real artifacts from the `us_equities_panel` case study: its holdout boundary, its stored prediction streams, its feature panels, and the SQLite registry that records every training run. The circuit-breaker notebook drives its market-risk rules with SPY's own path through the 2020 selloff; only its latency stream is synthetic, because this repository holds no recorded system-latency series. Broker connectivity stays out of scope.

Every notebook binds its settings in a `parameters` cell with a Settings section stating what each one decides, so the windows, thresholds and model configurations can be changed without editing the code that reads them.

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
| 26.4 | Safe Model Updates                    | Shadow mode, then capital-capped A/B, then staged rollout, gated by explicit promotion criteria and rollback. |
| 26.5 | Circuit Breakers and Safety           | Multi-level breakers (loss, position, anomaly, infrastructure) and the closed / open / half-open lifecycle. |
| 26.6 | MLOps Infrastructure Overview        | Feature stores, data versioning, model registries and CI/CD, right-sized to operational maturity.        |
| 26.7 | Summary                               | The three-layer governance model from detection to response to automated safety.                         |

## Notebooks

### Detection (§26.2–§26.3)

| Notebook                                                            | What It Teaches                                                                                                                                                                                                                                                                       |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`01_drift_monitoring`](01_drift_monitoring.ipynb)                  | Builds the four-panel drift dashboard behind Figure 26.2 on the case study's holdout prediction stream: the population stability index per feature, the launch-against-latest prediction-score density, and the rolling information coefficient and hit rate. Separates a broken data feed from a genuine distribution change by validating the feature panel before measuring anything, and reads the stability index beside the Kolmogorov-Smirnov test rather than instead of it. Every alert threshold is a declared parameter carrying the reason it holds its value. **Requires a holdout prediction set, which this case study's registry does not yet hold.** |
| [`02_online_drift_detection`](02_online_drift_detection.ipynb)      | Runs two sequential detectors over a year of real prediction error for two linear configurations: a two-window mean-shift test on error magnitude, and a bad-day frequency monitor in the style of the Drift Detection Method. Calibrates each on a period it is not then judged over, places the alerts against a volatility proxy calibrated on the prior year, and keeps the sign when measuring how far each alert fell from the nearest turbulent episode. Asserts that the two configurations are actually different streams before comparing detector behaviour across them. |

### Response (§26.4)

| Notebook                                                       | What It Teaches                                                                                                                                                                                                                                                                                  |
|---------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`03_safe_model_rollout`](03_safe_model_rollout.ipynb)         | Runs a candidate model in shadow mode against the deployed one on real prediction artifacts, and applies a five-criterion promotion gate whose thresholds are fixed before the evaluation starts. The criteria cover a minimum Sharpe improvement rather than a positive one, a minimum observation window, and two checks that the candidate is still trading the same kind of book. The decision is read off the gate, so a rebuilt case study reports what it produced. Stops where the gate stops: A/B testing and staged capital begin only after every criterion is met. |

### Automated Safety (§26.5)

| Notebook                                                  | What It Teaches                                                                                                                                                                                                                                                                                  |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`04_circuit_breakers`](04_circuit_breakers.ipynb)        | Builds one closed / open / half-open state machine and four halt rules on top of it: drawdown from the running peak, loss within a session, consecutive losing sessions, and system latency. The half-open state is the design decision the notebook exists to make, since a breaker that closes on a timer resumes trading into whatever stopped it. A manager log records every transition rather than only the trips, which is what lets an operator reconstruct a halt afterwards. Driven by SPY's own path through the 2020 selloff, with a synthetic latency stream for the infrastructure breaker. |

### MLOps Infrastructure (§26.6)

| Notebook                                                              | What It Teaches                                                                                                                                                                                                                                                                                                                                |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`05_feast_feature_store`](05_feast_feature_store.ipynb)              | Builds the control logic a feature store automates, by hand, against the case study's own feature tables: feature views with an entity key, an event timestamp and a time to live; a point-in-time offline join with a guard that refuses to reach past the holdout; an as-of retrieval of the kind a live system issues; and a measurement of what one session of look-ahead does to the served feature vector. |
| [`05b_feast_live`](05b_feast_live.ipynb)                              | Runs Feast end to end on the same artifacts and checks its answers against the hand-written join, feature by feature. The check is the point: a feature store is configured rather than written, so a wrong timestamp field or time to live produces a store that runs and answers wrongly. Restricts the event set to keys both sources hold first, since a store carries the last value forward and an exact-key join does not. Requires `feast>=0.40`. |
| [`06_mlflow_experiments`](06_mlflow_experiments.ipynb)                | Reads the case study's registry as an experiment tracker, rebuilds a searchable run catalog from it, resolves one catalog row back to the files that make it reproducible, then logs the same catalog into MLflow and checks that its queries return the same runs in the same order. Keeps ranking separate from selection, since ordering by information coefficient describes the predictions and is not what decides deployment. |

## Running the Notebooks

```bash
# Production run (from the repo root)
uv run python 26_mlops_governance/01_drift_monitoring.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "26_mlops_governance"

# Execute a notebook and keep its outputs
uv run jupyter nbconvert --to notebook --execute --inplace \
  26_mlops_governance/01_drift_monitoring.ipynb
```

Do not set `MPLBACKEND=Agg` or `PLOTLY_RENDERER=json` when executing these notebooks. Either one suppresses the figures the executed notebook is supposed to carry.

## Dependencies

**Upstream chapters**:

* Chapter 11 (ML Pipeline): the registry schema (`training_runs`, `prediction_sets`, `backtest_runs`) consumed throughout.
* Chapter 16 (Strategy Simulation): the baseline backtest metrics `06_mlflow_experiments` reads.
* Chapter 19 (Risk Management): the drawdown and limit framework §26.5 reuses.
* Chapter 25 (Live Trading Systems): the deployment verification that precedes this monitoring workflow.

**Forward references**: none. Chapter 26 closes the implementation arc; Chapter 27 reflects on what a systematic edge looks like once the full pipeline runs as one continuously validated system.

**Key external libraries**: Feast (>= 0.40) for the §26.6 live integration, MLflow (3.x) for experiment tracking, SciPy for the Kolmogorov-Smirnov test, and the `ml4t-data` and `ml4t-diagnostic` libraries for loaders, metrics and the case-study registry interface.

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
