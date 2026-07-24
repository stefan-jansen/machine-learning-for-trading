# Chapter 15: Causal Machine Learning

Chapter 7 introduced causal thinking as a falsification filter: directed acyclic graphs to encode mechanism assumptions, the structural roles of confounders, mediators, and colliders, and a set of single-feature plausibility checks. Those diagnostics are bivariate — they test one feature at a time and cannot detect multivariate confounding. This chapter supplies the multivariate estimation machinery that surviving features require.

The methods here answer harder questions than prediction does: how large is a treatment effect after orthogonalizing against high-dimensional confounders, did a discrete event actually move prices relative to a data-driven counterfactual, and which variables in a multivariate system drive which others at what lags. Each question demands a different estimator — Double Machine Learning for continuous treatments, Bayesian Structural Time-Series for discrete events, and causal discovery for structure learning — and each estimator rests on assumptions that must be stated and tested.

The chapter's recurring lesson is that a sophisticated estimator cannot rescue an identification failure. Credible causal work begins with treatment, outcome, estimand, and adjustment-set discipline, runs through validation and refutation, and ends with a clear-eyed reading of which claims survive scrutiny and which collapse.

## Learning Objectives

- Define a causal research question in terms of treatment, outcome, estimand, and counterfactual, and use DAGs to justify an admissible adjustment set
- Apply validation and refutation tools — placebo tests, sensitivity analysis, and subset-stability checks — to assess whether a causal claim is robust enough to inform trading research
- Use Double Machine Learning to estimate causal effects of continuous treatments in the presence of high-dimensional confounders, while respecting temporal cross-fitting and pre-treatment timing discipline
- Use Bayesian Structural Time-Series to estimate the impact of discrete events by constructing data-driven counterfactuals and evaluating spillover risk in control series
- Use causal discovery methods such as PCMCI, NOTEARS, and VAR-LiNGAM to generate candidate structures and interpret them as hypotheses requiring further validation rather than as definitive causal truth
- Distinguish predictive signal from causal effect, and interpret cross-dataset evidence with attention to confounding bias, multiple testing, and the gap between statistical significance and refutation survival

## Sections

### 15.1 From Theory to Estimation

Bridges the qualitative DAG assessment from Section 7.5 to quantitative estimation, and maps three distinct causal questions — continuous treatment effects, discrete-event impact, and structure discovery — to the estimators and Python libraries that address each.

- [`01_library_overview`](01_library_overview.ipynb) — Decision guide for choosing among EconML, DoWhy, CausalML, Tigramite, and causal-learn by question type, treatment type, and data structure. Demonstrates the core API patterns on synthetic data with a known true ATE so the libraries can be compared on common ground.

### 15.2 Identification and Validation

Establishes the shared framework that precedes every estimator: specifying treatment, outcome, and estimand; using DAGs and the backdoor criterion to choose admissible adjustment sets; and selecting among alternative identification designs (instrumental variables, difference-in-differences, regression discontinuity, and event-study counterfactuals) when backdoor adjustment is not credible.

### 15.3 Validation and Refutation

Develops the diagnostic workflow that any causal claim must survive — placebo and negative-control tests, sensitivity to omitted confounding, and stability across samples, specifications, and outcomes — and stresses that these checks can weaken a claim but never prove it.

- [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb) — Walks the full DoWhy validation workflow on the crypto funding-rate setting, comparing the same extreme-premium treatment against two outcomes (forward returns vs. premium reversion). Uses backdoor estimation, OLS+HAC inference, train/test splits with horizon purge, and the complete refutation battery, showing how outcome choice determines causal credibility.

### 15.4 Isolating Factor Effects with DML

Explains the residual-on-residual intuition, cross-fitting, and orthogonal scores behind Double Machine Learning, then applies it to ETF and crypto momentum, regime-conditional position sizing, and factor-zoo validation — repeatedly showing that apparent factor relationships shrink, move, or disappear once confounding is addressed.

- [`03_econml_dml`](03_econml_dml.ipynb) — Estimates the causal effect of skip-recent 6-1 momentum on 21-day forward ETF returns. Compares naive OLS against DML (EconML LinearDML and a manual implementation), demonstrates HAC standard-error inflation, walk-forward cross-fitting with embargo, block-permutation refutation, and nuisance-model sensitivity across linear and gradient-boosting learners.
- [`04_dml_crypto_regime`](04_dml_crypto_regime.ipynb) — Applies DML to the crypto perpetual-funding premium z-score on 8-hour forward returns, with subgroup effects by volatility regime and a single-model regime-interaction test. Demonstrates market-wide regime construction, block-permutation refutation across multiple block sizes, and the EconML W-vs-X distinction between controls and effect modifiers.
- [`05_momentum_causal_trading`](05_momentum_causal_trading.ipynb) — Uses CausalForestDML to estimate regime-conditional CATEs and tests whether causal position sizing improves out-of-sample performance against naive and heuristic baselines under transaction costs. Shows that causal scaling reduces degradation when training-period heterogeneity persists but cannot rescue a signal that inverts out of sample.
- [`11_factor_zoo_validation`](11_factor_zoo_validation.ipynb) — Applies post-double-selection LASSO (Belloni-Chernozhukov-Hansen 2014; Feng, Giglio, and Xiu 2020) to test which factors retain marginal pricing power for a held-out broad-market ETF. Building the zoo from PCA factors plus managed-portfolio candidates, it shows how the held-out outcome breaks the tautology that arises when the outcome lies in the column span of the controls.
- [`02_dowhy_causal_graph`](02_dowhy_causal_graph.ipynb) — Also supports this section's outcome-choice discussion through its two-outcome refutation comparison.

### 15.5 Measuring Event Impact with Bayesian Structural Time-Series

Turns from continuous exposures to discrete events, framing event studies as counterfactual-construction problems. Uses FOMC announcements to show both what BSTS can reveal and how it fails when controls are contaminated by spillover.

- [`06_fed_announcement_bsts`](06_fed_announcement_bsts.ipynb) - A BSTS event study of four FOMC announcements on the IEF Treasury ETF. Daily log returns give the cumulative effect return units. The notebook pairs `tfcausalimpact` with per-event control-as-target and placebo checks; these diagnostics do not flag the corrected run, while daily event timing and possible weak control spillover keep the estimates model-dependent.

### 15.6 Causal Discovery from Observational Data

Addresses the harder setting where the causal structure is unknown, introducing PCMCI, NOTEARS, VAR-LiNGAM, and Granger causality as hypothesis generators rather than truth machines, and emphasizing how strongly their conclusions depend on their assumptions.

- [`07_tigramite_time_series`](07_tigramite_time_series.ipynb) — Demonstrates PCMCI on a four-asset macro panel (GLD, IEF, SPY, VIX), testing lagged links with ParCorr independence and confirming with block-bootstrap stability, contrasted against pairwise Granger causality. Reports a null result: no lagged edge is stable above the robustness threshold.
- [`08_neural_causal_discovery`](08_neural_causal_discovery.ipynb) — Runs NOTEARS, VAR-LiNGAM (from scratch and via causal-learn), PCMCI, and Granger causality on a shared seven-ETF panel, enabling direct comparison on identical data. Includes synthetic-DGP validation, block-bootstrap stability, and effect-size interpretation, illustrating how method outputs diverge from zero to dozens of edges.
- [`09_adia_causal_benchmark`](09_adia_causal_benchmark.ipynb) — Reproduces the ADIA Lab Causal Discovery Challenge (Olivetti et al., 2026) in a simplified setting: a supervised classifier learns variable roles from engineered causal features and is contrasted with a PC-algorithm baseline. Frames the supervised win as amortized inference under a known synthetic regime, not transferable causal competence.

### 15.7 Case Study Causal Evidence

Synthesizes DML estimates across the nine case studies, scoring each primary treatment against two gates — HAC significance and block-permutation refutation — and showing that confounding bias is pervasive and that predictive power and causal effect are distinct objects.

- [`10_case_study_insights`](10_case_study_insights.ipynb) — Queries the per-case-study causal registries to build a cross-dataset forest of DML effects with HAC confidence intervals, a confounding-bias comparison against naive OLS, the HAC-by-refutation cross-tabulation, and multi-horizon coverage. Provides the chapter's reader-facing synthesis; per-case-study deep dives live in `case_studies/{cs}/12_causal_dml.py`.

### 15.8 Summary

Distills method selection into an assumption-fragility ordering and connects causal estimates to downstream use: weighting factors by causal confidence in portfolio construction (Chapter 17) and accounting for effect uncertainty when sizing positions in risk management (Chapter 19).

## Running the Notebooks

```bash
# From the repository root
uv run python 15_causal_estimation/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_chapter_notebooks.py -v -k "15_causal_estimation"

# Headless (no display)
MPLBACKEND=Agg PLOTLY_RENDERER=json uv run python 15_causal_estimation/<notebook>.py
```

> `08_neural_causal_discovery` runs ~40 minutes end-to-end (NOTEARS block-bootstrap plus PCMCI conditional-independence search).
>
> `06_fed_announcement_bsts` requires `/opt/bsts/bin/python` in the `ml4t-py312` Docker image. The isolated interpreter preserves the image's NumPy 2 signature stack for every other notebook while satisfying `tfcausalimpact`'s NumPy 1 and pandas 2.2 requirements.

## Dependencies

**Upstream**: Chapter 7 (§7.5) introduces DAGs, structural roles, and single-feature plausibility checks that this chapter generalizes to the multivariate setting.

**Downstream**: Chapter 17 (Portfolio Construction) can weight factors by causal confidence; Chapter 19 (Risk Management) can account for effect uncertainty when sizing positions.

**Key libraries**: `econml`, `dowhy`, `tfcausalimpact`, `tigramite`, `causal-learn`, `lightgbm`, and `ml4t-diagnostic` for HAC inference and refutation utilities.

## References

- **David H. Bailey and Marcos Lopez de Prado** (2014). [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality](https://doi.org/10.2139/ssrn.2460551).
- **Kay H. Brodersen et al.** (2015). [Inferring causal impact using Bayesian structural time-series models](https://doi.org/10.1214/14-AOAS788). *The Annals of Applied Statistics*.
- **Victor Chernozhukov et al.** (2018). [Double/debiased machine learning for treatment and structural parameters](https://doi.org/10.1111/ectj.12097). *The Econometrics Journal*.
- **Aapo Hyvärinen et al.** (2010). [Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity](https://www.jmlr.org/papers/v11/hyvarinen10a.html). *Journal of Machine Learning Research*.
- **Guido W. Imbens and Joshua D. Angrist** (1994). [Identification and Estimation of Local Average Treatment Effects](https://doi.org/10.2307/2951620). *Econometrica*.
- **Emanuele Olivetti et al.** (2026). [Can AI Learn Causal Structure? Evidence from ADIA Lab's Causal Discovery Challenge](https://doi.org/10.2139/ssrn.6125566).
- **Marcos Lopez de Prado** (2022). [Causal Factor Investing: Can Factor Investing Become Scientific?](https://doi.org/10.2139/ssrn.4205613).
- **Alexander Reisach et al.** (2021). [Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game](https://proceedings.neurips.cc/paper_files/paper/2021/hash/e987eff4a7c7b7e580d659feb6f60c1a-Abstract.html). *Curran Associates, Inc.*.
- **Jakob Runge et al.** (2019). [Inferring causation from time series in Earth system sciences](https://doi.org/10.1038/s41467-019-10105-3). *Nature Communications*.
- **Ali Shojaie and Emily B. Fox** (2022). [Granger Causality: A Review and Recent Advances](https://doi.org/10.1146/annurev-statistics-040120-010930). *Annual Review of Statistics and Its Application*.
- **Peter Spirtes et al.** (2000). Causation, Prediction, and Search.
- **Xun Zheng et al.** (2018). [DAGs with NO TEARS: Continuous Optimization for Structure Learning](https://doi.org/10.48550/arXiv.1803.01422).
