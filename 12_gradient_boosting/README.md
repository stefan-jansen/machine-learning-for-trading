# Chapter 12: Gradient Boosting and Advanced Tabular Models

The chapter gives readers the conceptual bridge from single trees to GBMs. It explains why trees are attractive for financial data in the first place, why Random Forests solve variance but not bias, and why boosting matters because it turns sequential error correction into a practical path for learning nonlinear, threshold-driven structure that linear models miss.

## Learning Objectives

* Explain how boosting differs from bagging and why sequential error correction makes GBMs effective for financial
  tabular prediction
* Select among XGBoost, LightGBM, and CatBoost based on categorical structure, compute environment, latency needs, and
  leakage risk
* Choose appropriate GBM objectives and constraints for financial tasks, including pointwise regression, learning to
  rank, and monotonic constraints
* Tune GBMs efficiently with Optuna using pruning, multi-objective search, and time-series-aware validation
* Use TreeSHAP to analyze feature effects, interactions, instability, and drift in deployed tree-based models
* Evaluate when tabular deep learning alternatives such as TabPFN, TabM, and TabR are worth considering relative to GBMs
* Interpret cross-case-study evidence to decide when nonlinear tree models earn their added complexity relative to
  linear baselines

## Sections

### 12.1 From Decision Trees to Ensembles

This section gives readers the conceptual bridge from single trees to GBMs. It explains why trees are attractive for financial data in the first place, why Random Forests solve variance but not bias, and why boosting matters because it turns sequential error correction into a practical path for learning nonlinear, threshold-driven structure that linear models miss.

- [`01_ensemble_foundations`](01_ensemble_foundations.ipynb) — Benchmarks RF vs XGBoost/LightGBM/CatBoost on the Chen-Pelger-Zhu academic firm characteristics panel; quantifies the GBM-over-bagging advantage and the small spread across boosting libraries. _Runtime: ~6 min._

### 12.2 The Workhorse: Gradient Boosting Machines

This is the operational core of the chapter. It explains the shared boosting framework, why GBMs fit tabular financial data so well, and how XGBoost, LightGBM, and CatBoost differ in regularization, speed, categorical handling, GPU behavior, and deployment tradeoffs. It also moves beyond library comparison into decisions practitioners actually face: when ranking matters more than point forecasts, when monotonic constraints improve robustness, and why native feature importance is not enough.

- [`02_gbm_comparison`](02_gbm_comparison.ipynb) — Systematic GBM benchmark: 4 libraries × 3 presets × CPU/GPU on the ETF panel, plus a 5M-row scale benchmark, learning-to-rank and monotone-constraint demos. _Runtime: ~45 min — Memory: ~52 GB peak; GPU recommended._

### 12.3 Deep Learning Alternatives for Tabular Data

This section updates the reader on the changing frontier without losing the chapter's practical center of gravity. It argues that GBMs remain the default for most financial tabular problems, while showing where newer neural approaches such as TabPFN, TabM, and TabR may be worth the added complexity. The value here is not hype but a regime-based decision framework that helps readers judge when deep tabular models are genuinely additive under temporal shift and production constraints.

- [`03_dl_vs_gbm`](03_dl_vs_gbm.ipynb) — Walk-forward IC, training time, and early-stopping behavior for LightGBM, TabM, TabPFN, and a minimal torch MLP on the ETF panel. GPU recommended for TabM training.

### 12.4 Advanced Hyperparameter Tuning with Optuna

This section turns tuning from ad hoc trial and error into a disciplined optimization problem. It shows why Bayesian search is especially useful for GBMs, how Optuna's TPE and pruning mechanisms reduce wasted computation, and how multi-objective and time-series-aware tuning better reflect trading reality. The section matters because it connects model quality to validation design, compute budget, and the ever-present risk of validation overfitting.

- [`04_optuna_tuning`](04_optuna_tuning.ipynb) — Full Optuna HPO workflow for LightGBM on ETFs — single-fold vs walk-forward HPO, pruning, fANOVA importance. _Runtime: ~11 min._
- [`05_cross_library_hpo`](05_cross_library_hpo.ipynb) — Cross-library GBM HPO (XGBoost, LightGBM, CatBoost) on Chen-Pelger-Zhu firm characteristics at identical search budgets. _Runtime: ~25 min — GPU recommended._
- [`06_optuna_multi_asset`](06_optuna_multi_asset.ipynb) — Multi-objective HPO (IC vs turnover) with Pareto front, plus cross-asset hyperparameter transfer between ETFs and CME futures.
- [`07_hpo_comparison`](07_hpo_comparison.ipynb) — Grid search vs Optuna TPE on identical budget, both on a discrete grid and continuous search space. _Runtime: ~15 min._

### 12.5 Model Explainability with SHAP

This section positions explainability as part of the model workflow rather than as a post hoc accessory. It shows how TreeSHAP makes dependence analysis, interaction discovery, and drift monitoring practical for tree ensembles, while also warning against overconfidence through discussion of instability and the Rashomon effect. Readers should care because the section ties interpretation to feature pruning, uncertainty diagnostics, and production monitoring rather than to static feature-importance charts.

- [`08_shap_analysis`](08_shap_analysis.ipynb) — TreeSHAP global + local explanation, MDI/PFI/SHAP consensus, interaction decomposition, walk-forward drift, and SHAP-based feature selection on ETFs.
- [`09_xai_limitations`](09_xai_limitations.ipynb) — Demonstrates XAI instability: similar predictions can have divergent SHAP explanations (Rashomon set), and SHAP differs systematically across nominally equivalent fits.
- [`10_shap_nlp_sentiment`](10_shap_nlp_sentiment.ipynb) — SHAP token-level attribution on FinBERT sentiment classification; GPU recommended for transformer inference.
- [`11_conformal_gbm`](11_conformal_gbm.ipynb) — Conformal prediction intervals for GBM regression (split conformal, QR-conformal, CQR) with empirical coverage diagnostics.

### 12.6 GBMs Across Nine Asset Classes

This section provides the empirical payoff for the chapter. Instead of treating GBMs as abstract best practice, it shows where they help across the book's case studies, where linear models still win, which losses and tree sizes tend to work, how horizon and label design can matter as much as model class, and why validation results remain fragile without holdout confirmation. It gives the chapter its most concrete message: nonlinear models often help, but the real edge often comes from matching model, label, horizon, and evaluation design.

- [`12_case_study_insights`](12_case_study_insights.ipynb) — Cross-case-study insight notebook for the GBM family: daily-pooled IC with HAC inference, loss/depth/leaf grids, checkpoint-trajectory peak distribution, holdout decay, and Linear/GBM/TabM three-way comparison.

## Running the Notebooks

```bash
# From the repository root
uv run python 12_gradient_boosting/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "12_gradient_boosting"
```

## References

- **Tianqi Chen and Carlos Guestrin** (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD '16*.
- **Nick Erickson et al.** (2020). [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505).
- **Nick Erickson et al.** (2025). [TabArena: A Living Benchmark for Tabular Prediction](https://arxiv.org/abs/2501.02474).
- **Jerome H. Friedman** (2001). [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451). *The Annals of Statistics*.
- **Yury Gorishniy et al.** (2024). [TabR: Tabular Deep Learning Meets Nearest Neighbors](https://arxiv.org/abs/2307.14338).
- **Yury Gorishniy et al.** (2025). [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210).
- **Léo Grinsztajn et al.** (2025). [Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks](https://arxiv.org/abs/2501.02945).
- **Noah Hollmann et al.** (2025). [Accurate Predictions on Small Data with a Tabular Foundation Model](https://doi.org/10.1038/s41586-024-08328-6). *Nature*.
- **David Holzmüller et al.** (2024). [Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data](https://doi.org/10.48550/arXiv.2407.04491).
- **Guolin Ke et al.** (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). *Curran Associates, Inc.*.
- **Scott M Lundberg et al.** (2017). [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf). *Curran Associates, Inc.*.
- **Scott M. Lundberg et al.** (2020). [From Local Explanations to Global Understanding with Explainable AI for Trees](https://doi.org/10.1038/s42256-019-0138-9). *Nature Machine Intelligence*.
- **Liudmila Prokhorenkova et al.** (2019). [CatBoost: unbiased boosting with categorical features](http://arxiv.org/abs/1706.09516). *arXiv:1706.09516 [cs]*.
- **Ivan Rubachev et al.** (2024). [TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks](https://arxiv.org/abs/2406.19380).
- **Ziyu Ye et al.** (2024). [TALENT: A Tabular Analytics and Learning Toolkit](https://openreview.net/forum?id=VhiUcSmK4a).
