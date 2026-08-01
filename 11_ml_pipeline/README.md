# Chapter 11: The ML Pipeline

The chapter explains why the chapter moves from classical econometric concerns toward predictive modeling. It shows that
in trading, unbiased parameter recovery is often less important than stable out-of-sample forecasts, especially when
features are numerous, correlated, and noisy. The payoff for the reader is a practical reason to prefer shrinkage
methods over plain OLS when the goal is signal generation rather than coefficient inference.

## Learning Objectives

- Choose between regression and classification formulations based on how predictions will be translated into trading
  decisions
- Fit leakage-safe regularized linear models, including Ridge, LASSO, Elastic Net, and logistic regression, using
  point-in-time preprocessing and standardization
- Tune and evaluate linear models with walk-forward validation, temporal buffers, and, when needed, nested
  cross-validation to reduce selection bias
- Interpret model behavior with SHAP-based diagnostics to assess feature importance, economic plausibility, and
  stability across refits
- Construct and evaluate conformal prediction intervals or prediction sets, and monitor where coverage degrades under
  non-stationary market conditions
- Use cross-case-study evidence to judge when linear models provide a strong baseline and when weak linear signal
  motivates more flexible models

## Sections

### 11.1 From Inference to Prediction

This section explains why the chapter moves from classical econometric concerns toward predictive modeling. It shows
that in trading, unbiased parameter recovery is often less important than stable out-of-sample forecasts, especially
when features are numerous, correlated, and noisy. The payoff for the reader is a practical reason to prefer shrinkage
methods over plain OLS when the goal is signal generation rather than coefficient inference.

- [`01_ols_inference`](01_ols_inference.ipynb) — This notebook shows what classical inference looks like before we leave
  it behind. Using the same ETF features and labels as the rest of Chapter 11, we fit a statsmodels OLS model and walk
  through the full inferential toolkit: coefficient significance, Gauss-Markov diagnostics, and robust standard errors.

### 11.2 Regularized Regression

This is the chapter's technical core. It introduces Ridge, LASSO, and Elastic Net as different ways to encode
assumptions about diffuse versus sparse signal, and then connects those choices to the real mechanics of deployment:
leakage-safe standardization, hyperparameter tuning, nested validation, alternative loss functions, sample weighting,
and evaluation with IC, error metrics, and turnover. Readers should care because this section turns "linear baseline"
from a textbook concept into a full research protocol that can actually survive trading use.

- [`02_regularization_paths`](02_regularization_paths.ipynb) — This notebook compares OLS, Ridge (L2), LASSO (L1), and
  Elastic Net regression for predicting 21-day forward returns on 100 ETFs. All models share the same 8-fold
  walk-forward CV from setup.yaml, ensuring apples-to-apples comparison.
- [`04_nested_cv_hpo`](04_nested_cv_hpo.ipynb) — This notebook develops a systematic approach to hyperparameter
  selection: an alpha-grid sweep that maps the regularization landscape, followed by Optuna-based single-loop and
  nested cross-validation comparisons that quantify the inflation in single-loop performance estimates.

### 11.3 Predicting Direction with Logistic Regression

This section extends the baseline from continuous return prediction to directional and class-based setups. It shows when
classification is the more natural framing, how probabilities can be converted into positions, and why calibration,
class imbalance, and turnover matter once the model output becomes a probability rather than a return forecast. For
readers, the value is practical flexibility: the chapter makes clear that the right predictive task depends on how
forecasts will be turned into trades.

- [`03_logistic_classification`](03_logistic_classification.ipynb) — This notebook applies logistic regression to
  predict the direction of 21-day forward returns (up vs down) using the same ETF features and walk-forward folds from
  02_regularization_paths. Direction prediction is often more tractable than magnitude prediction because most trading
  decisions reduce to long/short/flat.

### 11.4 Inside the Black Box: Model Interpretability with SHAP

This section argues that interpretability is part of model validation, not a cosmetic extra. It uses SHAP to connect
predictions back to features, making it possible to test whether the model is learning economically sensible
relationships, whether those relationships are stable across folds, and whether wrong high-conviction predictions point
to feature or model problems. Readers should care because the section gives them a disciplined way to distinguish
genuine signal from plausible-looking overfit.

- [`05_shap_analysis`](05_shap_analysis.ipynb) — This notebook uses SHAP (SHapley Additive exPlanations) to interpret a
  Ridge regression model trained on real ETF features from Ch8. For linear models, SHAP values decompose exactly
  into $\phi_j = \beta_j \cdot (x_j - \bar{x}_j)$, making attributions transparent and verifiable.

### 11.5 Quantifying Predictive Uncertainty

This section adds uncertainty estimation through conformal prediction, including split-conformal, adaptive conformal
inference, and conformalized quantile regression. Its importance is not just statistical: it links interval quality
directly to position sizing and risk allocation, while being honest that financial data violate exchangeability and
therefore require monitoring rather than blind trust in nominal guarantees. Readers should care because this is where
raw predictions become risk-aware forecasts.

- [`06_conformal_prediction`](06_conformal_prediction.ipynb) — This notebook demonstrates conformal prediction methods
  for generating prediction intervals with statistical coverage guarantees. Unlike classical confidence intervals that
  assume Gaussian residuals, conformal prediction provides finite-sample valid intervals under minimal assumptions (
  exchangeability).
  
### 11.6 Linear Models Across Nine Case Studies

This section broadens the chapter from method exposition to empirical judgment. It shows where linear models work well,
where they are only marginally useful, and where they largely fail, while also highlighting label sensitivity, horizon
effects, the relative strength of Ridge, and the gap between IC and net trading value once turnover enters the picture.
Readers should care because this section defines the baseline that later model chapters must beat and clarifies that
more model complexity is only justified when the linear benchmark genuinely leaves value on the table.

- [`07_case_study_insights`](07_case_study_insights.ipynb) — This notebook compares linear model results across all 9
  case studies, examining when and why regularized linear models succeed or fail across asset classes, frequencies, and
  horizons. Uses classification_metrics, coefficients, model_ic data.
- [`08_ml_backtest_intro`](08_ml_backtest_intro.ipynb) — This notebook provides a pedagogical backtest comparing
  ML-generated signals against a simple momentum baseline on the etfs case study. It demonstrates that positive IC does
  not guarantee portfolio profitability — turnover and transaction costs can destroy predictive edge.

## Running the Notebooks

```bash
# From the repository root
uv run python 11_ml_pipeline/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "11_ml_pipeline"
```

## References

- **Anastasios N. Angelopoulos and Stephen Bates** (
  2022). [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](http://arxiv.org/abs/2107.07511).
- **Rina Foygel Barber et al.** (
  2023). [Conformal prediction beyond exchangeability](https://doi.org/10.1214/23-AOS2276). *The Annals of Statistics*.
- **Leo Breiman** (2001). Statistical Modeling: The Two Cultures.
- **Gavin C Cawley and Nicola L C Talbot** (2010). On Over-ﬁtting in Model Selection and Subsequent Selection Bias in
  Performance Evaluation.
- **Peter J. Huber** (1964). [Robust Estimation of a Location Parameter](https://doi.org/10.1214/aoms/1177703732). *The
  Annals of Mathematical Statistics*.
- **I. Elizabeth Kumar et al.** (
  2020). [Problems with Shapley-value-based explanations as feature importance measures](https://doi.org/10.48550/arXiv.2002.11097).
- **Scott M Lundberg et al.** (
  2017). [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf).
  *Curran Associates, Inc.*.
- **Joseph Simonian** (
  2024). [Using Econometrics vs. Machine Learning: What, When, and How](https://doi.org/10.3905/jpm.2024.1.623). *The
  Journal of Portfolio Management*.
- **Sophia Sun and Rose Yu** (
  2025). [Conformal Prediction for Time-series Forecasting with Change Points](https://doi.org/10.48550/arXiv.2509.02844).
- **Ryan J. Tibshirani et al.** (
  2020). [Conformal Prediction Under Covariate Shift](https://doi.org/10.48550/arXiv.1904.06019).
- **Hui Zou and Trevor Hastie** (
  2005). [Regularization and Variable Selection Via the Elastic Net](https://doi.org/10.1111/j.1467-9868.2005.00503.x).
  *Journal of the Royal Statistical Society Series B: Statistical Methodology*.
