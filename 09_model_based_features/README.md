# Chapter 9: Model-Based Feature Extraction

Chapter 8 built features by aggregating observed data and applying deterministic transformations to it. This chapter builds them from **fitted procedures**: coefficients, persistence measures and latent states on one side, and filtered estimates, innovations, forecasts, conditional variances, regime probabilities and posterior uncertainty on the other. The recipe is the same throughout. Fit a procedure to the training data and use its outputs as columns.

The chapter is organized by extraction method rather than by economic family, because one fitted procedure generates several kinds of feature: a Kalman filter yields a level, a trend, an innovation and an uncertainty; a GARCH model yields a conditional volatility and a persistence parameter; a regime model yields probabilities, durations and a transition structure. What does not change across any of them is the point-in-time requirement. Every estimate, state, forecast and probability has to be computable from information available at the moment it is dated, re-estimated inside the walk-forward protocol, and versioned with the features it produced.

## Learning Objectives

* distinguish direct features from model-based features and judge when a fitted procedure adds useful information beyond
  direct transformations
* use fitted procedures to extract forecasts, filtered states, residuals, conditional volatility, regime probabilities,
  and uncertainty summaries as downstream features
* design a compact, interpretable set of model-based features from diagnostics, signal transforms, volatility models,
  uncertainty summaries, and regime models
* enforce point-in-time correctness by fitting and selecting models within training windows, using filtered rather than
  future-informed outputs, and aligning refit cadence and online updates with the walk-forward protocol
* transform asset-level temporal outputs into cross-sectional, benchmark-adjusted, pairwise, and universe-level features
  for multi-asset prediction tasks
* distinguish between exploratory time-series methods that are useful for research diagnosis and deployable features
  that are safe for live, point-in-time use
* use uncertainty and regime outputs primarily as conditioning features, and recognize when they should not be treated
  as stand-alone trading signals

## Sections

### 9.1 Diagnostics and Stationarity Features

Model-based features often begin as diagnostics, and here the diagnostics play both roles. They decide what preprocessing a series needs, and they produce quantities that go downstream as columns: a rolling stationarity statistic as a regime indicator, a detected break date as a conditioning variable, a persistence estimate as a summary of how fast shocks decay.

- [`01_visual_diagnostics`](01_visual_diagnostics.ipynb) — The diagnostic sequence for a financial series, from plotting it through stationarity tests and autocorrelation to the rolling versions of the same statistics as features. Uses etfs and macro data.
- [`02_structural_breaks`](02_structural_breaks.ipynb) — Where a series changes regime, found by the classical tests and by a change-point search, and how a break date becomes a column. Uses etfs data.
- [`03_fractional_differencing`](03_fractional_differencing.ipynb) — Differencing by a fractional order, which reaches stationarity while keeping more of the memory than a first difference does. Uses etfs data.

### 9.2 Transforming Signals to Uncover Hidden Structure

Rolling statistics discard most of what a sequence contains. A Kalman filter recovers latent state, spectral methods recover cycles, wavelets recover behavior local to a scale, and path signatures recover the geometry of the path. Each is a richer representation than a moving average, and each has to be made causal before it can be a feature.

- [`04_kalman_filter`](04_kalman_filter.ipynb) — The filter as a feature extractor: level, trend, the innovation as a surprise measure, and a hedge ratio that moves. Uses etfs data.
- [`05_spectral_features`](05_spectral_features.ipynb) — Frequency-domain features, from a wavelet decomposition through a rolling transform to a power spectral density estimate. Uses etfs data.
- [`06_path_signatures`](06_path_signatures.ipynb) — The signature of a path as a feature set, and what its terms say that a set of moments does not. Needs the `ml4t-py312` image; see the runtime note below. Uses etfs data.

### 9.3 Volatility Features

Volatility is the most forecastable part of the problem, and this section turns that predictability into columns. ARIMA residuals and forecast intervals, the GARCH family's conditional variance and persistence, HAR's horizon weights, and a roughness exponent are each a summary of persistence, asymmetry or horizon structure.

- [`07_arima_features`](07_arima_features.ipynb) — ARIMA read as a feature extractor rather than a forecaster: its residuals, its forecasts and the width of its intervals. Uses etfs data.
- [`08_garch_volatility`](08_garch_volatility.ipynb) — Conditional volatility, the persistence parameters, and the asymmetry between a rise and a fall. Uses etfs data.
- [`09_har_rough_volatility`](09_har_rough_volatility.ipynb) — Volatility at three horizons in one linear model, the range-based estimators it is built from, and the roughness exponent. Uses etfs and Nasdaq-100 minute data.

### 9.4 Uncertainty Features

A model's uncertainty is itself informative. A posterior width, a forecast standard error and an interval width separate a well identified estimate from a weakly identified one when the point forecast is the same, which is what a position size should depend on.

- [`10_uncertainty_features`](10_uncertainty_features.ipynb) — A stochastic volatility model refit forward, the sampler diagnostics that decide whether its posterior is worth reading, and the forecast interval as a column. Uses etfs data.

### 9.5 Regime Features

A changing market environment can be encoded as a feature, by a transparent threshold rule, by a hidden Markov model, or by clustering whole distributions. The message that matters is that regime information works better as soft conditioning than as a hard switch between separate models, because the regime call is least certain exactly where it matters most.

- [`11_hmm_regimes`](11_hmm_regimes.ipynb) — Regime inference from first principles, filtered against smoothed, measured against two rules that estimate nothing. Uses etfs and macro data.
- [`12_wasserstein_regimes`](12_wasserstein_regimes.ipynb) — Each window treated as a distribution and clustered under the Wasserstein distance, following Horvath et al. (2021), against a two-moment summary of the same windows. Uses S&P 500 index data.
- [`13_regime_as_feature`](13_regime_as_feature.ipynb) — The regime probability as a column against a model per regime, with the regime model refit inside every fold. Uses etfs and macro data.

### 9.6 Cross-Sectional and Panel Features

A conditional volatility of a quarter means one thing for a utility and another for a biotech. Temporal features are computed asset by asset from each series' own history, and they become more useful once ranked across a universe, measured against a benchmark, paired, or aggregated.

- [`14_panel_features`](14_panel_features.ipynb) — Cointegration and a filtered hedge ratio on one pair, then cross-sectional ranks, benchmark-relative features and universe aggregates on a panel. Uses etfs data.

### 9.7 Summary

- [`case_study_temporal_summary`](case_study_temporal_summary.ipynb) — What the nine case studies' model-based feature stages wrote, read from the artifact schemas, and what a schema cannot tell you about it. Uses case study artifacts.

## Running the Notebooks

```bash
# From the repository root
uv run python 09_model_based_features/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "09_model_based_features"
```

> Runtime, measured on a workstation: `10_uncertainty_features` about three minutes, which is the sampler; every other notebook in the chapter finishes in under 40 seconds.
>
> `06_path_signatures` requires the `ml4t-py312` Docker image, because `esig` has no Python 3.14 wheel:
>
> ```bash
> docker compose --profile py312 run --rm py312 \
>     python 09_model_based_features/06_path_signatures.py
> ```

## References

- **Andrew Ang and Geert Bekaert** (2002). [International Asset Allocation With Regime Shifts](https://doi.org/10.1093/rfs/15.4.1137). *Review of Financial Studies*.
- **Andrew Ang and Allan Timmermann** (2011). [Regime Changes and Financial Markets](https://doi.org/10.2139/ssrn.1919497).
- **Michael Betancourt** (2018). [A Conceptual Introduction to Hamiltonian Monte Carlo](http://arxiv.org/abs/1701.02434). *arXiv:1701.02434 [stat]*.
- **Tim Bollerslev** (1986). [Generalized autoregressive conditional heteroskedasticity](https://doi.org/10.1016/0304-4076(86)90063-1). *Journal of Econometrics*.
- **Ilya Chevyrev et al.** (2026). [A Primer on the Signature Method in Machine Learning](https://doi.org/10.1007/978-3-031-97239-3_1). *Springer Nature Switzerland*.
- **Fulvio Corsi** (2009). [A Simple Approximate Long-Memory Model of Realized Volatility](https://doi.org/10.1093/jjfinec/nbp001). *Journal of Financial Econometrics*.
- **Robert F. Engle** (1983). [Estimates of the Variance of U. S. Inflation Based upon the ARCH Model](https://doi.org/10.2307/1992480). *Journal of Money, Credit and Banking*.
- **Robert F. Engle and C. W. J. Granger** (1987). [Co-Integration and Error Correction: Representation, Estimation, and Testing](https://doi.org/10.2307/1913236). *Econometrica*.
- **Mark B. Garman and Michael J. Klass** (1980). [On the Estimation of Security Price Volatilities from Historical Data](https://www.jstor.org/stable/2352358). *The Journal of Business*.
- **Jim Gatheral et al.** (2014). [Volatility is rough](https://doi.org/10.48550/arXiv.1410.3394).
- **James D. Hamilton** (1989). [A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle](https://doi.org/10.2307/1912559). *Econometrica*.
- **Matthew D. Hoffman and Andrew Gelman** (2011). [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](http://arxiv.org/abs/1111.4246). *arXiv:1111.4246 [cs, stat]*.
- **Blanka Horvath et al.** (2021). [Clustering Market Regimes Using the Wasserstein Distance](https://doi.org/10.2139/ssrn.3947905).
- **Søren Johansen and Katarina Juselius** (1990). [Maximum Likelihood Estimation and Inference on Cointegration — with Applications to the Demand for Money](https://doi.org/10.1111/j.1468-0084.1990.mp52002003.x). *Oxford Bulletin of Economics and Statistics*.
- **Stephen Marra** (2023). [Time-Series Techniques: Estimating Volatility](https://doi.org/10.3905/jpm.2023.1.475). *The Journal of Portfolio Management*.
- **Alan Moreira and Tyler Muir** (2017). [Volatility-Managed Portfolios](https://doi.org/10.1111/jofi.12513). *The Journal of Finance*.
- **Daniel B. Nelson** (1991). [Conditional Heteroskedasticity in Asset Returns: A New Approach](https://doi.org/10.2307/2938260). *Econometrica*.
- **Marcos Lopez de Prado** (2018). Advances in Financial Machine Learning. *John Wiley & Sons*.
- **Yizhan Shu and John M. Mulvey** (2025). [Dynamic Factor Allocation Leveraging Regime-Switching Signals](https://doi.org/10.3905/jpm.2024.1.649). *The Journal of Portfolio Management*.
- **Sophia Sun and Rose Yu** (2025). [Conformal Prediction for Time-series Forecasting with Change Points](https://doi.org/10.48550/arXiv.2509.02844).
- **A. Sinem Uysal and John M. Mulvey** (2021). [A Machine Learning Approach in Regime-Switching Risk Parity Portfolios](https://doi.org/10.3905/jfds.2021.1.057). *The Journal of Financial Data Science*.
- **Dennis Yang and Qiang Zhang** (2000). [Drift‐Independent Volatility Estimation Based on High, Low, Open, and Close Prices](https://doi.org/10.1086/209650). *The Journal of Business*.
