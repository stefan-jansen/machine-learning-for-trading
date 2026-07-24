# Chapter 8: Financial Feature Engineering

The chapter gives the chapter its core editorial value: a disciplined way to move from a trading narrative to a feature specification. The three-step filter -- horizon alignment, driver hypothesis, and role separation -- turns feature design from indicator collecting into explicit hypothesis design, while the reference-frame, representation, and aggregation knobs make clear which choices actually change meaning and which only smooth noise.

## Learning Objectives

* Translate a trading hypothesis into a documented feature specification using horizon alignment, driver hypothesis, and role separation.
* Choose a feature's reference frame, representation, and aggregation to match the economic claim and execution horizon, and distinguish hypothesis-changing choices from noise-control choices.
* Distinguish signal features from state variables and identify when each should be used marginally, as an interaction, or as a conditioning variable.
* Design representative feature specifications across price-derived, structural and cross-instrument, and contextual data families, with explicit timing assumptions and failure modes.
* Combine signals with state variables using gating, scaling, and conditional variants, and evaluate whether the interaction adds incremental information.
* Apply point-in-time discipline to slow-moving and revised data, including reporting lags, event timing, and vintage-aware availability rules.
* Control feature-search degrees of freedom using one-knob-at-a-time exploration, within-family deduplication, and multiple-testing-aware triage.

## Sections

### 8.1 Capturing and Configuring the Economic Drivers

This section gives the chapter its core editorial value: a disciplined way to move from a trading narrative to a feature specification. The three-step filter -- horizon alignment, driver hypothesis, and role separation -- turns feature design from indicator collecting into explicit hypothesis design, while the reference-frame, representation, and aggregation knobs make clear which choices actually change meaning and which only smooth noise.

### 8.2 Price-Derived Features

This section builds the reusable feature families available from the minimum market dataset: trend, reversal, volatility, liquidity, and microstructure. Its value is not just cataloging common signals, but showing how each family encodes a specific economic claim, operates at particular horizons, and fails in recognizable ways when costs, latency, or regime shifts are ignored.

- [`01_price_volume_features`](01_price_volume_features.ipynb) — This notebook demonstrates the core feature families derived from a single asset's price and volume history. These are the workhorse features of most quantitative strategies — available for every tradeable instrument.
- [`02_microstructure_features`](02_microstructure_features.ipynb) — Microstructure features capture market dynamics invisible in daily OHLCV data. They proxy for liquidity, information flow, and execution quality.

### 8.3 Structural and Cross-Instrument Features

Here the chapter moves beyond single-series transformations to information that only appears in relationships across contracts, assets, and derivative markets. Carry, relative value, lead-lag structure, and options-implied features all expand the feature space in economically meaningful ways, and the section usefully emphasizes that construction choices such as maturity alignment, peer-set definition, and surface policy are part of the hypothesis, not implementation detail.

- [`03_structural_cross_instrument_features`](03_structural_cross_instrument_features.ipynb) — This notebook demonstrates features that require data beyond a single asset's price series: term structures, cross-instrument relationships, and derivatives-implied quantities. These encode information invisible in any individual price history.

### 8.4 Contextual and Slow-Moving Features

This section shows how fundamentals, calendars, and macro variables enter ML systems mainly as state variables that condition faster signals. Its main practical contribution is to make point-in-time correctness the central constraint, reminding readers that slow data is often more dangerous than fast data because reporting lags, revisions, and repeated values can easily create fake evidence.

- [`04_fundamentals_macro_calendar`](04_fundamentals_macro_calendar.ipynb) — Slow-moving features that condition faster signals: SEC XBRL fundamentals (value/quality factors with point-in-time ASOF alignment), FRED macro indicators (yield curve, VIX regimes, credit spreads with publication-lag handling), and calendar encodings (cyclical sin/cos, time-to-event proximity).

### 8.5 Cross-Cutting Features and the Limits of Aggregation

This section marks the conceptual boundary of the chapter. It explains when deterministic rolling transformations are enough and when hidden structure -- latent states, conditional dynamics, cycle strength, or path shape -- requires fitted models and learned representations, which sets up Chapter 9 cleanly without duplicating it.

### 8.6 Combining Features and Controlling Search

This is the chapter's second major contribution after the feature-design grammar. It shows that practical improvement often comes from signal-by-state interactions, but also that these interactions multiply degrees of freedom quickly, so gating, scaling, conditional variants, deduplication, and one-knob-at-a-time discipline are necessary to keep the search credible.

- [`05_feature_selection`](05_feature_selection.ipynb) — A feature engineering pipeline produces many candidates — different lookbacks, transforms, and interaction variants. This notebook demonstrates how to reduce that set to a focused, production-ready collection using systematic selection and deduplication.
- [`06_robustness_sensitivity`](06_robustness_sensitivity.ipynb) — A robust signal maintains performance across reasonable variations in parameters, regimes, and implementation choices. This notebook teaches how to assess robustness through parameter sweeps, regime conditioning, and signal × state interactions.
- [`07_event_studies`](07_event_studies.ipynb) — Event studies measure abnormal returns around specific events (signal triggers, macro announcements, earnings) to assess their predictive power. This is a key validation technique for trading signals.
- [`case_study_feature_summary`](case_study_feature_summary.ipynb) — Cross-case-study feature inventory: feature counts per case study, family heatmap (momentum/volatility/return everywhere; carry on futures/FX; options-implied on the options case studies), and a breadth-vs-IC view that combines best-IC-per-case-study from the registry with universe-size metadata (Fundamental Law: IR ≈ IC × √BR).

## Running the Notebooks

```bash
# From the repository root
uv run python 08_financial_features/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "08_financial_features"
```

> Memory: `03_structural_cross_instrument_features` peaks at ~7.4 GB RSS scanning the AlgoSeek S&P-500 options surface — recommend ≥8 GB system RAM for §8.3.

## References

- **Yakov Amihud** (2002). [Illiquidity and stock returns: cross-section and time-series effects](https://doi.org/10.1016/S1386-4181(01)00024-6). *Journal of Financial Markets*.
- **Andrew Ang and Allan Timmermann** (2011). [Regime Changes and Financial Markets](https://doi.org/10.2139/ssrn.1919497).
- **Clifford S. Asness et al.** (2013). [Value and Momentum Everywhere](https://www.jstor.org/stable/42002613). *The Journal of Finance*.
- **Peter Carr and Liuren Wu** (2009). [Variance Risk Premiums](https://doi.org/10.1093/rfs/hhn038). *The Review of Financial Studies*.
- **Rama Cont et al.** (2014). [The Price Impact of Order Book Events](https://doi.org/10.1093/jjfinec/nbt003). *Journal of Financial Econometrics*.
- **David Easley et al.** (2021). [Microstructure in the Machine Age](https://doi.org/10.1093/rfs/hhaa078). *The Review of Financial Studies*.
- **Eugene F. Fama and Kenneth R. French** (1992). [The Cross-Section of Expected Stock Returns](https://doi.org/10.1111/j.1540-6261.1992.tb04398.x). *The Journal of Finance*.
- **Mark B. Garman and Michael J. Klass** (1980). [On the Estimation of Security Price Volatilities from Historical Data](https://www.jstor.org/stable/2352358). *The Journal of Business*.
- **Campbell R. Harvey et al.** (2016). [...and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059). *Review of Financial Studies*.
- **Zura Kakushadze et al.** (2015). [101 Formulaic Alphas](https://doi.org/10.2139/ssrn.2701346).
- **Albert S. Kyle** (1985). [Continuous Auctions and Insider Trading](https://doi.org/10.2307/1913210). *Econometrica*.
- **Ari Levine and Lasse Heje Pedersen** (2016). [Which Trend Is Your Friend?](https://doi.org/10.2469/faj.v72.n3.3). *Financial Analysts Journal*.
- **Giuseppe A. Paleologo** (2025). The Elements of Quantitative Investing. *John Wiley & Sons*.
- **Joseph D. Piotroski** (2000). [Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers](https://doi.org/10.2307/2672906). *Journal of Accounting Research*.
- **Marcos Lopez de Prado** (2018). Advances in Financial Machine Learning. *John Wiley & Sons*.
- **Dennis Yang and Qiang Zhang** (2000). [Drift‐Independent Volatility Estimation Based on High, Low, Open, and Close Prices](https://doi.org/10.1086/209650). *The Journal of Business*.
