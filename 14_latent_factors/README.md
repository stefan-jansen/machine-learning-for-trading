# Chapter 14: Latent Factor Models

Hundreds of return predictors have been published, and the chapter does not ask which
of them to pick. It asks the prior question: given a panel of returns and
characteristics, what common structure can be recovered from the data itself, and what
can that structure be trusted to say?

Four of the six estimators share the chapter's three-stage arrangement and differ
inside Stage 1, though not all in the same way. PCA maximizes explained covariance and
RP-PCA changes that objective by adding a pricing-error penalty; IPCA instead changes
the parameterization, making loadings linear functions of characteristics, and the
conditional autoencoder makes that map nonlinear. The SDF and the supervised
autoencoder leave the arrangement altogether, one by learning the pricing object
directly and one by predicting returns end to end. Where the notebooks compare these
estimators, no single difference is available to explain the result.

The distinction the chapter keeps returning to is between explaining covariance and
pricing returns. A component can carry most of a panel's variance and none of its
expected return; a low-variance direction can be priced. Several notebooks here end
with intervals that span zero, and they say so rather than reporting the ordering of
the point estimates.

## Learning Objectives

* Distinguish covariance-explaining factors from priced factors, and explain why the difference matters for prediction, risk decomposition, and trading
* Implement PCA on a returns panel, read principal components as risk dimensions or as eigenportfolios, and diagnose covariance noise, component selection, and loading instability
* Explain how IPCA and RP-PCA extend PCA through characteristic-conditioned betas and a pricing-error penalty, and judge when either is preferable to plain variance maximization
* Implement a conditional autoencoder with validation-selected checkpoints and ensemble averaging, and separate its reconstruction quality from its forward ranking skill
* Explain how adversarial SDF estimation enforces no-arbitrage restrictions, and how that objective differs from reconstruction
* Compare latent estimators across panels and objectives, and read a comparison whose intervals overlap for what it is

## Sections

### 14.1 Making the case for latent factors

Reframes the factor zoo as a modeling problem rather than a selection problem, and
introduces the distinction between factors that explain covariance and factors that are
priced. That distinction organizes the rest of the chapter.

### 14.2 Extracting latent factors with PCA

The linear-algebra core, and the places it breaks on financial panels: noisy covariance
estimation in high dimensions, the gap between variance and pricing, and components
that move between samples.

- [`01_pca_equity_sectors`](01_pca_equity_sectors.ipynb) - PCA on sector ETF returns, with bootstrap confidence intervals on the loadings and a rolling decomposition of the factor structure. Reads the components as a market factor and a rotation factor, and asks of each loading whether it is distinguishable from zero before interpreting it.

### 14.3 Eigenportfolios for equity strategies

Reads eigenvectors as portfolio weights, which turns a decomposition into something a
risk model or a stat-arb book can use.

- [`02_eigenportfolios`](02_eigenportfolios.ipynb) - PCA on the 500 most liquid US equities, producing gross-normalized eigenportfolios, sector loading analysis, hierarchical PCA, residual persistence diagnostics and a two-speed covariance estimate. Separates sign ambiguity from genuine subspace drift in the rolling decomposition, which is the difference between a component that flipped and a component that changed, and presents the risk decomposition without a trading claim attached.

### 14.4 Decoding the yield curve

PCA's cleanest empirical success. Level, slope and curvature account for nearly all
yield-curve variation, which is why fixed income is where latent factors are least
contested.

- [`03_yield_curve_decomposition`](03_yield_curve_decomposition.ipynb) - Decomposes Treasury yield changes into three components and uses them for factor hedging. Descriptive throughout: no target, no model selection, no performance claim, and the notebook says why a train/test split does not apply to it.

### 14.5 Bridging economics and statistics with advanced models

IPCA and RP-PCA, the chapter's first two moves past plain variance extraction. One lets
characteristics determine time-varying betas; the other tilts the estimation objective
toward priced directions.

- [`04_ipca`](04_ipca.ipynb) - Alternating least squares on a synthetic panel with a known loading matrix, so subspace recovery can be checked against ground truth rather than asserted. Then passes the estimated factors through the chapter's three-stage forecasting adapter without assuming the factors are predictable.
- [`05_rp_pca`](05_rp_pca.ipynb) - Builds the modified covariance matrix and sweeps the pricing-error weight. Reports training mean fit and evaluation reconstruction separately and prints the range of each across the sweep, because the two move by very different amounts and an auto-scaled axis hides that.

### 14.6 The conditional autoencoder

The nonlinear member of the same family: the conditional-factor structure is kept and
only the linear loading map is replaced by a network.

- [`06_conditional_autoencoder`](06_conditional_autoencoder.ipynb) - A beta network over characteristics and a factor network over jointly estimated managed portfolios, trained as a contemporaneous reconstruction model, with a separate walk-forward adapter forecasting the next factor realization. Ensemble members are averaged at the asset-prediction surface, never at the loadings, because each member carries its own rotation.

### 14.7 The stochastic discount factor and the supervised autoencoder models

Two models that break the three-stage arrangement for opposite reasons: the SDF prices
directly and leaves no factor history to forecast; the supervised autoencoder predicts
directly and has no factor intermediate.

- [`07_stochastic_discount_factor`](07_stochastic_discount_factor.ipynb) - A portfolio-weight network trained against an adversarial moment network, with a separate beta network for the asset-level predictive head. Keeps factor Sharpe and pricing error, which assess the kernel, apart from rank IC, which assesses the ordering.
- [`08_supervised_autoencoder`](08_supervised_autoencoder.ipynb) - Reconstruction and two classification heads share one bottleneck. Every validation fold is purged by the longest label horizon and the test window sits behind a second embargo of the same length. Reports AUC per horizon with block-bootstrap intervals, because overlapping labels make a naive interval too narrow.

### 14.8 Case study insights

What the registered case-study evidence says when the latent estimators are put beside
the supervised families of Chapters 11 through 13.

- [`09_case_study_insights`](09_case_study_insights.ipynb) - Reads the case-study registries and trains nothing. Every ordering it reports is printed by the notebook rather than written into the prose, so a registry rebuild moves the numbers without leaving a stale sentence behind. Latent-versus-supervised differences are computed on inner-joined timestamp-entity keys, so each difference uses the same assets on the same dates.

### 14.9 Summary

The four adapter-based methods share Stages 2 and 3, and the chapter's comparisons
are validation diagnostics made after selection, not holdout tests.

## Running the Notebooks

```bash
# From the repository root
uv run python 14_latent_factors/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_chapter_notebooks.py -v -k "14_latent_factors"
```

> `06_conditional_autoencoder`, `07_stochastic_discount_factor` and
> `08_supervised_autoencoder` train on the GPU. `06_conditional_autoencoder` runs in
> the `ml4t-py312` image, the other two in `ml4t-gpu`; the remaining six are CPU-only
> and run under `ml4t`.
>
> Wall time and peak process memory, measured on this machine (NVIDIA RTX 3090). The
> memory figure is host RSS for the largest process, not GPU memory.
>
> | Notebook | Wall time | Peak RSS |
> |---|---|---|
> | `01_pca_equity_sectors` | 15 s | 1.0 GB |
> | `02_eigenportfolios` | 19 s | 2.9 GB |
> | `03_yield_curve_decomposition` | 8 s | 1.0 GB |
> | `04_ipca` | 27 s | 1.0 GB |
> | `05_rp_pca` | 12 s | 1.1 GB |
> | `06_conditional_autoencoder` | 2 min 2 s | 3.7 GB |
> | `07_stochastic_discount_factor` | 50 s | 3.7 GB |
> | `08_supervised_autoencoder` | 6 min 45 s | 4.6 GB |
> | `09_case_study_insights` | 13 s | 1.1 GB |
>
> No API keys are required. `04_ipca` generates its own panel and reads no dataset.
>
> `09_case_study_insights` reads each case study's
> `case_studies/<cs>/run_log/registry.db` together with the prediction artifacts those
> runs wrote, and expects the latent-factor pipelines to have populated them. A case
> study with no registered latent-factor row is absent from the coverage map rather
> than a failure.

## References

- **Marco Avellaneda and Jeong-Hyun Lee** (2010). [Statistical arbitrage in the US equities market](https://doi.org/10.1080/14697680903124632). *Quantitative Finance*.
- **Marco Avellaneda** (2019). [Hierarchical PCA and Applications to Portfolio Management](https://doi.org/10.48550/arXiv.1910.02310).
- **Matteo Bagnara** (2024). [Asset Pricing and Machine Learning: A critical review](https://doi.org/10.1111/joes.12532). *Journal of Economic Surveys*.
- **Francisco Barillas and Jay Shanken** (2018). [Comparing Asset Pricing Models](https://www.jstor.org/stable/26654648). *The Journal of Finance*.
- **Svetlana Bryzgalova et al.** (2025). [Forest through the Trees: Building Cross-Sections of Stock Returns](https://doi.org/10.1111/jofi.13477). *The Journal of Finance*.
- **Luyang Chen et al.** (2021). [Deep Learning in Asset Pricing](https://doi.org/10.48550/arXiv.1904.00745).
- **Andrew Y. Chen** (2024). [Most claimed statistical findings in cross-sectional return predictability are likely true](http://arxiv.org/abs/2206.15365).
- **John H. Cochrane** (2011). [Presidential Address: Discount Rates](https://doi.org/10.1111/j.1540-6261.2011.01671.x). *The Journal of Finance*.
- **Gregory Connor and Robert Korajczyk** (2009). Factor Models of Asset Returns.
- **Antoine Didisheim et al.** (2023). [Complexity in Factor Pricing Models](https://doi.org/10.3386/w31689).
- **Eugene F. Fama and Kenneth R. French** (1993). [Common risk factors in the returns on stocks and bonds](https://doi.org/10.1016/0304-405X(93)90023-5). *Journal of Financial Economics*.
- **Guanhao Feng et al.** (2020). [Taming the Factor Zoo: A Test of New Factors](https://doi.org/10.1111/jofi.12883). *The Journal of Finance*.
- **Amit Goyal** (2012). [Empirical cross-sectional asset pricing: a survey](https://doi.org/10.1007/s11408-011-0177-7). *Financial Markets and Portfolio Management*.
- **Shihao Gu et al.** (2019). [Autoencoder Asset Pricing Models](https://doi.org/10.2139/ssrn.3335536).
- **Campbell R. Harvey et al.** (2016). [...and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059). *Review of Financial Studies*.
- **Campbell R. Harvey and Yan Liu** (2019). [A Census of the Factor Zoo](https://doi.org/10.2139/ssrn.3341728).
- **Kewei Hou et al.** (2015). [Digesting Anomalies: An Investment Approach](https://doi.org/10.1093/rfs/hhu068). *The Review of Financial Studies*.
- **Kewei Hou et al.** (2020). [Replicating Anomalies](https://doi.org/10.1093/rfs/hhy131). *The Review of Financial Studies*.
- **Yoontae Hwang et al.** (2025). [Deep Learning in Asset Management: Architectures, Applications, and Challenges](https://doi.org/10.3905/jpm.2025.1.780). *The Journal of Portfolio Management*.
- **Thomas M. Idzorek et al.** (2024). [Domesticating the Factor Zoo with Economic Theory](https://doi.org/10.3905/jpm.2024.51.1.097). *The Journal of Portfolio Management*.
- **Theis Ingerslev Jensen et al.** (2022). Is There a Replication Crisis in Finance?.
- **Bryan T. Kelly et al.** (2019). [Characteristics are covariances: A unified model of risk and return](https://doi.org/10.1016/j.jfineco.2019.05.001). *Journal of Financial Economics*.
- **Bryan T. Kelly et al.** (2025). [Artificial Intelligence Asset Pricing Models](https://doi.org/10.3386/w33351).
- **Damian Kisiel et al.** (2023). [Portfolio Transformer for Attention-Based Asset Allocation](https://doi.org/10.1007/978-3-031-23492-7_6). *Springer International Publishing*.
- **Martin Lettau and Markus Pelger** (2020). [Estimating latent asset-pricing factors](https://doi.org/10.1016/j.jeconom.2019.08.012). *Journal of Econometrics*.
- **Robert B. Litterman and Josè Scheinkman** (1991). [Common Factors Affecting Bond Returns](https://doi.org/10.3905/jfi.1991.692347). *The Journal of Fixed Income*.
- **R. David McLean and Jeffrey Pontiff** (2016). [Does Academic Research Destroy Stock Return Predictability?](https://doi.org/10.1111/jofi.12365). *Journal of Finance*.
- **Giuseppe A. Paleologo** (2025). The Elements of Quantitative Investing. *John Wiley & Sons*.
- **Alexander Swade et al.** (2023). [Factor Zoo (.zip)](https://doi.org/10.2139/ssrn.4605976).
