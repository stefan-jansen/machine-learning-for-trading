# Chapter 14: Latent Factor Models

The chapter motivates the chapter by reframing the factor zoo debate as a practical modeling problem: rather than choosing from hundreds of named factors, the reader learns how to let the data extract lower-dimensional risk structure directly. It also introduces the chapter's key organizing distinction between covariance-explaining "attribution" factors and genuinely priced factors, which becomes the thread tying the statistical and economic sides of the chapter together.

## Learning Objectives

- Distinguish covariance-explaining attribution factors from priced factors, and explain why that distinction matters for prediction, risk decomposition, and trading applications
- Implement PCA on asset returns, interpret principal components as latent risk dimensions or eigenportfolios, and diagnose key practical issues including covariance noise, component selection, and loading instability
- Explain how IPCA and RP-PCA extend PCA by introducing time-varying characteristic-based betas and pricing-error penalties, and evaluate when these extensions are preferable to plain variance maximization
- Implement and evaluate Conditional Autoencoders using walk-forward validation, ensemble averaging, and interpretability diagnostics such as SHAP, while recognizing their main failure modes
- Explain how adversarial SDF estimation enforces no-arbitrage restrictions, how its objective differs from CAE reconstruction, and when direct pricing-error minimization is likely to add value
- Compare latent factor methods across datasets and modeling objectives, and choose among PCA, IPCA, RP-PCA, CAE, and SDF approaches based on dimensionality, economic goal, and evaluation design

## Sections

### 14.1 Making the Case for Latent Factors

This section motivates the chapter by reframing the factor zoo debate as a practical modeling problem: rather than choosing from hundreds of named factors, the reader learns how to let the data extract lower-dimensional risk structure directly. It also introduces the chapter's key organizing distinction between covariance-explaining "attribution" factors and genuinely priced factors, which becomes the thread tying the statistical and economic sides of the chapter together.

### 14.2 Principal Component Analysis: The Mathematical Foundation

This section gives the linear algebra core of latent factor modeling and shows why PCA remains the baseline method for extracting common variation from returns. Its value is not just in explaining eigenvalues and eigenvectors, but in showing where PCA breaks in real financial settings: noisy covariance estimation, high-dimensional panels, variance-pricing disconnects, and unstable components.

- [`01_pca_equity_sectors`](01_pca_equity_sectors.ipynb) — This notebook applies PCA to sector ETFs to extract latent risk factors and quantifies loading stability using bootstrap resampling. We demonstrate how PCA captures market and rotation factors, and how to assess whether factor loadings are statistically reliable.

### 14.3 Eigenportfolios for Equity Strategies

Here the chapter turns PCA from an abstract decomposition into something financially usable by interpreting eigenvectors as portfolio weights. The section matters because it connects latent factors to portfolio construction, risk decomposition, and stat-arb style applications, while also treating the key practical issues a practitioner actually faces: interpretability, instability, and production-grade stabilization.

- [`02_eigenportfolios`](02_eigenportfolios.ipynb) — This notebook applies PCA to the US Equities dataset (3,199 stocks) to extract latent equity risk factors and construct eigenportfolios. We demonstrate standard PCA, sector loading analysis, hierarchical PCA (HPCA), and applications to statistical arbitrage and risk decomposition.

### 14.4 The Yield Curve Decoded

This section provides the chapter's clearest example of latent factor success in practice. By showing that level, slope, and curvature explain most yield-curve variation, it gives readers an intuitive case where variance extraction and economic structure align unusually well, which helps clarify why latent factor methods are so much cleaner in fixed income than in equities.

- [`03_yield_curve_decomposition`](03_yield_curve_decomposition.ipynb) — This notebook demonstrates one of PCA's most celebrated applications: decomposing the Treasury yield curve into its three primary factors. Uses macro data.

### 14.5 Advanced Statistical Models: Bridging Economics and Data

This section introduces IPCA and RP-PCA as the chapter's first moves beyond plain variance extraction. Its real contribution is to show how latent factor estimation can be made more economically meaningful, either by allowing characteristics to determine time-varying betas or by pushing the estimation objective toward pricing relevance rather than raw covariance fit alone.

- [`04_ipca`](04_ipca.ipynb) — This notebook implements Instrumented PCA from Kelly, Pruitt, and Su (2019) "Characteristics are Covariances: A Unified Model of Risk and Return". Uses synthetic data.
- [`05_rp_pca`](05_rp_pca.ipynb) — This notebook implements Risk-Premium PCA from Lettau & Pelger (2020) "Estimating Latent Asset-Pricing Factors". Uses synthetic data.

### 14.6 Conditional Autoencoders and Deep Learning for Asset Pricing

This section is the chapter's conceptual high point, showing how deep learning extends the latent factor framework without collapsing into generic prediction. It carefully distinguishes two different goals that are often blurred in the literature: learning nonlinear factor exposures for return prediction versus directly estimating a no-arbitrage SDF. That distinction makes the deep learning material much more usable and much less buzzword-driven.

- [`06_conditional_autoencoder`](06_conditional_autoencoder.ipynb) — This notebook implements the Conditional Autoencoder (CAE) model of Gu, Kelly, and Xiu (2019) on US equities, walking through universe construction, walk-forward training, ensemble averaging, and SHAP-based interpretation. GPU-trained.
- [`07_stochastic_discount_factor`](07_stochastic_discount_factor.ipynb) — This notebook implements the adversarial moment-based Stochastic Discount Factor estimator of Chen, Pelger, and Zhu (2021), threading macro instruments into a no-arbitrage GMM-style objective on US equities. GPU-trained.

### 14.7 Building the Conditional Autoencoder

This section gives the reader an implementation path rather than just theory. It matters because it translates the CAE into an actual research workflow, covering universe definition, characteristic scaling, walk-forward validation, hyperparameter search, robustness checks, and common failure modes. That keeps the deep learning content grounded in ML4T's broader experimental discipline.

- [`06_conditional_autoencoder`](06_conditional_autoencoder.ipynb) — This notebook implements the Conditional Autoencoder (CAE) model of Gu, Kelly, and Xiu (2019) on US equities, walking through universe construction, walk-forward training, ensemble averaging, and SHAP-based interpretation. GPU-trained.
- [`08_supervised_autoencoder`](08_supervised_autoencoder.ipynb) — This notebook implements the Supervised Autoencoder architecture popularized by a winning entry in the [Jane Street Market Prediction Kaggle competition](https://www.kaggle.com/competitions/jane-street-market-prediction) (2020-2021); the [reference implementation](https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp) and [competition write-up](https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/224348) document the original architecture. Applied here to US equities to study how a shared encoder responds to a combined reconstruction + auxiliary + main-prediction loss. GPU-trained. _Runtime ~20 minutes; ~8.5 GB GPU memory — close other GPU processes first._

### 14.8 Case Study Results

This section earns the chapter's methodological range by forcing the models to compete across several datasets and label horizons. Its strongest message is that latent factor methods are conditional tools rather than universal winners: PCA works where structure is genuinely low-rank, IPCA shines in large equity panels, CAE benefits from rich cross-sectional signals, and SDF methods help most when unconstrained prediction becomes ill-conditioned.

- [`09_case_study_insights`](09_case_study_insights.ipynb) — Most datasets lack the cross-sectional breadth for reliable latent factor extraction. This notebook opens with that negative finding -- a Marchenko-Pastur dimensionality diagnostic for all 9 case studies -- then deep-dives on the case studies where latent factor models were trained.

## Running the Notebooks

Notebooks `06_conditional_autoencoder`, `07_stochastic_discount_factor`, and
`08_supervised_autoencoder` are GPU-trained (use the `ml4t-gpu` image or a local
CUDA install); the other six are CPU-only.

```bash
# From the repository root
uv run python 14_latent_factors/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "14_latent_factors"
```

| Notebook | Runtime | Peak memory | Hardware |
|---|---:|---:|---|
| `01_pca_equity_sectors` | ~25 s | ~0.9 GB | CPU |
| `02_eigenportfolios` | ~6 min | ~3.6 GB | CPU |
| `03_yield_curve_decomposition` | ~15 s | ~1.0 GB | CPU |
| `04_ipca` | ~15 s | ~0.9 GB | CPU |
| `05_rp_pca` | ~15 s | ~1.2 GB | CPU |
| `06_conditional_autoencoder` | ~3 min | ~5.2 GB | GPU |
| `07_stochastic_discount_factor` | ~10 min | ~7.8 GB | GPU |
| `08_supervised_autoencoder` | ~20 min | ~8.5 GB (close other GPU processes) | GPU |
| `09_case_study_insights` | ~20 s | ~1.5 GB | CPU |

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
