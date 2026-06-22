# Chapter 5: Synthetic Financial Data

The chapter explains why backtests are fragile even before any generative model enters the picture. Because strategy research is adaptive and path-limited, strong in-sample results may simply reflect favorable history and repeated search. The section then motivates synthetic data as a way to expand robustness analysis beyond one realized market path, while anchoring the discussion in the stylized facts that any useful generator must preserve.

## Learning Objectives

* Explain why trading research is path-limited and how adaptive search and multiple testing can inflate apparent backtest performance.
* Use classical simulation baselines, including bootstrap and stochastic volatility models, as interpretable benchmarks for synthetic data generation.
* Select a synthetic-data approach that matches the data structure and downstream objective, including learned generators for time series and tabular financial data.
* Diagnose generated data using stylized-fact, dependence, and task-based evaluation methods, including Train-Synthetic-Test-Real comparisons.
* Assess privacy and generator-specific risks, including leakage, bias amplification, overfitting to the generator, and limited scenario novelty.

## Sections

### 5.1 The Quant's Dilemma

This section explains why backtests are fragile even before any generative model enters the picture. Because strategy research is adaptive and path-limited, strong in-sample results may simply reflect favorable history and repeated search. The section then motivates synthetic data as a way to expand robustness analysis beyond one realized market path, while anchoring the discussion in the stylized facts that any useful generator must preserve.

### 5.2 Classical Simulation Baselines

This section gives the chapter intellectual discipline. Rather than jumping straight to deep generators, it shows that bootstrap methods, GBM, jump-diffusion, mean reversion, Heston, and GARCH remain important baselines because they are interpretable, sample-efficient, and easier to validate. Readers should care because these methods define the benchmark a learned generator ought to beat on the diagnostics that actually matter.

- [`00_classical_simulation`](00_classical_simulation.ipynb) — synthetic financial data, building the foundation for the learned generative models that follow. Uses etfs data.

### 5.3 Generative Model Taxonomy

This section provides the conceptual map for the rest of the chapter. It distinguishes discriminative from generative modeling and positions VAEs, GANs, diffusion models, and LLM-based tabular generators as alternative ways to learn joint distributions rather than hand-specify them. Its value is orientation: readers can see early that architecture choice is really about preserving the structure their downstream use case needs.

### 5.4 GANs for Financial Time Series

This is the chapter's most differentiated model survey. It moves from the basic adversarial setup to finance-specific variants such as TimeGAN, Tail-GAN, Sig-CWGAN, and GT-GAN, showing how each responds to a concrete weakness of vanilla GANs: temporal structure, tail risk, path fidelity, or irregular timestamps. The section matters because it teaches readers not to ask which GAN is best in general, but which inductive bias matches the task and its failure modes.

- [`01_timegan`](01_timegan.ipynb) — This notebook implements TimeGAN (Yoon, Jarrett & van der Schaar, NeurIPS 2019), the foundational architecture for synthetic financial time series generation. Uses multi_stock_data, state_dict, us_equities data.
- [`02_tailgan_tail_risk`](02_tailgan_tail_risk.ipynb) — This notebook implements Tail-GAN (Cont, Xu, and Zhang 2022), a GAN architecture that uses differentiable sorting to preserve tail risk characteristics (VaR, ES) in synthetic financial scenarios. Uses etf_returns, etfs, state_dict data.
- [`03_sigcwgan_signatures`](03_sigcwgan_signatures.ipynb) — > Docker required: This notebook uses signatory and esig, which are x86-only > packages not included in the default environment. Run with: > `bash > docker compose --profile py312 run --rm py312 python 05_synthetic_data/03_sigcwgan_signatures.py > ` Uses sp500_log_returns, state_dict data.
- [`04_gtgan_irregular`](04_gtgan_irregular.ipynb) — This notebook implements a GT-GAN-inspired model (based on Jeon et al., NeurIPS 2022) using Neural ODEs to handle time series with naturally irregular timestamps. Uses Chapter 3 NVDA dollar bars (Databento bar sampling).

### 5.5 Diffusion Models for Financial Time Series

This section presents diffusion models as a strong, often more stable alternative to adversarial training. It explains the denoising framework, why diffusion can fit financial return structure, how conditional guidance supports regime-aware stress testing, and why Diffusion-TS is a useful reference design for sequential financial data. Readers should care because this is the chapter's clearest candidate for a general-purpose learned generator, but one that still demands careful validation and use-case alignment.

- [`05_diffusion_ts`](05_diffusion_ts.ipynb) — This notebook implements Diffusion-TS (Yuan & Qiao, ICLR 2024), a diffusion model that decomposes the denoising prediction into trend (polynomial regression) and seasonal (Fourier basis) components. This interpretable structure encourages the model to separate slow drift from periodic patterns, analogous to classical STL decomposition but learned end-to-end within the diffusion framework.

### 5.6 LLMs for Structured Financial Data

This section extends the synthetic-data discussion beyond return series to mixed-type financial tables. By introducing serialization, the GReaT workflow, and constraint-based postprocessing, it shows where LLMs can be practical for credit, customer, and fundamental data. The key takeaway is pragmatic: LLMs can model heterogeneous schemas well, but they bring new risks around invalid rows, numerical fidelity, and privacy leakage.

- [`06_llm_tabular_great`](06_llm_tabular_great.ipynb) — This notebook implements GReaT (Generate Realistic Tabular Data) using the actual be-great library to generate synthetic financial tabular data with LLMs. Uses etf_tabular_data, etfs, from_dir data.

### 5.7 The Fidelity-Utility-Privacy Framework

This is the chapter's methodological center of gravity. It argues that validation matters more than architecture and organizes evaluation around fidelity, utility, and privacy, with TSTR as the main task-based benchmark and leakage or DP checks as privacy controls. This section matters because it turns synthetic data from a modeling curiosity into something readers can govern and assess in a research workflow.

- [`07_dp_gan`](07_dp_gan.ipynb) — This notebook demonstrates Differential Privacy (DP) for training generative models using Opacus, PyTorch's official DP library. We train a GAN with formal privacy guarantees that limit information leakage about individual records.

## Running the Notebooks

```bash
# From the repository root
uv run python 05_synthetic_data/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "05_synthetic_data"
```

> Runtime callouts (cold-start, no cached checkpoint):
> - `01_timegan`: ~12 min, GPU recommended
> - `02_tailgan_tail_risk`: ~6 min cold / <30 s with cached checkpoint, GPU recommended
> - `04_gtgan_irregular`: ~6 min, GPU recommended
> - `05_diffusion_ts`: ~5 min, GPU recommended
> - `06_llm_tabular_great`: ~13 min, GPU required (distilgpt2 fine-tune)
> - `07_dp_gan`: ~8 min, GPU recommended

> `03_sigcwgan_signatures` requires the `py312` docker profile (signatory/esig are x86-only):
> ```bash
> docker compose --profile py312 run --rm py312 python 05_synthetic_data/03_sigcwgan_signatures.py
> ```
