# Chapter 13: Deep Learning for Time Series

Chapters 11 and 12 treated prediction as a cross-sectional problem: given today's
features, forecast a future label. This chapter asks a narrower question - when does the
*order* of the observations carry signal that a cross-sectional model cannot see, and
which architectures extract it?

The chapter's answer is unusually sober for its subject. Recurrent networks pay a
sequential cost that does not buy proportional accuracy; the transformer generation that
replaced them was embarrassed by linear baselines on standard benchmarks; and across the
book's own case studies, deep learning wins clearly in a narrow part of the evidence.
The notebooks are built so that each of those claims is something you check rather than
something you are told.

## Learning Objectives

* Explain why recurrent sequence models became a computational and optimization bottleneck for long-context forecasting tasks
* Compare the main temporal modeling philosophies - decomposition-based, attention-based, state-space, and strong linear baselines - and explain when each is most appropriate
* Use strong baselines and diagnostics, including linear models and walk-forward evaluation, to judge whether sequence-model complexity is warranted
* Distinguish the design logic of modern time-series Transformer variants, including PatchTST, iTransformer, and TFT, and relate those choices to multivariate structure, covariates, and forecast horizon
* Decide when a financial prediction problem should be framed as direct panel regression with sequential inputs rather than multi-step time-series forecasting
* Evaluate time-series foundation model adaptation modes for financial applications, including the implications of transfer mismatch and pretraining contamination
* Apply practical uncertainty estimation methods, including MC Dropout and deep ensembles, to support risk-aware trading decisions

## Sections

### 13.1 Recurrent networks and their limits

The historical baseline and the bottleneck. LSTMs addressed vanishing gradients, but recurrence still imposes sequential computation and still struggles across long spans. Every later architecture in the chapter is an attempt to escape that.

- [`01_core_architectures`](01_core_architectures.ipynb) - Compares an MLP, a 1D-CNN, an LSTM and a GRU on the same panel, under a purged three-way split anchored on each example's target date. Times a single training step rather than a whole run, so the comparison is between architectures rather than between neighbours on shared hardware, and sweeps the window length with the scoring dates held fixed.

### 13.2 N-BEATS and explicit decomposition

N-BEATS answers the recurrent bottleneck with structure: basis functions that make trend and seasonality explicit parts of the forecast rather than something the network may or may not learn.

- [`02_nbeats_interpretable`](02_nbeats_interpretable.ipynb) - Builds N-BEATS from scratch in PyTorch in both its generic and interpretable configurations, and reads the decomposition for what it is - what a block was *allowed* to say while helping the forecast, not a finding about the series. Shows that the held-back partition lies entirely outside the training range, and that the final block's backcast head receives no gradient at all.

### 13.3 Attention for time series

How transformers entered forecasting and what had to change to make them work on temporal data: patching, positional encoding, decoder design. The section matters as much for the traps as for the promise.

### 13.4 Linear baselines versus transformers

The chapter's intellectual reset. Zeng and co-authors showed that simple linear models embarrassed a generation of transformer variants, and the section replaces architecture enthusiasm with a tougher standard: beat a serious simple baseline before complexity earns attention.

- [`03_great_debate`](03_great_debate.ipynb) - Puts Linear, D-Linear and N-Linear against a vanilla transformer on daily ETF returns, and runs the shuffle diagnostic that made the original critique sharp - measuring both the change in error and the distance the predictions moved, because the first alone cannot tell you whether a model used the ordering.

### 13.5 Modern transformer variants

The post-critique generation, presented as different answers to different structural problems rather than as progress: local patterning, cross-variate dependence, covariate-rich forecasting.

- [`04_transformers`](04_transformers.ipynb) - PatchTST makes a token a short run of days; iTransformer makes a token a whole feature's history. Both change what a token *is* rather than making the model bigger. Reads the iTransformer's attention per head rather than averaged, and states what an attention weight can and cannot tell you.

### 13.6 Alternative architectures and foundation models

Beyond the headline debate: convolution, mixing, state space models, image encodings and pretrained forecasters. Not a catalogue of upgrades - each is tied to context length, covariate structure, compute, and the finance-specific transfer gap.

- [`05_tcn`](05_tcn.ipynb) - A temporal convolutional network: causal convolution by padding and trimming, dilation doubling per layer, residual blocks. Separates what the trim actually enforces from what keeps the target out of the input, which is the windowing.
- [`06_tsmixer`](06_tsmixer.ipynb) - Drops attention and convolution and keeps only dense layers, applied along one axis at a time. The cost argument is derived and the printed parameter count lets a reader check it.
- [`07_mamba_ssm`](07_mamba_ssm.ipynb) - A selective state space model written as a readable Python loop, with its four departures from the reference implementation named and locatable in the code. Separates what makes the sweep linear-time from what makes it parallelisable.
- [`08_cnn_image_encoding`](08_cnn_image_encoding.ipynb) - Turns each window into a Gramian angular field and a Markov transition field and hands the pair to an image CNN. Both encodings normalise inside the window, so the level and range of the returns are gone before the network sees anything - which is stated because the label is a return.
- [`09_foundation_models`](09_foundation_models.ipynb) - Runs Chronos and TinyTimeMixer zero-shot against an LSTM and a ridge fitted on the panel, all four reading the same univariate context. Says what a rank-IC comparison between a forecast of one quantity and a label of another can support, and names pretraining as a leakage channel no temporal split can inspect.

### 13.7 A practical framework

Turns the survey into a decision process: baseline ladders, problem diagnosis, and the recognition that a task's formulation often matters more than the model's sophistication.

- [`11_library_landscape`](11_library_landscape.ipynb) - Writes the same univariate forecast through raw PyTorch, sktime and Darts, and compares the implementation experience. The accuracy columns cannot be compared across rows and the notebook says so; the last-value baseline is included because it is the one comparison that is like-for-like, and it wins.
- [`12_case_study_insights`](12_case_study_insights.ipynb) - Reads the registry rather than training anything: LSTM, NLinear, TSMixer, TCN and PatchTST across whichever case studies carry deep-learning pipelines, against the linear, gradient-boosted and TabM baselines of Chapters 11 and 12. Every comparison is restricted to configurations covering the same folds and days.

### 13.8 Quantifying prediction uncertainty

The bridge from prediction to action. A point forecast alone cannot size a position; two forecasts with the same expected return and different confidence do not deserve the same exposure.

- [`10_uncertainty`](10_uncertainty.ipynb) - MC Dropout and deep ensembles produce a spread; neither produces a coverage property, which the empirical coverage table makes visible. Split-conformal calibration is the step that converts one into the other, under an exchangeability assumption this setup violates twice - both violations stated where the quantiles are computed.

### 13.9 Case study insights

The aggregate picture from [`12_case_study_insights`](12_case_study_insights.ipynb), and it is deliberately sobering: deep learning wins clearly in only a narrow part of the book's evidence base, simple sequence models often beat more elaborate forecasting architectures, and strong tabular baselines remain hard to dislodge.

## Running the Notebooks

```bash
# From the repository root
uv run python 13_dl_time_series/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "13_dl_time_series"
```

> Every chapter-13 notebook except `12_case_study_insights` trains PyTorch on the GPU
> (`ml4t-gpu` Docker image); `12_case_study_insights` reads the registry and needs only
> the `ml4t` image.
>
> Wall time and peak process memory, measured on this machine (NVIDIA RTX 3090). The
> memory figure is host RSS for the largest process, not GPU memory.
>
> | Notebook | Wall time | Peak RSS |
> |---|---|---|
> | `01_core_architectures` | 7 min | 2.0 GB |
> | `02_nbeats_interpretable` | 51 s | 1.5 GB |
> | `03_great_debate` | 38 s | 1.6 GB |
> | `04_transformers` | 2 min 13 s | 4.9 GB |
> | `05_tcn` | 50 s | 4.9 GB |
> | `06_tsmixer` | 55 s | 4.9 GB |
> | `07_mamba_ssm` | 2 min 39 s | 4.9 GB |
> | `08_cnn_image_encoding` | 34 s | 3.5 GB |
> | `09_foundation_models` | 3 min 33 s | 3.1 GB |
> | `10_uncertainty` | 3 min 49 s | 4.8 GB |
> | `11_library_landscape` | 23 s | 2.2 GB |
> | `12_case_study_insights` | 16 s | 1.2 GB |
>
> No API keys are required. `09_foundation_models` and `11_library_landscape` download
> HuggingFace checkpoints on their first run and cache them afterwards.
>
> `12_case_study_insights` reads each case study's `case_studies/<cs>/run_log/registry.db`
> and expects that study's per-architecture training notebooks (`dl_lstm.py`,
> `dl_nlinear.py`, `dl_tsmixer.py`, `dl_tcn.py`, `dl_patchtst.py`) to have populated it.
> A case study without eligible runs shows up as a blank row rather than as a failure.

## References

- **Taha Aksu and et al.** (2025). GIFT-Eval: Benchmarking Zero-Shot Time Series Forecasting.
- **Abdul Fatir Ansari and et al.** (2025). Chronos-2: Multivariate Probabilistic Time Series Foundation Models.
- **Si-An Chen et al.** (2023). [TSMixer: An All-MLP Architecture for Time Series Forecasting](https://arxiv.org/abs/2303.06053).
- **Yarin Gal and Zoubin Ghahramani** (2016). [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.html). *PMLR*.
- **Albert Gu and Tri Dao** (2024). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://doi.org/10.48550/arXiv.2312.00752).
- **Sepp Hochreiter et al.** (2001). Gradient flow in recurrent nets: the difficulty of learning long-term dependencies.
- **Yuan Hu and et al.** (2025). FinMamba: Market-Aware Mamba for Stock Movement Prediction.
- **Jingwen Jiang et al.** (2020). [(Re-)Imag(in)ing Price Trends](https://doi.org/10.2139/ssrn.3756587).
- **Balaji Lakshminarayanan et al.** (2017). [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html). *Curran Associates, Inc.*.
- **Bryan Lim and Stefan Zohren** (2021). [Time-series forecasting with deep learning: a survey](https://doi.org/10.1098/rsta.2020.0209). *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*.
- **Xu Liu et al.** (2024). [Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts](https://doi.org/10.48550/arXiv.2410.10469).
- **Zhiyuan Luo and et al.** (2025). Multi-Scale Mamba for Financial Time Series.
- **Yuqi Nie et al.** (2023). [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://doi.org/10.48550/arXiv.2211.14730).
- **Boris N. Oreshkin et al.** (2019). [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://openreview.net/forum?id=r1ecqn4YwB).
- **Eghbal Rahimikia et al.** (2025). [Re(Visiting) Time Series Foundation Models in Finance](https://doi.org/10.2139/ssrn.5770562).
- **Syama Sundar Rangapuram et al.** (2018). Deep State Space Models for Time Series Forecasting.
- **Slawek Smyl** (2020). A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Time Series Forecasting. *International Journal of Forecasting*.
- **Ashish Vaswani et al.** (2017). [Attention Is All You Need](http://arxiv.org/abs/1706.03762). *arXiv:1706.03762 [cs]*.
- **Ailing Zeng et al.** (2022). [Are Transformers Effective for Time Series Forecasting?](https://doi.org/10.48550/arXiv.2205.13504).
- **Zihao Zhang et al.** (2019). [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://doi.org/10.1109/TSP.2019.2907260). *IEEE Transactions on Signal Processing*.
- **Han Zou and et al.** (2025). TIME Benchmark: Fresh Datasets for Zero-Shot Forecasting Integrity.
