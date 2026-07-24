# Chapter 13: Deep Learning for Time Series

The chapter explains why recurrent models mattered and why they stopped being enough. It gives the reader the historical baseline, shows how LSTMs addressed vanishing gradients, and then makes the core limitation unmistakable: recurrence imposes sequential computation and still struggles with very long dependencies. A reader should care because this is the bottleneck every later architecture is trying to escape.

## Learning Objectives

* Explain why recurrent sequence models became a computational and optimization bottleneck for long-context forecasting tasks
* Compare the main temporal modeling philosophies — decomposition-based, attention-based, state-space, and strong linear baselines — and explain when each is most appropriate
* Use strong baselines and diagnostics, including linear models and walk-forward evaluation, to judge whether sequence-model complexity is warranted
* Distinguish the design logic of modern time-series Transformer variants, including PatchTST, iTransformer, and TFT, and relate those choices to multivariate structure, covariates, and forecast horizon
* Decide when a financial prediction problem should be framed as direct panel regression with sequential inputs rather than multi-step time-series forecasting
* Evaluate time-series foundation model adaptation modes for financial applications, including the implications of transfer mismatch and pretraining contamination
* Apply practical uncertainty estimation methods, including MC Dropout and deep ensembles, to support risk-aware trading decisions

## Sections

### 13.1 The Recurrent Paradigm and Its Discontents

This section explains why recurrent models mattered and why they stopped being enough. It gives the reader the historical baseline, shows how LSTMs addressed vanishing gradients, and then makes the core limitation unmistakable: recurrence imposes sequential computation and still struggles with very long dependencies. A reader should care because this is the bottleneck every later architecture is trying to escape.

- [`01_core_architectures`](01_core_architectures.ipynb) — This notebook compares foundational neural network architectures for time series forecasting: MLP, 1D-CNN, LSTM, and GRU. We evaluate training efficiency versus predictive accuracy to understand the tradeoffs that motivated newer architectures like N-BEATS and Transformers.

### 13.2 The Decomposition Philosophy: N-BEATS

This section presents N-BEATS as the structured, decomposition-first answer to the recurrent bottleneck. Its importance is not just architectural novelty but the argument that explicit trend and seasonality structure can regularize learning and improve interpretability. Readers should care because it offers a deep model that behaves more like a disciplined forecaster than a generic black box.

- [`02_nbeats_interpretable`](02_nbeats_interpretable.ipynb) — This notebook implements the N-BEATS architecture from scratch in PyTorch, focusing on its interpretable configuration that decomposes forecasts into trend and seasonality components. Uses etfs, state_dict data.

### 13.3 The Attention Revolution for Time Series

Here the chapter explains how Transformers entered forecasting and what had to change to make them work on temporal data: patching, positional encoding, and decoder design. The section matters because it shows both the promise and the hidden traps of attention in time series, especially around temporal order and leakage. Readers should care because many modern forecasting claims rest on these design choices.

### 13.4 The Great Debate: When Simple Outperforms Complex

This is the chapter's intellectual reset. It explains why the Zeng et al. critique mattered, how simple linear baselines embarrassed many Transformer variants, and why benchmark claims in forecasting became harder to trust. Readers should care because the section replaces architecture hype with a tougher standard: a model must beat serious simple baselines before its complexity deserves attention.

- [`03_great_debate`](03_great_debate.ipynb) — The Great Debate: Are Transformers Effective for Time Series?. Uses etfs, state_dict data.

### 13.5 The Transformer's Evolution

This section shows how the post-critique generation tried to make Transformers more time-series-aware. PatchTST, iTransformer, and TFT are presented not as generic progress but as different answers to different structural problems: local temporal patterning, cross-variate dependence, and covariate-rich forecasting. Readers should care because the section turns "use a Transformer" into a more defensible architecture-selection conversation.

- [`04_transformers`](04_transformers.ipynb) — This notebook compares two modern Transformer variants for time series: PatchTST (patching along time) and iTransformer (attention over features). Both address limitations identified in the Great Debate (Section 13.4).

### 13.6 The Full Toolkit: Alternative Architectures and Foundation Models

The chapter broadens beyond the headline debate and surveys the wider toolkit: TCNs, TSMixer, CNN encodings, hybrid models, state space models, and time-series foundation models. What makes this section valuable is that it does not treat new architectures as automatic upgrades; it ties them to context length, covariate structure, compute, transferability, and the finance-specific transfer gap. Readers should care because this is where the chapter becomes a realistic map of the current landscape rather than a narrow model catalog.

- [`05_tcn`](05_tcn.ipynb) _(~3 min, ~10 GB peak GPU RAM)_ — This notebook implements a Temporal Convolutional Network for predicting forward returns on ETFs. TCNs use dilated causal convolutions to capture long-range temporal dependencies without the sequential bottleneck of recurrent networks.
- [`06_tsmixer`](06_tsmixer.ipynb) — This notebook implements TSMixer (Google, 2023) for predicting forward ETF returns. TSMixer achieves competitive results with only MLPs by alternating between time-mixing (learning temporal patterns) and feature-mixing (learning cross-variate interactions) -- no attention or convolutions needed.
- [`07_mamba_ssm`](07_mamba_ssm.ipynb) — This notebook implements a pedagogical version of the Mamba architecture (Gu and Dao, 2023) for predicting forward ETF returns. Mamba belongs to the family of Structured State Space Models (SSMs) that process sequences with O(T) complexity -- linear in sequence length -- compared to O(T^2) for self-attention.
- [`08_cnn_image_encoding`](08_cnn_image_encoding.ipynb) — This notebook converts time series windows into 2D images using Gramian Angular Fields (GAF) and Markov Transition Fields (MTF), then classifies them with a standard CNN. The approach treats the forecasting problem as image classification, allowing convolutional networks to detect visual patterns in the encoded representations.
- [`09_foundation_models`](09_foundation_models.ipynb) _(~6 min runtime)_ — This notebook evaluates Time Series Foundation Models (TSFMs) on ETF return prediction, testing the key question from recent literature: Uses dl_dataset, etfs data.

### 13.7 A Practitioner's Framework

This is the chapter's center of gravity. It translates the architectural survey into a concrete decision process: establish baseline ladders, diagnose the problem type, choose by data and horizon, and recognize when direct panel regression is more appropriate than classical multi-step forecasting. Readers should care because this section turns theory into deployment judgment and explains why the formulation of the task can matter more than the sophistication of the model.

- [`11_library_landscape`](11_library_landscape.ipynb) — Library Landscape: Unified Forecasting with sktime. Uses etfs data.
- [`12_case_study_insights`](12_case_study_insights.ipynb) — This notebook aggregates DL results from all case studies with deep learning pipelines and compares them against Ridge, GBM, and TabM baselines from Chapters 11-12. Each case study runs per-architecture notebooks (dl_lstm.py, dl_patchtst.py, etc.) with walk-forward validation; here we synthesize those results to identify where DL adds value.

### 13.8 Quantifying Prediction Uncertainty

This section makes the bridge from prediction to action. By introducing MC Dropout, deep ensembles, and calibration concerns for pretrained models, it shows how uncertainty estimates can inform position sizing and risk control rather than remain an academic afterthought. Readers should care because a point forecast alone is not enough for trading decisions.

- [`10_uncertainty`](10_uncertainty.ipynb) — This notebook implements two complementary approaches to uncertainty estimation: MC Dropout (Gal and Ghahramani, 2016) and Deep Ensembles (Lakshminarayanan et al., 2017). Both provide confidence intervals around point predictions, essential for position sizing in trading.

### 13.9 Cross-Dataset Insights

This section aggregates the empirical evidence across the case studies in
[`12_case_study_insights`](12_case_study_insights.ipynb), and the picture is
deliberately sobering: deep learning wins clearly in only a narrow part of the
book's evidence base, simple sequence models often beat more elaborate
forecasting architectures, and strong tabular baselines remain hard to dislodge.
The notebook aggregates DL results from all case studies with deep learning
pipelines and compares them against Ridge, GBM, and TabM baselines from
Chapters 11-12. Each case study runs per-architecture notebooks (dl_lstm.py,
dl_patchtst.py, etc.) with walk-forward validation; this aggregator synthesizes
those results to identify where DL actually adds value.

## Running the Notebooks

```bash
# From the repository root
uv run python 13_dl_time_series/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "13_dl_time_series"
```

> Every chapter-13 notebook except `12_case_study_insights` trains PyTorch on the GPU (`ml4t-gpu` Docker image); `12_case_study_insights` is pure registry analytics (`ml4t` image).
>
> Memory and runtime notes (production load, NVIDIA RTX 3090):
> - `01_core_architectures`: ~5 min — sequence-length scaling sweep over 8 ETFs and 4 lookbacks
> - `04_transformers`: ~3 min, ~6 GB peak — PatchTST + iTransformer training over 388 k sequences
> - `05_tcn`: ~3 min, ~10 GB peak — full ETF panel training
> - `06_tsmixer`: ~3 min, ~7 GB peak — full ETF panel training; the time-mixing block is memory-heavy
> - `09_foundation_models`: ~6 min, ~5 GB peak — Chronos/TTM zero-shot + LSTM baseline; HuggingFace downloads on first run
> - `10_uncertainty`: ~5 min, ~9 GB peak — five LSTM ensemble members plus 50-pass MC Dropout
>
> No external API keys are required; foundation-model notebooks (`09_foundation_models`, `11_library_landscape`) cache HuggingFace artifacts on first run.

## References

- **Taha Aksu and et al.** (2025). GIFT-Eval: Benchmarking Zero-Shot Time Series Forecasting.
- **Abdul Fatir Ansari and et al.** (2025). Chronos-2: Multivariate Probabilistic Time Series Foundation Models.
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
