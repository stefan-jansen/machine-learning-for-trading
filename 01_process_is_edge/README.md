# Chapter 1: The Process Is Your Edge

The chapter establishes the chapter's central claim: in trading, durable performance depends less on picking a sophisticated model than on maintaining a disciplined research process that can survive changing markets, noisy signals, and real-world frictions. It gives readers a usable vocabulary for market change, shows why recent shocks exposed fragile assumptions, and reframes ML for trading as an adaptation problem rather than a model-selection contest.

## Learning Objectives

* Distinguish structural breaks, regimes, data drift, concept drift, and online detection, and explain why static trading models degrade in changing markets
* Explain the ML4T Workflow as a research-to-production system, including its data infrastructure foundation, scoping invariants, iterative research modules, and feedback loops from live trading back to research
* Define the evidence boundary between exploration and confirmation, and explain how trial logging, sealed holdouts, and selection-aware evaluation preserve research integrity
* Describe how causal inference and generative AI fit within a disciplined trading workflow, including the main benefits they provide and the new failure modes they introduce
* Apply regime thinking, implementability checks, and monitoring logic to diagnose strategy vulnerabilities and to adapt workflow discipline across independent and institutional settings

## Sections

### 1.1 Why Process Discipline Matters

This section establishes the chapter's central claim: in trading, durable performance depends less on picking a sophisticated model than on maintaining a disciplined research process that can survive changing markets, noisy signals, and real-world frictions. It gives readers a usable vocabulary for market change, shows why recent shocks exposed fragile assumptions, and reframes ML for trading as an adaptation problem rather than a model-selection contest.

### 1.2 Introducing the ML4T Workflow

This section presents the book's core framework: a research-to-production workflow built on point-in-time-correct data infrastructure, explicit scoping rules, iterative feature and model development, realistic strategy design, deployment discipline, and ongoing monitoring. The key value for readers is that it turns trading research into a managed lifecycle with auditable artifacts, clear handoffs, and an explicit boundary between exploration and confirmation.

### 1.3 Causal Inference and Generative AI in the Workflow

This section places two modern method families inside the workflow rather than treating them as standalone trends. Causal inference is framed as a way to sharpen mechanisms, assumptions, and diagnosis; generative AI is framed as a way to expand research and unstructured-data processing while also creating new risks such as leakage, hallucination, and workflow bloat. Readers should care because the section makes clear that new tools increase the value of discipline rather than replacing it.

### 1.4 Market Regimes: Change Is the Constant

This section turns non-stationarity into something operational. It shows how regime concepts can support explanation, robustness checks, and live monitoring, while insisting that regimes are primarily a risk lens rather than a reliable timing signal. The factor and macro examples make the idea concrete: regime methods are useful when they help identify adverse environments and connect them to predefined risk actions.

- [`factor_regimes`](factor_regimes.ipynb) — Demonstrates unsupervised learning for market regime detection using Gaussian Mixture Models (GMM) on factor returns from the AQR Century of Factor Premia dataset.
- [`macro_regimes`](macro_regimes.ipynb) — Demonstrates unsupervised learning for market regime detection using macroeconomic indicators from FRED, validated against S&P 500 volatility and drawdowns.

### 1.5 In the Real World: Independent vs. Institutional

This section translates the workflow into real operating contexts. It explains how institutions benefit from built-in friction and review, while independent researchers must create their own governance through documentation, checkpoints, and explicit stop criteria. The practical payoff is strong: it helps readers see where solo practitioners are vulnerable, where they can still compete, and how reusable infrastructure compounds research quality over time.

## Running the Notebooks

```bash
# From the repository root
uv run python 01_process_is_edge/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "01_process_is_edge"
```

## References

- **Andrew Ang and Geert Bekaert** (2002). [International Asset Allocation With Regime Shifts](https://doi.org/10.1093/rfs/15.4.1137). *Review of Financial Studies*.
- **Robert D. Arnott et al.** (2018). [A Backtesting Protocol in the Era of Machine Learning](https://doi.org/10.2139/ssrn.3275654).
- **Darrell Duffie** (2020). [Still the World's Safe Haven? Redesigning the U.S. Treasury Market After the COVID-19 Crisis](https://www.brookings.edu/wp-content/uploads/2020/05/WP62_Duffie_v2.pdf).
- **David Easley et al.** (2012). [The Volume Clock: Insights into the High Frequency Paradigm](https://doi.org/10.2139/ssrn.2034858).
- **Frank J. Fabozzi et al.** (2024). [Paradigm Shift: Embracing Holism in Causal Modeling for Investment Applications](https://doi.org/10.3905/jpm.2024.51.1.159). *The Journal of Portfolio Management*.
- **Frank J. Fabozzi and Caleb C. Stenholm** (2025). [Strategic Discipline: How Asset Management Mirrors Military Operations](https://doi.org/10.3905/jpm.2025.1.769). *The Journal of Portfolio Management*.
- **Ziang Fang and Jason Moore** (2025). What AI Can (and Can't Yet) Do for Alpha.
- **Stefano Giglio et al.** (2022). [Factor Models, Machine Learning, and Asset Pricing](https://doi.org/10.1146/annurev-financial-101521-104735). *Annual Review of Financial Economics*.
- **Campbell R. Harvey et al.** (2016). [...and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059). *Review of Financial Studies*.
- **Blanka Horvath et al.** (2021). [Clustering Market Regimes Using the Wasserstein Distance](https://doi.org/10.2139/ssrn.3947905).
- **Antti Ilmanen et al.** (2021). [How Do Factor Premia Vary Over Time? A Century of Evidence](https://doi.org/10.2139/ssrn.3400998).
- **Justina Lee** (2025). [Man Group Says Agentic AI Is Now Devising Quant Trading Signals](https://www.bloomberg.com/news/articles/2025-07-10/man-group-says-agentic-ai-is-now-devising-quant-trading-signals). *Bloomberg.com*.
- **Andrew W. Lo** (2004). [The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective](https://papers.ssrn.com/abstract=602222).
- **Martin Luk** (2023). [Generative AI: Overview, Economic Impact, and Applications in Asset Management](https://doi.org/10.2139/ssrn.4574814).
- **Judea Pearl** (2019). [The seven tools of causal inference, with reflections on machine learning](https://doi.org/10.1145/3241036). *Communications of the ACM*.
- **Marcos López de Prado** (2018). The 10 Reasons Most Machine Learning Funds Fail. *The Journal of Portfolio Management*.
- **Marcos Lopez de Prado et al.** (2024). [The Case for Causal Factor Investing](https://doi.org/10.2139/ssrn.4774522).
- **Marcos López de Prado and Vincent Zoonekynd** (2025). [Correcting the Factor Mirage: A Research Protocol for Causal Factor Investing](https://doi.org/10.3905/jpm.2025.1.794). *The Journal of Portfolio Management*.
- **James Ryseff et al.** (2024). [The Root Causes of Failure for Artificial Intelligence Projects and How They Can Succeed: Avoiding the Anti-Patterns of AI](https://www.rand.org/pubs/research_reports/RRA2680-1.html).
- **Bernhard Schölkopf et al.** (2021). [Towards Causal Representation Learning](https://doi.org/10.48550/arXiv.2102.11107).
- **Stefan Studer et al.** (2021). [Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology](https://doi.org/10.3390/make3020020). *Machine Learning and Knowledge Extraction*.
- **A. Sinem Uysal and John M. Mulvey** (2021). [A Machine Learning Approach in Regime-Switching Risk Parity Portfolios](https://doi.org/10.3905/jfds.2021.1.057). *The Journal of Financial Data Science*.
